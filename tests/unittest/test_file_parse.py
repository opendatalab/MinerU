"""同步包装服务的参数、恢复、认证和传输故障契约。"""

from __future__ import annotations

import asyncio
import hashlib
import io
import itertools
import json
import sqlite3
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient
from filelock import Timeout
from typer.testing import CliRunner

from mineru.kit.file_parse import FileParseSettings, create_app
from mineru.kit.file_parse.app import TemporaryFileResponse
from mineru.kit.file_parse.client import TaskClient
from mineru.kit.file_parse.models import ParseOptions, Record, fingerprint
from mineru.kit.file_parse.settings import origin
from mineru.kit.file_parse.store import RecordStore
from mineru.types import Tier


def artifact_bytes(formats: list[str]) -> bytes:
    """构造可校验的不同格式附件，下载字节必须原样返回。"""
    if formats == ["html"]:
        return b"<!doctype html><html><body>file parse result</body></html>"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        name = "word/document.xml" if formats == ["docx"] else "markdown.md"
        # 固定 ZIP 时间戳，避免上游响应和预期附件跨两秒时间边界后产生不同字节。
        archive.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), "file parse result")
    return output.getvalue()


class Upstream:
    """模拟带路径前缀的 SaaS API 和独立对象存储，不提供 Jobs 或 Files 接口。"""

    def __init__(self) -> None:
        """初始化可控故障点及全部 HTTP 调用记录。"""
        self.tasks: dict[str, dict[str, Any]] = {}
        self.api_url = "https://api.test/api"
        self.storage_url = "https://store.test"
        self.calls: list[tuple[str, str, str | None]] = []
        self.complete = True
        self.cache_hit = False
        self.create_failure: str | None = None
        self.upload_response_lost = False
        self.upload_failures = 0
        self.upload_delay = 0.0
        self.ready_delay = 0.0
        self.complete_delay = 0.0
        self.download_expired = False
        self.download_corrupt = False
        self.download_interrupted = False
        self.redirect = False
        self.gone = False
        self.uploads = 0
        self.downloads = 0

    def payload(self, task_id: str) -> dict[str, Any]:
        """返回共同协议快照，签名中的转义字符用于校验地址未被重新拼装。"""
        task = self.tasks[task_id]
        ready = task["ready"] and time.monotonic() >= task["uploaded_at"] + self.ready_delay
        complete = ready and self.complete and time.monotonic() >= task["uploaded_at"] + self.complete_delay
        expired = not ready and task["expires_at"] <= time.time()
        status = "failed" if expired else "completed" if complete else "running" if ready else "waiting-file"
        result_url = f"{self.storage_url}/result/{task_id}?signature=a%2Fb&version={task['version']}" if complete else None
        if result_url and self.redirect:
            result_url = f"{self.api_url}/redirect/{task_id}"
        return {
            "task_id": task_id,
            "status": status,
            "output_formats": task["options"]["output_formats"],
            "files": [
                {
                    "file_id": f"file-{task_id}",
                    "file_name": task["options"]["files"][0]["file_name"],
                    "bytes": task["options"]["files"][0]["bytes"],
                    "status": status,
                    "upload": None
                    if task["ready"] or expired
                    else {
                        "url": f"{self.storage_url}/upload/{task_id}",
                        "headers": {"x-upload-token": "storage-token"},
                        "expires_at": task["expires_at"],
                    },
                    "result_url": result_url,
                    "error": {"code": "upload_expired"} if expired else None,
                }
            ],
        }

    async def handle(self, request: httpx.Request) -> httpx.Response:
        """执行清单、上传、查询和下载，并在每一跳检查认证边界。"""
        self.calls.append((request.method, str(request.url), request.headers.get("authorization")))
        if origin(str(request.url)) == origin(self.api_url):
            assert request.headers.get("authorization") == "Bearer secret"
        else:
            assert origin(str(request.url)) == origin(self.storage_url)
            assert request.headers.get("authorization") is None
        if request.method == "POST":
            assert request.url.path == "/api/v1/tasks"
            if self.create_failure == "auth":
                return httpx.Response(401, json={"error": {"code": "invalid_api_key", "message": "Bad upstream key"}})
            body = json.loads(await request.aread())
            assert len(body["files"]) == 1
            assert {"client_file_id", "file_id"}.isdisjoint(body["files"][0])
            task_id = f"task_{len(self.tasks) + 1}"
            self.tasks[task_id] = {
                "options": body,
                "ready": self.cache_hit,
                "uploaded_at": time.monotonic(),
                "version": 0,
                "expires_at": int(time.time()) + 3600,
            }
            if self.create_failure == "lost":
                raise httpx.ReadError("creation response lost")
            if self.create_failure == "timeout":
                raise httpx.ReadTimeout("creation response timed out")
            payload = self.payload(task_id)
            if self.create_failure == "malformed":
                payload["task_id"] = None
            return httpx.Response(
                202,
                json={
                    "task_id": payload["task_id"],
                    "status": payload["status"],
                    "files": [
                        {
                            key: value
                            for key, value in payload["files"][0].items()
                            if key in {"file_id", "file_name", "status", "upload"} and value is not None
                        }
                    ],
                },
            )
        task_id = request.url.path.rsplit("/", 1)[-1]
        task = self.tasks[task_id]
        if request.method == "PUT":
            self.uploads += 1
            assert request.headers["x-upload-token"] == "storage-token"
            content = await request.aread()
            assert len(content) == task["options"]["files"][0]["bytes"]
            if self.upload_failures:
                self.upload_failures -= 1
                return httpx.Response(503)
            await asyncio.sleep(self.upload_delay)
            task["ready"] = True
            task["uploaded_at"] = time.monotonic()
            return httpx.Response(503 if self.upload_response_lost else 200)
        if "/v1/tasks/" in request.url.path:
            return httpx.Response(404) if self.gone else httpx.Response(200, json=self.payload(task_id))
        if "/redirect/" in request.url.path:
            return httpx.Response(302, headers={"Location": f"{self.storage_url}/result/{task_id}?signature=a%2Fb"})
        self.downloads += 1
        assert "signature=a%2Fb" in str(request.url)
        if self.download_expired:
            self.download_expired = False
            task["version"] += 1
            return httpx.Response(403)
        if self.download_corrupt:
            self.download_corrupt = False
            return httpx.Response(200, content=b"corrupt result")
        if self.download_interrupted:
            self.download_interrupted = False
            return httpx.Response(200, stream=InterruptedStream())
        return httpx.Response(200, content=artifact_bytes(task["options"]["output_formats"]))


def assert_no_upstream_cancellation(upstream: Upstream) -> None:
    """超时、断连和关闭包装服务均不得发送新取消请求或旧 DELETE 请求。"""
    cancellations = [
        (method, url)
        for method, url, _authorization in upstream.calls
        if method == "DELETE" or (method == "POST" and urlsplit(url).path.endswith("/cancel"))
    ]
    assert not cancellations


class InterruptedStream(httpx.AsyncByteStream):
    """模拟结果传输到一半时上游断开连接。"""

    async def __aiter__(self) -> Any:
        """发出部分 ZIP 头后报错，重试必须从头覆盖临时文件。"""
        yield b"PK\x03\x04partial"
        raise httpx.ReadError("download disconnected")


def application(tmp_path: Path, upstream: Upstream, **settings: Any) -> Any:
    """用真实包装层和可控 HTTP 传输构造测试应用。"""
    return create_app(
        FileParseSettings("https://api.test/api", "secret", tmp_path, poll_interval=0.01, **settings),
        transport=httpx.MockTransport(upstream.handle),
    )


def submit(client: TestClient, *, request_id: str = "request_1", content: bytes = b"source", **options: Any) -> httpx.Response:
    """使用真实 multipart 编码，列表选项通过重复同名字段发送。"""
    fields: list[tuple[str, Any]] = [("file", ("report.pdf", content, "application/pdf")), ("request_id", (None, request_id))]
    for name, value in options.items():
        for item in value if isinstance(value, list) else [value]:
            fields.append((name, (None, str(item))))
    return client.post("/v1/file_parse", files=fields)


FORMATS = [
    list(values) for size in range(1, 5) for values in itertools.combinations(("default", "docx", "html", "latex"), size)
]


@pytest.mark.parametrize("tier", [None, "flash", "basic", "standard", "advanced"])
def test_file_parse_forwards_tier_only_when_provided(tmp_path: Path, tier: str | None) -> None:
    """同步包装层不猜测上游默认档位，显式值原样进入 Tasks 请求。"""
    upstream = Upstream()
    with TestClient(application(tmp_path, upstream)) as client:
        response = submit(client, **({"tier": tier} if tier is not None else {}))
        assert response.status_code == 200, response.text
    options = upstream.tasks["task_1"]["options"]
    if tier is None:
        assert "tier" not in options
    else:
        assert options["tier"] == tier


@pytest.mark.parametrize("formats", FORMATS)
def test_formats_are_returned_unchanged(tmp_path: Path, formats: list[str]) -> None:
    """覆盖全部十五种格式组合及附件字节、文件名、MIME 和临时文件释放。"""
    upstream = Upstream()
    app = application(tmp_path, upstream)
    with TestClient(app) as client:
        response = submit(client, output_formats=formats)
        assert response.status_code == 200, response.text
        assert response.content == artifact_bytes(formats)
        extension = formats[0] if formats in (["docx"], ["html"]) else "zip"
        assert f"report.{extension}" in response.headers["content-disposition"]
        assert response.headers["x-request-id"] == "request_1"
        assert response.headers["x-task-id"] == "task_1"
        assert not list(app.state.service.store.sources.iterdir())
        assert not list(app.state.service.store.results.iterdir())


def test_timeout_recovery_deduplication_and_conflict(tmp_path: Path) -> None:
    """零等待返回包装地址，同 ID 复用任务，参数冲突不会替换原任务。"""
    upstream = Upstream()
    upstream.complete = False
    app = application(tmp_path, upstream)
    with TestClient(app, root_path="/gateway") as client:
        response = submit(client, wait_timeout=0, output_formats=["html", "html"])
        assert response.status_code == 202
        payload = response.json()
        assert response.headers["location"] == payload["status_url"]
        assert payload["status_url"] == "http://testserver/gateway/v1/file_parse/request_1"
        assert payload["output_formats"] == ["html"]
        assert payload["result_url"] is None
        assert client.get(payload["status_url"] + "/result").status_code == 409
        assert submit(client, wait_timeout=0, output_formats=["html"]).status_code == 202
        assert len(upstream.tasks) == 1
        assert submit(client, content=b"different", wait_timeout=0).status_code == 409
        upstream.complete = True
        response = client.get(payload["status_url"])
        assert response.json()["status"] == "completed"
        assert client.get(response.json()["result_url"]).content == artifact_bytes(["html"])
        assert not list(app.state.service.store.results.iterdir())


def test_wait_budget_excludes_upload(tmp_path: Path) -> None:
    """慢速上游上传不占一秒等待预算，解析完成仍能直接返回附件。"""
    upstream = Upstream()
    upstream.upload_delay = 1.1
    upstream.complete_delay = 0.15
    with TestClient(application(tmp_path, upstream)) as client:
        started = time.monotonic()
        response = submit(client, wait_timeout=1)
        assert response.status_code == 200, response.text
        assert time.monotonic() - started >= 1.1


def test_expired_wait_does_not_cancel_upstream(tmp_path: Path) -> None:
    """等待预算到期返回 202，后台继续观察任务直至完成。"""
    upstream = Upstream()
    upstream.complete = False
    with TestClient(application(tmp_path, upstream)) as client:
        started = time.monotonic()
        response = submit(client, wait_timeout=1)
        assert response.status_code == 202
        assert 0.9 <= time.monotonic() - started < 3
        assert_no_upstream_cancellation(upstream)
        upstream.complete = True
        assert client.get(response.json()["status_url"]).json()["status"] == "completed"
    assert_no_upstream_cancellation(upstream)


def test_wait_budget_excludes_storage_notification_delay(tmp_path: Path) -> None:
    """存储已经接收字节但网关尚未确认 ready 时，不消耗同步解析等待预算。"""
    upstream = Upstream()
    upstream.ready_delay = 1.1
    upstream.complete_delay = 1.2
    with TestClient(application(tmp_path, upstream)) as client:
        response = submit(client, wait_timeout=1)
        assert response.status_code == 200, response.text
    assert upstream.uploads == 1


@pytest.mark.parametrize("failure", ["lost", "timeout", "malformed"])
def test_unknown_creation_is_never_replayed(tmp_path: Path, failure: str) -> None:
    """上游创建成功但响应丢失或身份缺失时，重试和重启都不能再次 POST。"""
    upstream = Upstream()
    upstream.create_failure = failure
    with TestClient(application(tmp_path, upstream)) as client:
        response = submit(client)
        assert response.status_code == 502
        assert response.json()["status"] == "submission_unknown"
        assert response.json()["task_id"] is None
        assert submit(client).status_code in {409, 502}
    with TestClient(application(tmp_path, upstream)) as client:
        assert client.get("/v1/file_parse/request_1").json()["status"] == "submission_unknown"
        assert submit(client).status_code in {409, 502}
    assert len(upstream.tasks) == 1


@pytest.mark.parametrize("deployment", ["api", "router"])
@pytest.mark.parametrize(
    "tiers,tier,code,upstream_status",
    [
        (("advanced",), "standard", "invalid_request", 400),
        (("flash",), "basic", "invalid_request", 400),
    ],
)
def test_real_task_tier_rejection_is_failed_and_not_replayed(
    tmp_path: Path, deployment: str, tiers: tuple[Tier, ...], tier: Tier | None, code: str, upstream_status: int
) -> None:
    """真实 Tasks 拒单保留错误原因，包装层重复提交和重启都不能再次创建任务。"""
    from test_task_tiers import _deployment

    async def scenario() -> None:
        """同一事件循环驱动真实 API、Router 和包装服务，观察上游提交次数与持久状态。"""
        async with _deployment(tmp_path / "upstream", deployment, {"worker": tiers}) as (_client, upstream, apps, _calls):
            transport = httpx.ASGITransport(app=upstream)
            submitted: list[httpx.Request] = []

            async def handle(request: httpx.Request) -> httpx.Response:
                """记录包装层实际发送的请求，错误正文仍由真实 Tasks 生成。"""
                submitted.append(request)
                return await transport.handle_async_request(request)

            fields = [("file", ("report.pdf", b"source")), ("request_id", (None, "rejected"))]
            if tier is not None:
                fields.append(("tier", (None, tier)))
            settings = FileParseSettings("http://upstream", state_dir=tmp_path / "wrapper", poll_interval=0.01)
            for _restart in range(2):
                app = create_app(settings, transport=httpx.MockTransport(handle))
                async with app.router.lifespan_context(app):
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://wrapper") as client:
                        for _repeat in range(2):
                            response = await client.post("/v1/file_parse", files=fields)
                            assert response.status_code == 502, response.text
                            payload = response.json()
                            assert payload["status"] == "failed" and payload["task_id"] is None
                            assert payload["error"]["code"] == code
                            assert payload["error"]["upstream_status"] == upstream_status
                            assert payload["error"]["message"] != "Task creation outcome is unknown; not resubmitting"
                        response = await client.get("/v1/file_parse/rejected")
                        assert response.status_code == 200
                        assert response.json()["status"] == "failed"
                        assert response.json()["error"]["code"] == code
                        assert not list(app.state.service.store.sources.iterdir())
            assert len(submitted) == 1 and submitted[0].method == "POST"
            assert all(not app.state.task_store._records for app in apps)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status,payload",
    [
        (503, b"<html>Service unavailable</html>"),
        (503, {"error": {"code": "temporary", "message": "Try later"}}),
        (503, {"error": {"code": "quality_tier_unavailable"}}),
        (503, {"error": {"message": "Missing error code"}}),
        (503, {"error": {"code": None, "message": "Null error code"}}),
        (503, {"error": {"code": "", "message": "Empty error code"}}),
        (503, {"error": {"type": "api_error", "code": "quality_tier_unavailable", "message": "Legacy error type"}}),
        (
            503,
            {"task_id": "maybe-created", "error": {"code": "quality_tier_unavailable", "message": "x"}},
        ),
        (500, {"error": {"code": "quality_tier_unavailable", "message": "Unexpected status"}}),
        (408, {"error": {"code": "quality_tier_unavailable", "message": "Timed out"}}),
    ],
)
def test_ambiguous_creation_errors_remain_unknown(tmp_path: Path, status: int, payload: bytes | dict[str, Any]) -> None:
    """未列为明确拒单的状态或畸形错误信封保持未知，不能靠错误码字符串判断已拒绝。"""
    calls: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        """模拟代理错误或矛盾响应，记录是否发生意外重放。"""
        calls.append(request)
        assert request.method == "POST"
        return httpx.Response(status, content=payload) if isinstance(payload, bytes) else httpx.Response(status, json=payload)

    for _restart in range(2):
        app = create_app(FileParseSettings("https://api.test", state_dir=tmp_path), transport=httpx.MockTransport(handle))
        with TestClient(app) as client:
            for _repeat in range(2):
                response = submit(client)
                assert response.status_code == 502
                assert response.json()["status"] == "submission_unknown"
                assert response.json()["error"]["code"] == "submission_unknown"
            assert client.get("/v1/file_parse/request_1").json()["status"] == "submission_unknown"
    assert len(calls) == 1


def test_auth_failure_cleans_source_without_retry(tmp_path: Path) -> None:
    """上游认证错误是配置错误，不向无认证的包装入口伪报 401。"""
    upstream = Upstream()
    upstream.create_failure = "auth"
    app = application(tmp_path, upstream)
    with TestClient(app) as client:
        response = submit(client)
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "invalid_api_key"
        assert response.json()["status_url"]
        assert not list(app.state.service.store.sources.iterdir())
    assert len(upstream.calls) == 1


def test_upload_response_loss_is_reconciled(tmp_path: Path) -> None:
    """上传已成功但响应丢失时先查 ready，避免再次发送文件。"""
    upstream = Upstream()
    upstream.upload_response_lost = True
    with TestClient(application(tmp_path, upstream)) as client:
        assert submit(client).status_code == 200
    assert upstream.uploads == 1


def test_upload_retry_budget_and_explicit_resume(tmp_path: Path) -> None:
    """自动上传最多尝试三次；调用者重传相同请求可以继续原任务而不再次创建。"""
    upstream = Upstream()
    upstream.upload_failures = 3
    app = application(tmp_path, upstream)
    with TestClient(app) as client:
        app.state.service.client.retry_delay = 0
        response = submit(client)
        assert response.status_code == 502
        assert response.json()["task_id"] == "task_1"
        assert upstream.uploads == 3
        assert list(app.state.service.store.sources.iterdir())
        assert submit(client).status_code == 200
        assert not list(app.state.service.store.sources.iterdir())
    assert upstream.uploads == 4 and len(upstream.tasks) == 1


@pytest.mark.parametrize("fault", ["download_expired", "download_corrupt", "download_interrupted", "redirect"])
def test_download_retries_and_cross_origin_headers(tmp_path: Path, fault: str) -> None:
    """签名失效和损坏结果可以恢复；重定向每跳都验证未向存储发送 API Key。"""
    upstream = Upstream()
    setattr(upstream, fault, True)
    app = application(tmp_path, upstream)
    with TestClient(app) as client:
        app.state.service.client.retry_delay = 0
        response = submit(client)
        assert response.status_code == 200
        assert response.content == artifact_bytes(["default"])
        assert upstream.downloads == (1 if fault == "redirect" else 2)
        assert not list(app.state.service.store.results.iterdir())


def test_cache_hit_avoids_upload(tmp_path: Path) -> None:
    """源文件 hash 缓存命中时直接等待，不尝试构造或访问上传链接。"""
    upstream = Upstream()
    upstream.cache_hit = True
    with TestClient(application(tmp_path, upstream)) as client:
        assert submit(client).status_code == 200
    assert upstream.uploads == 0


def test_concurrent_duplicate_requests_share_one_task(tmp_path: Path) -> None:
    """并发提交相同请求 ID 只创建、上传一次，但每个连接可分别下载。"""
    upstream = Upstream()
    upstream.upload_delay = 0.1
    with TestClient(application(tmp_path, upstream)) as client, ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(lambda _: submit(client), range(4)))
        assert all(response.status_code == 200 for response in responses)
    assert len(upstream.tasks) == 1
    assert upstream.uploads == 1


def test_restart_uses_existing_task_and_isolates_upstream(tmp_path: Path) -> None:
    """进程重启后查询同一任务；更换上游配置不会访问旧记录。"""
    upstream = Upstream()
    upstream.complete = False
    with TestClient(application(tmp_path, upstream)) as client:
        response = submit(client, wait_timeout=0)
        assert response.status_code == 202
    upstream.complete = True
    with TestClient(application(tmp_path, upstream)) as client:
        assert client.get("/v1/file_parse/request_1").json()["status"] == "completed"
        assert client.get("/v1/file_parse/request_1/result").status_code == 200
    assert len(upstream.tasks) == 1
    app = create_app(FileParseSettings("https://different.test", state_dir=tmp_path))
    with TestClient(app) as client:
        assert client.get("/v1/file_parse/request_1").status_code == 404


def test_new_protocol_preserves_but_does_not_restore_legacy_records(tmp_path: Path) -> None:
    """新协议隔离旧数据库和源文件，同名请求只恢复新空间中已确认的任务。"""
    upstream = Upstream()
    settings = FileParseSettings(upstream.api_url, "secret", tmp_path)
    legacy_namespace = hashlib.sha256(f"{settings.api_url}\0secret".encode()).hexdigest()
    assert settings.namespace == "tasks-v5-" + legacy_namespace
    legacy = tmp_path / ("tasks-v4-" + legacy_namespace)
    source = legacy / "sources" / "old-input"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"old source")
    result = legacy / "results" / "old-result.zip"
    result.parent.mkdir()
    result.write_bytes(b"old result")
    old_options = {**ParseOptions().model_dump(), "include_model_output": True}
    old_record = Record("request_1", "report.pdf", 10, "a" * 64, old_options, str(source), "old-fingerprint")
    database = legacy / "requests.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE requests (request_id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        connection.execute("INSERT INTO requests VALUES (?, ?)", (old_record.request_id, old_record.to_json()))
    before = {path: path.read_bytes() for path in legacy.rglob("*") if path.is_file()}

    with TestClient(application(tmp_path, upstream)) as client:
        assert client.get("/v1/file_parse/request_1").status_code == 404
        assert upstream.calls == []
        assert submit(client).status_code == 200
        assert "include_model_output" not in upstream.tasks["task_1"]["options"]
    with TestClient(application(tmp_path, upstream)) as client:
        assert client.get("/v1/file_parse/request_1").json()["task_id"] == "task_1"
        assert client.get("/v1/file_parse/request_1/result").status_code == 200
    assert len(upstream.tasks) == 1
    assert {path: path.read_bytes() for path in legacy.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("enabled", ["false", "true"])
def test_removed_model_output_form_field_is_rejected(tmp_path: Path, enabled: str) -> None:
    """旧表单字段无论取值都报错，不创建上游任务或恢复记录。"""
    upstream = Upstream()
    app = application(tmp_path, upstream)
    with TestClient(app) as client:
        response = client.post(
            "/v1/file_parse",
            files={"file": ("report.pdf", b"source", "application/pdf")},
            data={"request_id": "removed-option", "include_model_output": enabled},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"
        assert not app.state.service.records and not upstream.calls


@pytest.mark.parametrize("stage", ["prepared", "pending", "expired", "submitting"])
def test_restart_at_upload_boundaries(tmp_path: Path, stage: str) -> None:
    """覆盖崩溃前尚未创建、已创建待上传、上传过期和提交不确定四个恢复边界。"""
    upstream = Upstream()
    settings = FileParseSettings("https://api.test/api", "secret", tmp_path, poll_interval=0.01)

    async def seed() -> None:
        """直接落盘一个崩溃时刻的真实恢复记录，不伪造运行中的任务对象。"""
        store = RecordStore(settings)
        await store.open()
        source = store.sources / "retained-source"
        source.write_bytes(b"source")
        options = ParseOptions()
        record = Record(
            "request_1",
            "report.pdf",
            6,
            "a" * 64,
            options.model_dump(),
            str(source),
            fingerprint("report.pdf", 6, "a" * 64, options),
        )
        if stage in {"pending", "expired"}:
            remote = TaskClient(settings, httpx.MockTransport(upstream.handle))
            try:
                task = await remote.create(record)
                record.remote = task.model_dump()
                record.status = "waiting-file"
                if stage == "expired":
                    upstream.tasks[task.task_id]["expires_at"] = int(time.time()) - 1
            finally:
                await remote.close()
        elif stage == "submitting":
            record.status = "submitting"
        await store.save(record)
        await store.close()

    asyncio.run(seed())
    app = application(tmp_path, upstream)
    with TestClient(app) as client:
        deadline = time.monotonic() + 3
        while True:
            response = client.get("/v1/file_parse/request_1")
            status = response.json()["status"]
            if status in {"completed", "failed", "submission_unknown"} or time.monotonic() > deadline:
                break
            time.sleep(0.01)
        assert status == ("submission_unknown" if stage == "submitting" else "failed" if stage == "expired" else "completed")
        assert len(upstream.tasks) == (0 if stage == "submitting" else 1)
        assert upstream.uploads == (1 if stage in {"prepared", "pending"} else 0)
        assert not list(app.state.service.store.sources.iterdir())


def test_disconnect_does_not_cancel_upload(tmp_path: Path) -> None:
    """真实 ASGI 断连消息只能取消连接等待，已经接收的源文件继续上传。"""
    upstream = Upstream()
    upstream.upload_delay = 0.1
    app = application(tmp_path, upstream)

    async def exercise() -> None:
        """发送完整 multipart 后立即断连，再通过同一请求 ID 检查完成结果。"""
        async with app.router.lifespan_context(app):
            encoded = httpx.Request(
                "POST",
                "http://wrapper/v1/file_parse",
                files={
                    "file": ("report.pdf", b"source"),
                    "request_id": (None, "request_1"),
                },
            )
            body = encoded.read()
            received = False

            async def receive() -> dict[str, Any]:
                """第一次返回完整文件，后续接收返回真实断连事件。"""
                nonlocal received
                if not received:
                    received = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return {"type": "http.disconnect"}

            async def send(message: dict[str, Any]) -> None:
                """模拟断开的客户端，不消费任何响应内容。"""
                assert message["type"] in {"http.response.start", "http.response.body"}

            scope = {
                "type": "http",
                "method": "POST",
                "path": "/v1/file_parse",
                "root_path": "",
                "raw_path": b"/v1/file_parse",
                "query_string": b"",
                "headers": [(key.lower(), value) for key, value in encoded.headers.raw],
                "server": ("wrapper", 80),
                "client": ("client", 123),
                "scheme": "http",
                "http_version": "1.1",
                "asgi": {"version": "3.0", "spec_version": "2.4"},
            }
            await app(scope, receive, send)
            for _ in range(100):
                record = app.state.service.records["request_1"]
                if record.status == "completed":
                    break
                await asyncio.sleep(0.01)
            assert record.status == "completed"
            assert upstream.uploads == 1 and len(upstream.tasks) == 1
            assert not list(app.state.service.store.sources.iterdir())

    asyncio.run(exercise())
    assert_no_upstream_cancellation(upstream)


def test_attachment_cleanup_after_send_disconnect(tmp_path: Path) -> None:
    """附件发送中断也执行清理，而非只依赖发送成功后的后台回调。"""
    path = tmp_path / "result.zip"
    path.write_bytes(artifact_bytes(["default"]))

    async def exercise() -> None:
        """在响应已经开始后断开下游，验证 finally 释放结果文件。"""

        async def receive() -> dict[str, Any]:
            """提供断连事件，兼容 ASGI 响应实现。"""
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            """在文件主体发送时模拟客户端 socket 关闭。"""
            if message["type"] == "http.response.body":
                raise OSError("client disconnected")

        with pytest.raises(OSError):
            await TemporaryFileResponse(path)(
                {"type": "http", "method": "GET", "headers": [], "asgi": {"spec_version": "2.4"}}, receive, send
            )

    asyncio.run(exercise())
    assert not path.exists()


def test_retention_and_state_directory_lock(tmp_path: Path) -> None:
    """同目录第二进程无法接管；恢复记录按首次终态时间回收。"""
    upstream = Upstream()
    app = application(tmp_path, upstream, retention_seconds=1)
    with TestClient(app) as client:
        assert submit(client).status_code == 200
        with pytest.raises(Timeout), TestClient(application(tmp_path, upstream)):
            pass
        app.state.service.records["request_1"].terminal_at = time.time() - 2
        assert client.get("/v1/file_parse/request_1").status_code == 404
        client.portal.call(app.state.service.collect)
        assert "request_1" not in app.state.service.records


def test_late_query_cannot_replace_reused_request_id(tmp_path: Path) -> None:
    """记录过期并复用 ID 后，旧任务的迟到查询不能覆盖新任务的持久化映射。"""
    upstream = Upstream()
    started, release = threading.Event(), asyncio.Event()
    pause = False

    async def delayed(request: httpx.Request) -> httpx.Response:
        """仅延迟旧任务查询，保证新任务的创建和下载仍可正常完成。"""
        if pause and request.method == "GET" and request.url.path.endswith("/v1/tasks/task_1"):
            started.set()
            await release.wait()
        return await upstream.handle(request)

    app = create_app(
        FileParseSettings("https://api.test/api", "secret", tmp_path, retention_seconds=1, poll_interval=0.01),
        transport=httpx.MockTransport(delayed),
    )
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        assert submit(client).status_code == 200
        pause = True
        pending = executor.submit(client.get, "/v1/file_parse/request_1")
        assert started.wait(2)
        app.state.service.records["request_1"].terminal_at = time.time() - 2
        response = submit(client)
        assert response.status_code == 200
        assert response.headers["x-task-id"] == "task_2"
        client.portal.call(release.set)
        assert pending.result(timeout=2).status_code == 200
        assert app.state.service.records["request_1"].task.task_id == "task_2"
    with TestClient(application(tmp_path, upstream)) as client:
        assert client.get("/v1/file_parse/request_1").json()["task_id"] == "task_2"


@pytest.mark.parametrize(
    "options",
    [
        {"wait_timeout": -1},
        {"wait_timeout": 3601},
        {"wait_timeout": "1.5"},
        {"request_id": "../bad"},
        {"output_formats": ["markdown"]},
        {"output_formats": [""]},
        {"tier": "wrong"},
        {"tier": ""},
        {"tier": "null"},
        {"ocr_mode": "wrong"},
        {"api_url": "https://different.test"},
    ],
)
def test_invalid_parameters_do_not_create_tasks(tmp_path: Path, options: dict[str, Any]) -> None:
    """所有本地可校验参数错误在创建上游任务前返回。"""
    upstream = Upstream()
    with TestClient(application(tmp_path, upstream)) as client:
        assert submit(client, **options).status_code == 400
    assert not upstream.tasks


def test_single_file_content_type_and_scalar_duplicates(tmp_path: Path) -> None:
    """拒绝 JSON、空文件、重复文件和重复标量，格式列表保持唯一例外。"""
    upstream = Upstream()
    with TestClient(application(tmp_path, upstream)) as client:
        assert client.post("/v1/file_parse", json={}).status_code == 415
        assert submit(client, content=b"").status_code == 400
        assert client.post("/v1/file_parse", files=[("file", ("a.pdf", b"a")), ("file", ("b.pdf", b"b"))]).status_code == 400
        assert (
            client.post(
                "/v1/file_parse",
                files=[
                    ("file", ("a.pdf", b"a")),
                    ("tier", (None, "flash")),
                    ("tier", (None, "flash")),
                ],
            ).status_code
            == 400
        )
    assert not upstream.tasks


def test_openapi_and_light_imports(tmp_path: Path) -> None:
    """OpenAPI 说明格式数组及附件，独立导入包装应用不加载解析运行时。"""
    schema = application(tmp_path, Upstream()).openapi()
    post = schema["paths"]["/v1/file_parse"]["post"]
    request = post["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert request["properties"]["wait_timeout"]["maximum"] == 3600
    assert request["properties"]["output_formats"]["items"]["enum"] == ["default", "docx", "html", "latex"]
    assert "tier" not in request["required"]
    tier = request["properties"]["tier"]
    assert tier["type"] == "string" and tier["enum"] == ["flash", "basic", "standard", "advanced"]
    assert {"anyOf", "nullable", "default"}.isdisjoint(tier)
    assert "Omit this multipart field" in tier["description"]
    assert "application/zip" in post["responses"]["200"]["content"]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from mineru.kit.file_parse import create_app; "
                "assert not any(x in sys.modules for x in ('torch', 'transformers', 'mineru.parser.api_server'))"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cli_configuration_and_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """真实 CLI 解析环境变量及显式覆盖，把上游配置交给独立应用。"""
    from mineru.kit.main import app

    captured: list[tuple[Any, dict[str, Any]]] = []

    def run(application: Any, **options: Any) -> None:
        """捕获启动参数，通过应用 lifespan 验证最终配置。"""
        captured.append((application, options))
        with TestClient(application):
            assert application.state.service.settings.api_url == "https://explicit.test/api"
            assert application.state.service.settings.api_key == "environment-key"

    monkeypatch.setattr("uvicorn.run", run)
    monkeypatch.setenv("MINERU_API_URL", "https://environment.test")
    monkeypatch.setenv("MINERU_API_KEY", "environment-key")
    result = CliRunner().invoke(
        app, ["file-parse-server", "--api-url", "https://explicit.test/api", "--state-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert captured[0][1] == {"host": "127.0.0.1", "port": 8003}


@pytest.mark.parametrize("cached", [False, True])
def test_restart_after_creation_receipt_and_failed_first_query(tmp_path: Path, cached: bool) -> None:
    """创建成功后查询失败仍先持久化 ID，重启仅恢复原任务，包括已完成的缓存回执。"""

    class FailingQueries(Upstream):
        """把首次进程的详情查询固定为临时失败，保留创建和对象存储行为。"""

        fail_queries = True

        async def handle(self, request: httpx.Request) -> httpx.Response:
            """让失败发生在已确认创建之后，避免测试误走创建结果未知分支。"""
            if self.fail_queries and request.method == "GET" and "/v1/tasks/" in request.url.path:
                return httpx.Response(503, json={"error": {"code": "temporary", "message": "retry query"}})
            return await super().handle(request)

    upstream = FailingQueries()
    upstream.cache_hit = cached
    app = application(tmp_path, upstream)
    with TestClient(app) as client:
        app.state.service.client.retry_delay = 0
        response = submit(client)
        assert response.status_code == 502, response.text
        record = app.state.service.records["request_1"]
        assert record.task.task_id == "task_1"
        assert record.task.files[0].file_id == "file-task_1"
        assert not record.task.details_loaded
        assert record.status != "submission_unknown"
        if not cached:
            assert Path(record.source_path).is_file()
    upstream.fail_queries = False
    with TestClient(application(tmp_path, upstream)) as client:
        response = submit(client)
        assert response.status_code == 200, response.text
        assert response.content == artifact_bytes(["default"])
        state = client.get("/v1/file_parse/request_1").json()
        assert state["task_id"] == "task_1" and state["status"] == "completed"
    assert len(upstream.tasks) == 1 and upstream.uploads == (0 if cached else 1)
