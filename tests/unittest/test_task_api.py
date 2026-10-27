"""真实 API 与双 worker Router 的表单任务、逐文件上传和 ZIP 链接契约。"""

from __future__ import annotations

from task_test_utils import pending_files, task_url

import asyncio
import hashlib
import io
import json
import threading
import time
import zipfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mineru.kit.router import RouterSettings
from mineru.kit.router import create_app as create_router
from mineru.parser import ParseResult, api_server
from mineru.tasks.service import TERMINAL_TASK_STATUSES

HTML = b"<!doctype html><html><body><h1>Shared tasks</h1><p>Preserved content.</p></body></html>"
SECOND = b"<h1>Second document</h1><p>Second content.</p>"


@pytest.fixture
def url_downloads(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """仅替换 worker 的外部下载传输，Router 和客户端仍经过真实 ASGI 协议。"""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """保留完整签名 URL 和请求头，并提供下载成功及失败来源。"""
        requests.append(request)
        if request.url.path == "/timeout":
            raise httpx.ReadTimeout("source timed out", request=request)
        return httpx.Response(
            404 if request.url.path == "/missing" else 200,
            content=b"" if request.url.path == "/empty" else HTML,
            headers={"content-type": "text/html; charset=utf-8"},
        )

    def download_client(**kwargs: Any) -> httpx.AsyncClient:
        """使用真实 httpx 流式客户端，隔离源站网络而不替换下载业务实现。"""
        return httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(api_server, "httpx", SimpleNamespace(AsyncClient=download_client))
    return requests


def _execution_apps(app: FastAPI) -> list[FastAPI]:
    """取得真正执行 URL 下载的服务，用于验证部署配置及取消后的资源释放。"""
    return list(app.state.test_transport.apps.values()) if hasattr(app.state, "test_transport") else [app]


def _source_references(app: FastAPI, task: dict[str, Any]) -> set[str]:
    """从内部存储定位源引用，公共 Tasks 响应不再暴露源文件标识。"""
    upload = app.state.task_store._records[task["task_id"]].file_states[0].upload
    assert upload is not None
    if hasattr(app.state, "source_store"):
        return {key for key, stored in app.state.source_store._files.items() if stored.path == upload.path}
    assert upload.file_id is not None
    return {upload.file_id}


def _delete_source_references(client: TestClient, app: FastAPI, references: set[str]) -> None:
    """通过内部清理旧源引用复现共享字节竞争，不依赖公开删除接口。"""
    assert references
    for reference in references:
        if hasattr(app.state, "source_store"):
            client.portal.call(app.state.source_store.delete_file, reference)
        else:
            client.portal.call(app.state.file_store.delete_file, reference)


def _url_manifest(**metadata: Any) -> dict[str, Any]:
    """构造与下载地址名称无关的逻辑文件条目，默认不声明大小或 hash。"""
    return {
        "file_name": "logical.html",
        "url": "https://source.example/opaque?signature=a%2Fb&part=1",
        **metadata,
    }


class _WorkerTransport(httpx.AsyncBaseTransport):
    """使用真实 worker 路由验证源文件权限和任务生命周期，不以宽松 fake 替代。"""

    def __init__(self, apps: dict[str, FastAPI]) -> None:
        """保存 worker 和真实 ASGI transport，供断连与调度测试观察。"""
        self.apps = apps
        self.transports = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}
        self.calls: list[tuple[str, str, str]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """按主机选择 worker，并记录真实上游协议。"""
        self.calls.append((request.url.host, request.method, request.url.path))
        return await self.transports[request.url.host].handle_async_request(request)

    async def aclose(self) -> None:
        """等待任务与 Job 清理完成，再释放测试 worker 的解析运行时。"""
        for app in self.apps.values():
            await app.state.task_store.shutdown()
            await app.state.execution_store.shutdown()


@pytest.fixture(params=["api", "router"])
def service(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[tuple[TestClient, FastAPI]]:
    """对本地服务与双 worker Router 执行同一份共同协议验收。"""
    if request.param == "api":
        app = api_server.create_app(upload_dir=str(tmp_path / "direct"), tiers=("flash",))
    else:
        workers = {
            host: api_server.create_app(upload_dir=str(tmp_path / host), tiers=("flash",)) for host in ("worker-a", "worker-b")
        }
        transport = _WorkerTransport(workers)
        app = create_router(
            RouterSettings(
                upstream_urls=tuple(f"http://{host}" for host in workers), local_gpus="none", worker_refresh_interval_seconds=0
            ),
            transport=transport,
        )
        app.state.test_transport = transport
    with TestClient(app) as client:
        yield client, app


def _submit(
    client: TestClient,
    inputs: list[tuple[str, bytes]],
    *,
    hashes: bool = True,
    headers: dict[str, str] | None = None,
    **options: str | list[str],
) -> dict[str, Any]:
    """创建真实任务并读取初始详情，响应模型本身由独立契约测试覆盖。"""
    manifest = [
        {
            "file_name": name,
            "bytes": len(data),
            **({"sha256sum": hashlib.sha256(data).hexdigest()} if hashes else {}),
        }
        for name, data in inputs
    ]
    response = client.post("/v1/tasks", json={"files": manifest, "tier": "flash", **options}, headers=headers)
    assert response.status_code == 202, response.text
    detail = client.get(task_url(client, response.json()), headers=headers)
    assert detail.status_code == 200, detail.text
    return detail.json()


def _upload(
    client: TestClient, task: dict[str, Any], inputs: list[tuple[str, bytes]], *, headers: dict[str, str] | None = None
) -> None:
    """仅上传返回链接对应的缺失文件，按 file_id 保持同名文件的关联。"""
    for index, link in enumerate(task["files"]):
        if link.get("upload") is None:
            continue
        response = client.put(
            link.get("upload").get("url"),
            content=inputs[index][1],
            headers={**link.get("upload")["headers"], **(headers or {})},
        )
        assert response.status_code == 200, response.text


def _terminal(client: TestClient, task: dict[str, Any], *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """只轮询状态接口；未完成的文件不能发布 result_url。"""
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        response = client.get(task_url(client, task), headers=headers)
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] in TERMINAL_TASK_STATUSES:
            return payload
        assert all(file.get("result_url") is None for file in payload["files"] if file["status"] != "completed")
        time.sleep(0.01)
    pytest.fail("Task did not reach a terminal state")


def _file_done(client: TestClient, task: dict[str, Any], index: int) -> dict[str, Any]:
    """只等待指定文件完成，父任务可继续等待其他上传。"""
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        payload = client.get(task_url(client, task)).json()
        if payload["files"][index]["status"] in {"completed", "failed", "canceled"}:
            return payload
        time.sleep(0.01)
    pytest.fail("File did not finish")


def _archive(client: TestClient, task: dict[str, Any], index: int = 0) -> dict[str, bytes]:
    """从返回的独立 Files 链接下载并读取真实 ZIP 成员。"""
    assert "/v1/files/" in task["files"][index].get("result_url")
    response = client.get(task["files"][index].get("result_url"))
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _job_count(app: FastAPI) -> int:
    """观察唯一执行队列，准备任务不应该产生 Job 或 worker 预占。"""
    if hasattr(app.state, "execution_store"):
        return len(app.state.execution_store._executions)
    return len(app.state.registry.list("job"))


@pytest.mark.parametrize("hashes", [False, True])
def test_manifest_uploads_and_zip_only_status(service: tuple[TestClient, FastAPI], hashes: bool) -> None:
    """先上传文件立即执行并可下载，同名文件通过客户端标识保持独立。"""
    client, app = service
    inputs = [("同名 文档.html", HTML), ("同名 文档.html", SECOND)]
    task = _submit(client, inputs, hashes=hashes, output_formats=["default"])
    assert task["status"] == "waiting-file" and "result_url" not in task
    assert len(pending_files(task)) == 2 and _job_count(app) == 0
    assert client.put(pending_files(task)[0].get("upload").get("url"), content=HTML).status_code == 200
    ready = _file_done(client, task, 0)
    assert ready["status"] == "waiting-file"
    assert [file["status"] for file in ready["files"]] == ["completed", "waiting-file"]
    assert [link["file_id"] for link in pending_files(ready)] == [task["files"][1]["file_id"]]
    assert _job_count(app) == 1
    assert b"Preserved content" in _archive(client, ready)["markdown.md"]
    _upload(client, ready, inputs)
    result = _terminal(client, task)
    assert result["status"] == "completed" and _job_count(app) == 2
    assert result["task_id"] == task["task_id"]
    assert all("content" not in file and "output_files" not in file for file in result["files"])
    assert client.get(f"/v1/parse/jobs/{task['task_id']}").status_code == 404
    assert b"Second content" in _archive(client, result, 1)["markdown.md"]
    members = _archive(client, result)
    assert "manifest.json" not in members and "model_output.json" not in members
    assert json.loads(members["middle_json.json"])["schema"] == "docvortex.middle"
    assert client.get(f"/v1/tasks/{task['task_id']}/result").status_code == 404
    assert client.post("/v1/file_parse").status_code == 404
    assert client.post("/tasks").status_code == 404
    assert all({"client_file_id", "name", "sha256sum", "page_range"}.isdisjoint(file) for file in result["files"])


def test_direct_and_form_submissions_are_rejected(service: tuple[TestClient, FastAPI]) -> None:
    """无论单文件还是批量，创建请求都不能携带表单或文件主体。"""
    client, app = service
    for response in (
        client.post("/v1/tasks", data={"tier": "flash"}),
        client.post("/v1/tasks", files={"files": ("a.html", HTML)}),
        client.post("/v1/tasks", content=HTML, headers={"content-type": "application/octet-stream"}),
    ):
        assert response.status_code == 415
    assert _job_count(app) == 0


class _UnreadableBody(httpx.SyncByteStream):
    """重复上传已确认文件时，读取主体应立即使测试失败。"""

    def __iter__(self) -> Iterator[bytes]:
        """验证服务端幂等返回，既不覆盖已确认字节也不读取重传主体。"""
        raise AssertionError("Ready file body must not be read")
        yield b""  # pragma: no cover


def test_mixed_cache_hit_survives_source_deletion(service: tuple[TestClient, FastAPI]) -> None:
    """命中项在等待其他文件时持有独立引用，原源文件删除不导致缓存竞态。"""
    client, app = service
    initial = _submit(client, [("original.html", HTML)])
    _upload(client, initial, [("original.html", HTML)])
    original = _terminal(client, initial)
    old_references = _source_references(app, original)
    inputs = [("renamed.html", HTML), ("new.html", SECOND)]
    task = _submit(client, inputs)
    assert [link["file_id"] for link in pending_files(task)] == [task["files"][1]["file_id"]]
    assert task["files"][0]["status"] in {"pending", "running", "completed"} and task["files"][0].get("upload") is None
    _delete_source_references(client, app, old_references)
    duplicate = client.put(
        f"/v1/tasks/{task['task_id']}/uploads/{task['files'][0]['file_id']}",
        content=_UnreadableBody(),
        headers={"content-length": str(len(HTML)), "expect": "100-continue"},
    )
    assert duplicate.status_code == 200 and duplicate.headers["connection"] == "close"
    _upload(client, task, inputs)
    result = _terminal(client, task)
    assert result["status"] == "completed" and result["files"][0]["file_name"] == "renamed.html"
    cached = _submit(client, inputs)
    assert pending_files(cached) == [] and cached["status"] in {"pending", "running", "completed"}
    assert _terminal(client, cached)["status"] == "completed"
    assert _job_count(app) == 5


def test_no_hash_still_deduplicates_source_storage(service: tuple[TestClient, FastAPI]) -> None:
    """不发送 hash 时仍计算实际内容标识，相同内容只保留一份源字节。"""
    client, app = service
    first = _submit(client, [("first.html", HTML)], hashes=False)
    _upload(client, first, [("first.html", HTML)])
    assert _terminal(client, first)["status"] == "completed"
    second = _submit(client, [("second.html", HTML)], hashes=False)
    assert len(pending_files(second)) == 1
    _upload(client, second, [("second.html", HTML)])
    assert _terminal(client, second)["status"] == "completed"
    records = app.state.task_store._records
    assert records[first["task_id"]].file_states[0].upload.path == records[second["task_id"]].file_states[0].upload.path


def test_waiting_file_retains_deleted_source_views(service: tuple[TestClient, FastAPI]) -> None:
    """删除已解析源视图不影响已生成结果，也不阻止其他文件继续上传。"""
    client, app = service
    inputs = [("a.html", HTML), ("b.html", SECOND)]
    task = _submit(client, inputs, hashes=False)
    assert client.put(pending_files(task)[0].get("upload").get("url"), content=HTML).status_code == 200
    ready = _file_done(client, task, 0)
    _delete_source_references(client, app, _source_references(app, ready))
    assert client.put(pending_files(task)[1].get("upload").get("url"), content=SECOND).status_code == 200
    result = _terminal(client, task)
    assert result["status"] == "completed", result
    assert b"Preserved content" in _archive(client, result)["markdown.md"]


def test_failed_upload_can_retry_without_reuploading_ready_file(service: tuple[TestClient, FastAPI]) -> None:
    """校验失败可在期限内重传，不影响已经执行的其他文件。"""
    client, app = service
    inputs = [("a.html", HTML), ("b.html", SECOND)]
    task = _submit(client, inputs)
    assert client.put(pending_files(task)[0].get("upload").get("url"), content=HTML).status_code == 200
    failed = client.put(pending_files(task)[1].get("upload").get("url"), content=SECOND.replace(b"Second", b"Broken"))
    assert failed.status_code == 400 and failed.json().get("error")["code"] == "file_hash_mismatch"
    mismatch = client.put(pending_files(task)[1].get("upload").get("url"), content=SECOND + b"x")
    assert mismatch.status_code == 400 and mismatch.json().get("error")["code"] == "upload_size_mismatch"
    pending = _file_done(client, task, 0)
    assert pending["status"] == "waiting-file" and _job_count(app) == 1
    assert pending["files"][0]["status"] in {"pending", "running", "completed"}
    assert pending["files"][1]["status"] == "waiting-file"
    assert list(app.state.task_store._records[task["task_id"]].directory.iterdir()) == []
    _upload(client, pending, inputs)
    assert _terminal(client, task)["status"] == "completed" and _job_count(app) == 2


def test_concurrent_last_uploads_and_duplicate_put_submit_once(service: tuple[TestClient, FastAPI]) -> None:
    """并发上传与重复 PUT 共用提交门闩，每个文件只产生一个 Job。"""
    client, app = service
    task = _submit(client, [("a.html", HTML), ("b.html", SECOND)])

    async def upload_all() -> list[httpx.Response]:
        """在服务所属事件循环同时发起两个文件及同一文件的重复请求。"""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as asynchronous:
            return await asyncio.gather(
                asynchronous.put(pending_files(task)[0].get("upload").get("url"), content=HTML),
                asynchronous.put(pending_files(task)[1].get("upload").get("url"), content=SECOND),
                asynchronous.put(pending_files(task)[1].get("upload").get("url"), content=SECOND),
            )

    responses = client.portal.call(upload_all)
    assert all(response.status_code == 200 for response in responses)
    assert _terminal(client, task)["status"] == "completed"
    assert _job_count(app) == 2
    if hasattr(app.state, "worker_pool"):
        assert sum(worker.active_jobs for worker in app.state.worker_pool.workers) == 0


@pytest.mark.parametrize("expired", [False, True])
def test_cancel_and_expire_waiting_file_release_inputs(service: tuple[TestClient, FastAPI], expired: bool) -> None:
    """取消或上传过期只终结未完成文件，已生成结果继续可下载。"""
    client, app = service
    task = _submit(client, [("a.html", HTML), ("b.html", SECOND)])
    assert client.put(pending_files(task)[0].get("upload").get("url"), content=HTML).status_code == 200
    ready = _file_done(client, task, 0)
    record = app.state.task_store._records[task["task_id"]]
    if expired:
        record.expires_at = int(time.time()) - 1
        response = client.get(task_url(client, task))
        assert response.json()["files"][1].get("error")["code"] == "upload_expired"
        assert response.json()["files"][1]["status"] == "failed"
    else:
        response = client.post(task_url(client, task) + "/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == ("partial" if expired else "canceled")
    assert pending_files(response.json()) == [] and _job_count(app) == 1
    assert response.json()["files"][0].get("result_url") == ready["files"][0].get("result_url")
    assert b"Preserved content" in _archive(client, response.json())["markdown.md"]
    rejected = client.put(pending_files(task)[1].get("upload").get("url"), content=SECOND)
    assert rejected.status_code == (410 if expired else 409)
    if expired:
        assert client.put(pending_files(task)[0].get("upload").get("url"), content=HTML).status_code == 410

    async def wait_cleanup() -> None:
        """等待父任务持有的清理句柄，避免按定时猜测引用是否已释放。"""
        await record.operation

    client.portal.call(wait_cleanup)
    assert not record.directory.exists()
    if hasattr(app.state, "source_store"):
        assert all(not file.backend._prepared_sources and not file.backend._task_sources for file in record.file_states)
        assert not app.state.source_store._active_references
    else:
        assert all(file.backend._owned_files == [] for file in record.file_states)


def test_cancel_running_task_releases_execution_slot(
    service: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch
) -> None:
    """运行任务取消经过同一执行队列，HTTP 返回后不存在重复预占或悬挂解析。"""
    client, app = service
    started = threading.Event()

    async def blocked_parse(*args: Any, **kwargs: Any) -> None:
        """保持真实后台解析句柄可取消，避免依赖 GPU 或定时竞态。"""
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(api_server, "parse_async", blocked_parse)
    task = _submit(client, [("a.html", HTML)])
    _upload(client, task, [("a.html", HTML)])
    assert started.wait(5)
    response = client.post(task_url(client, task) + "/cancel")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "canceled"
    assert client.get(task_url(client, task)).json()["files"][0].get("result_url") is None
    deadline = time.monotonic() + 5
    operation = app.state.task_store._records[task["task_id"]].operation
    while operation is not None and not operation.done():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    if hasattr(app.state, "worker_pool"):
        assert sum(worker.active_jobs for worker in app.state.worker_pool.workers) == 0
    else:
        assert not app.state.execution_store._tasks


def test_status_never_reads_or_rebuilds_zip(service: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch) -> None:
    """产物按输入各构建一次，多次状态查询和下载不读取整包或重新渲染。"""
    from mineru.parser import task_outputs

    client, _ = service
    build = task_outputs.build_task_output
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> Path:
        """统计逐文件产物构建，避免轮询触发重复导出。"""
        nonlocal calls
        calls += 1
        return build(*args, **kwargs)

    monkeypatch.setattr(task_outputs, "build_task_output", counted)
    inputs = [("a.html", HTML), ("b.html", SECOND)]
    task = _submit(client, inputs)
    _upload(client, task, inputs)
    result = _terminal(client, task)
    assert result["status"] == "completed" and calls == 2

    def forbid_read(*args: Any, **kwargs: Any) -> bytes:
        """状态与流式下载禁止把完整结果读入内存。"""
        raise AssertionError("Result body was read into memory")

    monkeypatch.setattr(api_server.FileStore, "read_blob", forbid_read)
    monkeypatch.setattr(api_server.FileStore, "read_file_data", forbid_read)
    for _ in range(3):
        assert client.get(task_url(client, task)).json()["files"] == result["files"]
        assert "markdown.md" in _archive(client, result)
        assert "markdown.md" in _archive(client, result, 1)
    assert calls == 2


def test_partial_and_packaging_failure_have_honest_results(
    service: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch
) -> None:
    """解析或打包失败都按文件报告，不发布不可用链接，其他文件继续。"""
    from mineru.parser import task_outputs

    client, _ = service
    inputs = [("same.html", HTML), ("broken.pdf", b"not a pdf")]
    task = _submit(client, inputs)
    _upload(client, task, inputs)
    result = _terminal(client, task)
    assert result["status"] == "partial" and result["files"][1].get("error")
    assert result["files"][1].get("result_url") is None
    assert "markdown.md" in _archive(client, result)
    original = task_outputs.build_task_output

    def fail_packaging(result: Any, formats: Any, directory: Path) -> Path:
        """仅让第二个输入导出失败，验证文件级隔离及诚实的成功状态。"""
        if "Second content" in result.markdown():
            raise OSError("simulated full disk")
        return original(result, formats, directory)

    monkeypatch.setattr(task_outputs, "build_task_output", fail_packaging)
    good = [("a.html", HTML), ("b.html", SECOND)]
    task = _submit(client, good)
    _upload(client, task, good)
    failed = _terminal(client, task)
    assert failed["status"] == "partial" and failed["files"][1].get("result_url") is None
    assert failed["files"][1].get("error")["code"] == "result_packaging_failed"
    assert "markdown.md" in _archive(client, failed)


def test_input_validation_and_openapi(service: tuple[TestClient, FastAPI]) -> None:
    """请求与 OpenAPI 都使用严格 JSON 数组，旧格式与额外字段在接单前拒绝。"""
    client, app = service
    valid = {"file_name": "a.html", "bytes": len(HTML)}
    for manifest in (
        [],
        [valid] * 101,
        [{**valid, "bytes": 0}],
        [{**valid, "sha256sum": "invalid"}],
        [{**valid, "page_range": "invalid"}],
        json.dumps([valid]),
    ):
        assert client.post("/v1/tasks", json={"files": manifest}).status_code == 400
    for formats in ([], "default", ["markdown"], ["zip"], ["pdf"], [None]):
        assert client.post("/v1/tasks", json={"files": [valid], "output_formats": formats}).status_code == 400
    assert client.post("/v1/tasks?tier=flash", json={"files": [valid]}).status_code == 400
    assert client.post("/v1/tasks", json={"files": [valid], "wait_timeout": 300}).status_code == 400
    assert client.post("/v1/tasks", json={"files": []}).status_code == 400
    for body in (
        {"manifest": [valid]},
        {"files": [valid], "manifest": [valid]},
        {"files": [{"filename": "a.html", "bytes": len(HTML)}]},
        {"files": [{**valid, "filename": "a.html"}]},
        {"files": [{"url": "https://source.example/a.html"}]},
    ):
        assert client.post("/v1/tasks", json=body).status_code == 400
    assert client.post("/v1/tasks", content=b"{", headers={"content-type": "application/json"}).status_code == 400
    assert _job_count(app) == 0
    spec = client.get("/openapi.json").json()
    assert "/v1/file_parse" not in spec["paths"] and "/v1/tasks/{task_id}/result" not in spec["paths"]
    creation = spec["paths"]["/v1/tasks"]["post"]
    assert not creation.get("parameters")
    assert set(creation["requestBody"]["content"]) == {"application/json"}
    request_schema = creation["requestBody"]["content"]["application/json"]["schema"]
    assert request_schema["required"] == ["files"]
    assert "manifest" not in request_schema["properties"]
    files_schema = request_schema["properties"]["files"]
    assert files_schema["type"] == "array"
    assert files_schema["minItems"] == 1 and files_schema["maxItems"] == 100
    assert "file_name" in files_schema["items"]["required"]
    assert {"file_id", "client_file_id"}.isdisjoint(files_schema["items"]["properties"])
    assert "filename" not in files_schema["items"]["properties"]
    id_schema = spec["components"]["schemas"]["TaskFileDetail"]
    assert "file_id" in id_schema["required"]
    assert id_schema["properties"]["file_id"]["type"] == "string"
    assert set(spec["components"]["schemas"]["TaskUploadLink"]["properties"]) == {"url", "headers", "expires_at"}
    if hasattr(app.state, "execution_store"):
        metadata_schema = spec["components"]["schemas"]["FileObjectModel"]
        assert "file_name" in metadata_schema["required"]
        assert "filename" not in metadata_schema["properties"]
    assert "result_url" not in spec["components"]["schemas"]["TaskDetailResponse"]["properties"]
    assert "result_url" in spec["components"]["schemas"]["TaskFileDetail"]["properties"]
    assert "output_files" not in spec["components"]["schemas"]["TaskFileDetail"]["properties"]
    limits = client.get("/v1/usage").json()["limits"]
    assert limits["max_file_size_bytes"] is None and limits["max_pages_per_file"] is None
    assert limits["max_files_per_job"] == 100


@pytest.mark.parametrize("count", [1, 2, 3])
def test_server_file_ids_survive_upload_and_router_forwarding(service: tuple[TestClient, FastAPI], count: int) -> None:
    """每个同名输入在创建时获得不同 ID，查询与产物沿用本层公共 ID。"""
    client, app = service
    inputs = [("same.html", HTML)] * count
    task = _submit(client, inputs, hashes=False)
    expected_ids = [item["file_id"] for item in task["files"]]
    assert len(set(expected_ids)) == count and all(value.startswith("file-") for value in expected_ids)
    _upload(client, task, inputs)
    completed = _terminal(client, task)
    assert [item["file_id"] for item in completed["files"]] == expected_ids
    assert all(item["result_url"].endswith(f"/{item['file_id']}/content") for item in completed["files"])
    record = app.state.task_store._records[task["task_id"]]
    for item, state in zip(completed["files"], record.file_states):
        assert state.execution.file_id == item["file_id"]
        assert state.execution.result_file_id == item["file_id"]
        assert state.execution.execution_id == state.execution_id
    if hasattr(app.state, "execution_store"):
        for execution in app.state.execution_store._executions.values():
            assert execution.request.source.file_id != execution.request.file_id
    if hasattr(app.state, "test_transport"):
        worker_ids = {
            item.file_id
            for worker in app.state.test_transport.apps.values()
            for record in worker.state.task_store._records.values()
            for item in record.inputs
        }
        assert len(worker_ids) == count and worker_ids.isdisjoint(expected_ids)


def test_parent_task_aggregates_single_file_executions_and_usage(
    service: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch
) -> None:
    """父 Task 独占部分成功状态，用量仍按两个真实执行和一个成功文件累计。"""
    client, app = service
    original = api_server.parse_async

    async def infer(path: str, **kwargs: Any) -> ParseResult:
        """只让指定文件失败，其余文件经过真实 Flash 解析及产物保存。"""
        if b"intentional-failure" in Path(path).read_bytes():
            raise ValueError("intentional file failure")
        return await original(path, **kwargs)

    monkeypatch.setattr(api_server, "parse_async", infer)
    before = client.get("/v1/usage").json()["current"]
    inputs = [("good.html", HTML), ("bad.html", b"<html><body>intentional-failure</body></html>")]
    created = _submit(client, inputs, hashes=False)
    _upload(client, created, inputs)
    completed = _terminal(client, created)
    assert completed["status"] == "partial"
    assert completed["progress"] == {"completed": 1, "failed": 1, "total": 2}
    assert [item["status"] for item in completed["files"]] == ["completed", "failed"]
    states = app.state.task_store._records[created["task_id"]].file_states
    assert [state.execution.status for state in states] == ["completed", "failed"]
    after = client.get("/v1/usage").json()["current"]
    assert after["jobs_created"] - before["jobs_created"] == 2
    assert after["files_processed"] - before["files_processed"] == 1


@pytest.mark.parametrize("file_id", ["", "with space", "bad/id", "x" * 65, 123, False])
def test_invalid_explicit_file_ids_fail_before_creation(service: tuple[TestClient, FastAPI], file_id: object) -> None:
    """可选标识不放宽显式值校验，错误仍定位到新 files 字段且不创建任务。"""
    client, app = service
    response = client.post("/v1/tasks", json={"files": [{"file_name": "a.html", "bytes": len(HTML), "file_id": file_id}]})
    assert response.status_code == 400
    assert response.json().get("error")["param"] == "files.0.file_id"
    assert not app.state.task_store._records and _job_count(app) == 0


def test_files_metadata_and_attachment_keep_logical_names(service: tuple[TestClient, FastAPI]) -> None:
    """Tasks 和 Files 只返回新文件名字段，源文件与结果附件仍使用各自逻辑名称。"""
    client, app = service
    inputs = [("report.v2.html", HTML)]
    task = _submit(client, inputs)
    _upload(client, task, inputs)
    completed = _terminal(client, task)
    item = completed["files"][0]
    assert item["file_name"] == "report.v2.html" and "name" not in item and "filename" not in item
    result = client.get(item.get("result_url").removesuffix("/content")).json()
    if hasattr(app.state, "file_store"):
        source_id = next(iter(_source_references(app, completed)))
        source = client.get(f"/v1/files/{source_id}").json()
        assert source["file_name"] == "report.v2.html" and "filename" not in source
        assert client.get(f"/v1/files/{source_id}/content").status_code == 403
    else:
        assert all(route.payload["purpose"] == "parse_output" for route in app.state.registry.list("file"))
    assert result["file_name"] == "report.v2.zip" and "filename" not in result
    assert set(item.get("parse")) == {"tier", "duration_ms", "parser_version"}
    assert item.get("parse")["tier"] == "flash" and "tier" not in completed
    assert pending_files(completed) == []
    assert "uploads" not in completed and "include_model_output" not in completed
    download = client.get(item.get("result_url"))
    assert download.status_code == 200
    assert download.headers["content-disposition"] == 'attachment; filename="report.v2.zip"'


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"bytes": None},
        {"bytes": len(HTML)},
        {"sha256sum": hashlib.sha256(HTML).hexdigest()},
        {"bytes": len(HTML), "sha256sum": hashlib.sha256(HTML).hexdigest()},
        {"url": "https://source.example/opaque?signature=" + "a" * 3000},
    ],
)
def test_url_inputs_download_in_worker_and_report_actual_metadata(
    service: tuple[TestClient, FastAPI], url_downloads: list[httpx.Request], metadata: dict[str, Any]
) -> None:
    """无扩展名签名 URL 使用逻辑文件名，未知大小在下载后补齐且无需上传。"""
    client, app = service
    item = _url_manifest(**metadata)
    if hasattr(app.state, "worker_pool"):
        app.state.worker_pool.workers[0].features["sources"] = ["file_id"]
    response = client.post("/v1/tasks", json={"files": [item]})
    assert response.status_code == 202, response.text
    task = response.json()
    expected_id = task["files"][0]["file_id"]
    assert task["files"][0]["file_id"] == expected_id
    assert task["status"] == "waiting-file" and pending_files(task) == []
    assert "bytes" not in task["files"][0]
    assert task["files"][0]["status"] == "waiting-file" and task["files"][0].get("upload") is None
    result = _terminal(client, task)
    assert result["status"] == "completed"
    file = result["files"][0]
    assert file["file_id"] == expected_id
    assert file["file_name"] == "logical.html" and "name" not in file
    assert file.get("bytes") == len(HTML) and "sha256sum" not in file
    assert file["status"] == "completed" and "client_file_id" not in file and file.get("upload") is None
    assert file.get("url") == item.get("url")
    assert b"Preserved content" in _archive(client, result)["markdown.md"]
    assert len(url_downloads) == 1 and str(url_downloads[0].url) == item.get("url")
    assert url_downloads[0].method == "GET" and "authorization" not in url_downloads[0].headers
    assert client.put(f"/v1/tasks/{task['task_id']}/uploads/{expected_id}", content=HTML).status_code == 400
    if hasattr(app.state, "test_transport"):
        assert not any(method == "PUT" for _, method, _ in app.state.test_transport.calls)
        assert all(host == "worker-b" for host, method, path in app.state.test_transport.calls if path == "/v1/tasks")


def test_url_manifest_validation(service: tuple[TestClient, FastAPI], url_downloads: list[httpx.Request]) -> None:
    """大小只对 URL 可选，非法 URL、非正整数字节数及混合直传在接单前拒绝。"""
    client, app = service
    invalid = [
        {"file_name": "a.html"},
        {"file_name": "a.html", "bytes": None},
        *[_url_manifest(url=value) for value in ("", "file:///tmp/a.html", "https://", "ftp://source/a.html")],
        *[_url_manifest(bytes=value) for value in (0, -1, True, "10", 10.5)],
    ]
    for item in invalid:
        response = client.post("/v1/tasks", json={"files": [item]})
        assert response.status_code == 400, response.text
    response = client.post("/v1/tasks", files={"files": ("a.html", HTML)})
    assert response.status_code == 415
    assert _job_count(app) == 0 and url_downloads == []
    schema = client.get("/openapi.json").json()
    assert "upload_status" not in schema["components"]["schemas"]["TaskFileDetail"]["properties"]


def test_mixed_url_upload_partial_and_no_early_reservation(
    service: tuple[TestClient, FastAPI], url_downloads: list[httpx.Request]
) -> None:
    """URL 文件不等待客户端上传，错误和完成结果都按清单顺序独立返回。"""
    client, app = service
    manifest = [
        _url_manifest(file_name="same.html"),
        {"file_name": "same.html", "bytes": len(SECOND)},
        _url_manifest(url="https://source.example/missing"),
    ]
    response = client.post("/v1/tasks", json={"files": manifest})
    assert response.status_code == 202, response.text
    task = response.json()
    assert task["status"] == "waiting-file"
    assert [link["file_id"] for link in pending_files(task)] == [task["files"][1]["file_id"]]
    ready = _file_done(client, task, 0)
    assert b"Preserved content" in _archive(client, ready)["markdown.md"]
    assert ready["files"][1]["status"] == "waiting-file"
    assert client.put(pending_files(task)[0].get("upload").get("url"), content=SECOND).status_code == 200
    result = _terminal(client, task)
    assert result["status"] == "partial" and result["progress"] == {"completed": 2, "failed": 1, "total": 3}
    assert result["files"][0].get("bytes") == len(HTML)
    assert result["files"][2].get("error") and result["files"][2].get("result_url") is None
    assert b"Second content" in _archive(client, result, 1)["markdown.md"]
    assert [item["file_id"] for item in result["files"]] == [item["file_id"] for item in task["files"]]
    assert _job_count(app) == 3


@pytest.mark.parametrize(
    ("metadata", "code"),
    [
        ({"bytes": len(HTML) + 1}, "bytes_mismatch"),
        ({"sha256sum": "0" * 64}, "file_hash_mismatch"),
        ({"url": "https://source.example/empty"}, "invalid_request"),
        ({"url": "https://source.example/timeout"}, "parse_failed"),
    ],
)
def test_url_download_failures_are_per_file(
    service: tuple[TestClient, FastAPI], url_downloads: list[httpx.Request], metadata: dict[str, Any], code: str
) -> None:
    """下载及完整性错误进入文件错误；全失败不生成结果链接或实际元数据。"""
    client, _ = service
    response = client.post("/v1/tasks", json={"files": [_url_manifest(**metadata)]})
    assert response.status_code == 202
    result = _terminal(client, response.json())
    assert result["status"] == "failed" and result["files"][0].get("result_url") is None
    assert result["files"][0]["status"] == "failed" and result["files"][0].get("error")["code"] == code
    assert result["files"][0].get("bytes") == metadata.get("bytes")
    assert len(url_downloads) == 1


def test_url_hash_does_not_skip_download(service: tuple[TestClient, FastAPI], url_downloads: list[httpx.Request]) -> None:
    """即使相同 hash 已在上传缓存中，URL 来源仍每次重新读取并校验。"""
    client, _ = service
    inputs = [("cached.html", HTML)]
    cached = _submit(client, inputs)
    _upload(client, cached, inputs)
    assert _terminal(client, cached)["status"] == "completed"
    for _ in range(2):
        response = client.post("/v1/tasks", json={"files": [_url_manifest(sha256sum=hashlib.sha256(HTML).hexdigest())]})
        assert response.status_code == 202
        assert _terminal(client, response.json())["status"] == "completed"
    assert len(url_downloads) == 2


def test_url_download_obeys_worker_policy_and_limit(
    service: tuple[TestClient, FastAPI], url_downloads: list[httpx.Request]
) -> None:
    """HTTP 开关和 URL 下载预算继续由真正执行下载的服务控制。"""
    client, app = service
    item = _url_manifest(url="http://source.example/opaque")
    response = client.post("/v1/tasks", json={"files": [item]})
    if hasattr(app.state, "worker_pool"):
        assert response.status_code == 202
        assert _terminal(client, response.json())["status"] == "failed"
    else:
        assert response.status_code == 400
    assert url_downloads == []
    for worker in _execution_apps(app):
        worker.state.allow_http_source = True
        worker.state.max_url_bytes = len(HTML) - 1
        worker.state.url_timeout = 7
    response = client.post("/v1/tasks", json={"files": [item]})
    result = _terminal(client, response.json())
    assert result["status"] == "failed"
    assert "max_url_bytes" in result["files"][0].get("error")["message"]
    assert url_downloads[0].extensions["timeout"]["read"] == 7
    for worker in _execution_apps(app):
        worker.state.max_url_bytes = len(HTML)
    response = client.post("/v1/tasks", json={"files": [item]})
    assert _terminal(client, response.json())["status"] == "completed"


def test_url_html_preserves_source_context(
    service: tuple[TestClient, FastAPI], url_downloads: list[httpx.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    """URL HTML 解析继续使用源站地址和编码，而不是临时路径作为资源基址。"""
    client, _ = service
    contexts: list[Any] = []
    original = api_server.parse_async

    async def inspect_context(*args: Any, **kwargs: Any) -> api_server.ParseResult:
        """观察真实解析调用的来源上下文，再继续执行原解析器生成 ZIP。"""
        contexts.append(kwargs.get("source_context"))
        return await original(*args, **kwargs)

    monkeypatch.setattr(api_server, "parse_async", inspect_context)
    response = client.post("/v1/tasks", json={"files": [_url_manifest()]})
    assert _terminal(client, response.json())["status"] == "completed"
    assert len(contexts) == 1 and len(url_downloads) == 1
    assert contexts[0].source_uri == _url_manifest().get("url")
    assert contexts[0].transport_encoding == "utf-8"
    assert contexts[0].local_resource_root is None


def test_url_logical_filename_drives_flash_fallback(
    service: tuple[TestClient, FastAPI], url_downloads: list[httpx.Request]
) -> None:
    """逻辑 HTML 文件保留轻量格式回退，不能被 URL 路径的 PDF 后缀误导。"""
    client, _ = service
    response = client.post(
        "/v1/tasks",
        json={
            "files": [_url_manifest(url="https://source.example/document.pdf")],
            "tier": "standard",
        },
    )
    assert response.status_code == 202, response.text
    result = _terminal(client, response.json())
    assert result["status"] == "completed", result
    assert result["files"][0]["file_name"] == "logical.html"
    assert b"Preserved content" in _archive(client, result)["markdown.md"]
    assert len(url_downloads) == 1


def test_cancel_url_download_closes_stream_and_releases_worker(
    service: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch
) -> None:
    """下载中取消立即返回，并关闭源站流、释放 Job 与 Router 预占。"""
    client, app = service
    started, closed = threading.Event(), threading.Event()

    class SlowSource(httpx.AsyncByteStream):
        """模拟已经收到部分字节但源站迟迟不结束的下载。"""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            """收到取消时退出等待，避免后台下载在任务取消后继续运行。"""
            try:
                yield HTML[:10]
                started.set()
                await asyncio.Future()
            finally:
                closed.set()

        async def aclose(self) -> None:
            """记录 httpx 已关闭下载资源。"""
            closed.set()

    def respond(request: httpx.Request) -> httpx.Response:
        """返回实际异步流，使取消发生在读取源字节期间。"""
        return httpx.Response(200, stream=SlowSource(), headers={"content-type": "text/html"})

    def download_client(**kwargs: Any) -> httpx.AsyncClient:
        """仅隔离 URL 来源网络，不改变任务和 Router 的生命周期实现。"""
        return httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(api_server, "httpx", SimpleNamespace(AsyncClient=download_client))
    response = client.post("/v1/tasks", json={"files": [_url_manifest()]})
    assert response.status_code == 202
    task = response.json()
    assert started.wait(5)
    before = time.monotonic()
    canceled = client.post(task_url(client, task) + "/cancel")
    assert canceled.status_code == 200 and canceled.json()["status"] == "canceled"
    assert time.monotonic() - before < 2
    assert closed.wait(5)
    deadline = time.monotonic() + 5
    while any(worker.state.execution_store._tasks for worker in _execution_apps(app)):
        assert time.monotonic() < deadline, "Canceled URL job retained its execution slot"
        time.sleep(0.01)
    if hasattr(app.state, "worker_pool"):
        while any(worker.active_jobs for worker in app.state.worker_pool.workers):
            assert time.monotonic() < deadline, "Canceled URL task retained its worker reservation"
            time.sleep(0.01)
    final = client.get(task_url(client, task)).json()
    assert final["status"] == "canceled" and final["files"][0].get("result_url") is None
    assert final["files"][0].get("bytes") is None


def test_router_private_hash_cache_and_task_scope(tmp_path: Path) -> None:
    """Router 的缓存、任务与文件地址均隔离调用方，不能仅凭已知 hash 使用他人输入。"""
    workers = {"worker-a": api_server.create_app(upload_dir=str(tmp_path / "worker"), tiers=("flash",))}
    app = create_router(
        RouterSettings(upstream_urls=("http://worker-a",), local_gpus="none", worker_refresh_interval_seconds=0),
        transport=_WorkerTransport(workers),
    )
    with TestClient(app) as client:
        owner, stranger = {"authorization": "Bearer owner"}, {"authorization": "Bearer stranger"}
        task = _submit(client, [("a.html", HTML)], headers=owner)
        _upload(client, task, [("a.html", HTML)], headers=owner)
        result = _terminal(client, task, headers=owner)
        assert client.get(task_url(client, task), headers=stranger).status_code == 404
        assert client.get(result["files"][0].get("result_url"), headers=stranger).status_code == 404
        other = _submit(client, [("a.html", HTML)], headers=stranger)
        assert len(pending_files(other)) == 1
        assert client.post(task_url(client, other) + "/cancel", headers=stranger).status_code == 200
        owned = _submit(client, [("a.html", HTML)], headers=owner)
        assert not pending_files(owned)
        assert _terminal(client, owned, headers=owner)["status"] == "completed"


@pytest.mark.parametrize("root", ["", "/mineru"])
def test_task_links_respect_proxy_root(tmp_path: Path, root: str) -> None:
    """固定查询路由保留客户端部署前缀，上传和结果仍使用服务返回的链接。"""
    app = api_server.create_app(upload_dir=str(tmp_path), tiers=("flash",), api_key="secret")
    with TestClient(
        app, base_url="http://testserver" + root, root_path=root, headers={"Authorization": "Bearer secret"}
    ) as client:
        manifest = [{"file_name": "a.html", "bytes": len(HTML)}]
        response = client.post("/v1/tasks", json={"files": manifest, "tier": "flash"})
        assert response.status_code == 202
        task = response.json()
        assert task_url(client, task).startswith("http://testserver" + root + "/v1/tasks/")
        _upload(client, task, [("a.html", HTML)])
        result = _terminal(client, task)
        assert result["files"][0].get("result_url").startswith("http://testserver" + root + "/v1/files/")
        assert "markdown.md" in _archive(client, result)


@pytest.mark.parametrize("interrupted", [False, True])
def test_inflight_upload_may_finish_after_deadline(service: tuple[TestClient, FastAPI], interrupted: bool) -> None:
    """截止前受理的流可以传完；若跨截止时间断连，该文件失败且不能续传。"""
    from starlette.requests import ClientDisconnect

    client, app = service
    task = _submit(client, [("a.html", HTML), ("b.html", SECOND)], hashes=False)
    record = app.state.task_store._records[task["task_id"]]
    expires_at = pending_files(task)[0].get("upload")["expires_at"]
    assert 3598 <= expires_at - time.time() <= 3600
    assert pending_files(client.get(task_url(client, task)).json())[0].get("upload")["expires_at"] == expires_at

    async def cross_deadline() -> httpx.Response:
        """让真实 ASGI 上传跨过时限，并在受理流中途并发查询和尝试新 PUT。"""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as transport:

            async def body() -> AsyncIterator[bytes]:
                """第一块已经被接收后模拟到期，仍在传输的文件不能被清理。"""
                yield HTML[:10]
                assert record.file_states[0].upload_active
                record.expires_at = int(time.time()) - 1
                state = (await transport.get(task_url(client, task))).json()
                assert state["files"][0]["status"] == "waiting-file"
                assert state["files"][1]["status"] == "failed"
                assert pending_files(state) == []
                rejected = await transport.put(pending_files(task)[1].get("upload").get("url"), content=SECOND)
                assert rejected.status_code == 410
                if interrupted:
                    raise ClientDisconnect
                yield HTML[10:]

            return await transport.put(pending_files(task)[0].get("upload").get("url"), content=body())

    response = client.portal.call(cross_deadline)
    result = _terminal(client, task)
    if interrupted:
        assert response.status_code == 400
        assert result["status"] == "failed"
        assert all(file.get("error")["code"] == "upload_expired" for file in result["files"])
    else:
        assert response.status_code == 200
        assert result["status"] == "partial"
        assert b"Preserved content" in _archive(client, result)["markdown.md"]
    assert client.put(pending_files(task)[0].get("upload").get("url"), content=HTML).status_code == 410


def test_expired_upload_rejects_before_reading_body(service: tuple[TestClient, FastAPI]) -> None:
    """链接截止后连重传主体也不读取，查询不能延长或重建上传窗口。"""
    client, app = service
    task = _submit(client, [("a.html", HTML)], hashes=False)
    app.state.task_store._records[task["task_id"]].expires_at = int(time.time()) - 1
    response = client.put(pending_files(task)[0].get("upload").get("url"), content=_UnreadableBody())
    assert response.status_code == 410 and response.json().get("error")["code"] == "upload_expired"
    for _ in range(2):
        state = client.get(task_url(client, task)).json()
        assert state["status"] == "failed" and pending_files(state) == []
        assert state["files"][0]["status"] == "failed"
        assert state["files"][0].get("error")["code"] == "upload_expired"
    assert _job_count(app) == 0


def test_completed_child_retention_starts_after_parent_finishes(service: tuple[TestClient, FastAPI]) -> None:
    """子结果不能在父任务仍等待上传时过期，父任务完成后才开始统一保留期。"""
    client, app = service
    app.state.task_store.retention_seconds = 1
    if hasattr(app.state, "registry"):
        app.state.registry.retention_seconds = 1
    for worker in _execution_apps(app):
        worker.state.task_store.retention_seconds = 1
        worker.state.execution_store.retention_seconds = 1
        worker.state.file_store.retention_seconds = 1
    task = _submit(client, [("a.html", HTML), ("b.html", SECOND)], hashes=False)
    assert client.put(pending_files(task)[0].get("upload").get("url"), content=HTML).status_code == 200
    ready = _file_done(client, task, 0)
    result_url = ready["files"][0].get("result_url")
    time.sleep(1.05)
    for worker in _execution_apps(app):
        worker.state.execution_store.collect_expired()
        worker.state.file_store.collect_expired()
    assert client.get(task_url(client, task)).json()["files"][0]["status"] == "completed"
    assert client.get(result_url).status_code == 200
    assert client.put(pending_files(task)[1].get("upload").get("url"), content=SECOND).status_code == 200
    assert _terminal(client, task)["status"] == "completed"
    record = app.state.task_store._records[task["task_id"]]

    async def finish_retention() -> None:
        """等待统一保留起点，确保测试不依赖后台轮询间隔。"""
        await record.operation

    client.portal.call(finish_retention)
    assert client.get(result_url).status_code == 200
    time.sleep(1.05)
    assert client.get(task_url(client, task)).status_code == 404
    assert client.get(result_url).status_code == 404


def test_router_batch_uses_independent_workers_and_task_capabilities(tmp_path: Path) -> None:
    """单个父任务的文件按负载分发，旧 worker 即使声明 zip 也不能接新 Tasks。"""
    workers = {
        host: api_server.create_app(upload_dir=str(tmp_path / host), tiers=("flash",)) for host in ("worker-a", "worker-b")
    }
    app = create_router(
        RouterSettings(
            upstream_urls=("http://worker-a", "http://worker-b"), local_gpus="none", worker_refresh_interval_seconds=0
        ),
        transport=_WorkerTransport(workers),
    )
    with TestClient(app) as client:
        task = _submit(client, [("a.html", HTML), ("b.html", SECOND)])
        _upload(client, task, [("a.html", HTML), ("b.html", SECOND)])
        assert _terminal(client, task)["status"] == "completed"
        assert {route.worker_id for route in app.state.registry.list("job")} == {
            worker.worker_id for worker in app.state.worker_pool.workers
        }
        assert all(worker.active_jobs == 0 for worker in app.state.worker_pool.workers)
        for worker in app.state.worker_pool.workers:
            worker.features.pop("task_output_formats")
        response = client.post("/v1/tasks", json={"files": [{"file_name": "a.html", "bytes": len(HTML)}]})
        assert response.status_code == 503
        assert all(worker.active_jobs == 0 for worker in app.state.worker_pool.workers)


def test_worker_upload_renews_result_lease_after_long_transfer(tmp_path: Path) -> None:
    """Router 转传超过内部租约长度时，上传完成会保护刚生成的子结果。"""
    app = api_server.create_app(upload_dir=str(tmp_path), tiers=("flash",))
    with TestClient(app) as client:
        task = _submit(client, [("a.html", HTML)], hashes=False)
        record = app.state.task_store._records[task["task_id"]]

        async def upload_long_stream() -> None:
            """在流中模拟内部租约已到期，不改变独立的一小时上传窗口。"""

            async def content() -> AsyncIterator[bytes]:
                """接收第一块后推进租约边界，余下数据仍正常上传。"""
                yield HTML[:10]
                record.lease_expires_at = time.time() - 1
                yield HTML[10:]

            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as transport:
                response = await transport.put(
                    pending_files(task)[0].get("upload").get("url"),
                    content=content(),
                    headers={"x-mineru-task-lease-seconds": "120"},
                )
                assert response.status_code == 200
                assert record.lease_expires_at > time.time() + 110

        client.portal.call(upload_long_stream)
        result = _terminal(client, task)
        assert result["status"] == "completed" and record.retention_epoch is None
        assert client.get(result["files"][0].get("result_url")).status_code == 200
        assert client.get(task_url(client, task), headers={"x-mineru-task-lease-seconds": "0"}).status_code == 200


def test_router_ends_local_result_lease_even_if_worker_rejects_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """保留期结束后 worker 拒绝释放请求也不能永久保留过期的本地路由。"""
    from mineru.kit.router.tasks import maintain_task_result_leases

    worker = api_server.create_app(upload_dir=str(tmp_path), tiers=("flash",))
    transport = _WorkerTransport({"worker-a": worker})
    app = create_router(
        RouterSettings(upstream_urls=("http://worker-a",), local_gpus="none", worker_refresh_interval_seconds=0),
        transport=transport,
    )
    with TestClient(app) as client:
        task = _submit(client, [("a.html", HTML)])
        _upload(client, task, [("a.html", HTML)])
        result = _terminal(client, task)
        record = app.state.task_store._records[task["task_id"]]

        async def prepare() -> None:
            """等待父任务清理，并把其子路由保留期限推进到过去。"""
            await record.operation
            for route in app.state.registry.list("job"):
                route.lease.retain_until = time.time() - 1

        client.portal.call(prepare)
        original = transport.handle_async_request

        async def reject_release(request: httpx.Request) -> httpx.Response:
            """模拟 worker 的凭证轮换，结果租约释放返回明确的认证错误。"""
            if request.headers.get("x-mineru-task-lease-seconds") == "0":
                return httpx.Response(403, request=request)
            return await original(request)

        monkeypatch.setattr(transport, "handle_async_request", reject_release)
        client.portal.call(maintain_task_result_leases, app.state.registry, app.state.worker_pool)
        assert all(not route.lease.held for route in app.state.registry.list("job", include_expired=True))
        assert all(
            not route.lease.held for route in app.state.registry.list("file", include_expired=True)
        )
        assert result["files"][0].get("result_url")


def test_removed_lists_preserve_task_creation_query_and_cancel(service: tuple[TestClient, FastAPI]) -> None:
    """删除全局枚举后，已知任务的创建、查询和取消仍使用完整的新协议。"""
    client, _ = service
    task = _submit(client, [("input.html", HTML)])
    assert client.get("/v1/tasks").status_code == 405
    assert client.get("/v1/files").status_code == 404
    assert "uploads" not in task and "include_model_output" not in task
    assert {"tier", "status_url", "pending_uploads"}.isdisjoint(task)
    assert set(pending_files(task)[0].get("upload")) == {"url", "headers", "expires_at"}
    assert "/uploads/" in pending_files(task)[0].get("upload").get("url")
    assert pending_files(client.get(task_url(client, task)).json()) == pending_files(task)
    canceled = client.post(task_url(client, task) + "/cancel")
    assert canceled.status_code == 200
    assert canceled.json()["status"] == "canceled" and pending_files(canceled.json()) == []


def test_tasks_only_routes_and_health(service: tuple[TestClient, FastAPI]) -> None:
    """所有旧入口均消失，嵌套 Task 上传及独立 Files 仍然存在。"""
    client, _ = service
    schema = client.get("/openapi.json").json()
    assert not any(path.startswith(("/v1/uploads", "/v1/parse/jobs")) for path in schema["paths"])
    for method, path in [
        ("POST", "/v1/uploads"),
        ("GET", "/v1/uploads/a"),
        ("PUT", "/v1/uploads/a/content"),
        ("POST", "/v1/uploads/a/complete"),
        ("POST", "/v1/uploads/a/cancel"),
        ("POST", "/v1/parse/jobs"),
        ("GET", "/v1/parse/jobs"),
        ("GET", "/v1/parse/jobs/a"),
        ("DELETE", "/v1/parse/jobs/a"),
    ]:
        assert client.request(method, path).status_code == 404
    features = client.get("/v1/health").json()["features"]
    assert set(features["sources"]) == {"upload", "url"}
    assert "output_formats" not in features and "task_model_output" not in features
    assert set(features["task_output_formats"]) == {"default", "html", "docx", "latex"}
    assert "get" not in schema["paths"]["/v1/tasks"]
    assert "/v1/files" not in schema["paths"]
    assert {"TaskListItem", "TaskListResponse", "FileListResponse"}.isdisjoint(schema["components"]["schemas"])
    status_properties = schema["components"]["schemas"]["TaskDetailResponse"]["properties"]
    assert {"status_url", "tier", "pending_uploads", "uploads", "include_model_output"}.isdisjoint(status_properties)
    file_properties = schema["components"]["schemas"]["TaskFileDetail"]["properties"]
    assert {"sha256sum", "page_range", "client_file_id", "name", "upload_status"}.isdisjoint(file_properties)
    assert "upload" not in schema["components"]["schemas"]["TaskFileDetail"]["required"]
    assert "upload_method" not in schema["components"]["schemas"]["TaskUploadLink"]["properties"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("output_formats", [["default"], ["html"]])
def test_removed_model_output_option_is_rejected(
    service: tuple[TestClient, FastAPI], enabled: bool, output_formats: list[str]
) -> None:
    """旧选项即使为 false 也必须在创建任何任务之前报错，禁止静默降级。"""
    client, app = service
    response = client.post(
        "/v1/tasks",
        json={
            "files": [{"file_name": "input.html", "bytes": len(HTML)}],
            "tier": "flash",
            "output_formats": output_formats,
            "include_model_output": enabled,
        },
    )
    assert response.status_code == 400
    assert response.json().get("error")["param"] == "include_model_output"
    assert not app.state.task_store._records and _job_count(app) == 0


def test_router_lease_timeout_does_not_expire_other_worker_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """多个失联任务不能阻塞健康 worker 续租，较短的 worker 保留期也不提前删除产物。"""
    from mineru.kit.router import tasks as router_tasks

    workers = {
        host: api_server.create_app(upload_dir=str(tmp_path / host), tiers=("flash",), retention_seconds=1)
        for host in ("worker-a", "worker-b")
    }
    transport = _WorkerTransport(workers)
    app = create_router(
        RouterSettings(
            upstream_urls=("http://worker-a", "http://worker-b"),
            local_gpus="none",
            worker_refresh_interval_seconds=0,
            retention_seconds=3600,
        ),
        transport=transport,
    )
    with TestClient(app) as client:
        pool = app.state.worker_pool
        pool.workers[1].healthy = False
        for index in range(2):
            task = _submit(client, [(f"a{index}.html", HTML)])
            _upload(client, task, [(f"a{index}.html", HTML)])
            _terminal(client, task)
        pool.workers[0].healthy = False
        pool.workers[1].healthy = True
        task = _submit(client, [("b.html", HTML)])
        _upload(client, task, [("b.html", HTML)])
        result = _terminal(client, task)

        async def check_renewal() -> None:
            """缩短测试超时预算，验证健康续租完成时另一 worker 仍在等待响应。"""
            await asyncio.gather(*(record.operation for record in app.state.task_store._records.values()))
            original = transport.handle_async_request
            stalled = asyncio.Event()
            renewed = asyncio.Event()
            calls: list[str] = []

            async def handle(request: httpx.Request) -> httpx.Response:
                """只阻塞故障 worker，其他请求继续使用真实 ASGI 服务及结果租约。"""
                if request.url.path.startswith("/v1/tasks/"):
                    calls.append(request.url.host)
                    if request.url.host == "worker-a":
                        stalled.set()
                        await asyncio.Event().wait()
                    else:
                        await stalled.wait()
                        response = await original(request)
                        assert response.status_code == 200
                        renewed.set()
                        return response
                return await original(request)

            worker_record = next(iter(workers["worker-b"].state.task_store._records.values()))
            worker_record.lease_expires_at = time.time() + 0.1
            with monkeypatch.context() as scoped:
                scoped.setattr(transport, "handle_async_request", handle)
                scoped.setattr(router_tasks, "_TASK_LEASE_REFRESH_TIMEOUT_SECONDS", 0.05)
                await asyncio.wait_for(router_tasks.maintain_task_result_leases(app.state.registry, pool), timeout=1)
            assert renewed.is_set()
            assert calls.count("worker-a") == 1
            assert worker_record.lease_expires_at > time.time() + 100
            assert worker_record.retention_epoch is None

        client.portal.call(check_renewal)
        assert client.get(task_url(client, result)).json()["status"] == "completed"
        assert client.get(result["files"][0].get("result_url")).status_code == 200
