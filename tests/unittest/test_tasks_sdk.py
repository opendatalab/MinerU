"""Tasks SDK 的传输、单文件映射、恢复和产物契约；替代旧 Uploads/Jobs 客户端场景。"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest

from mineru.parser import ApiJobStatus, ApiTaskStatus, MinerUApiParser
from mineru.parser import api_client
from mineru.tasks.contracts import CreateTaskRequest
from mineru.parser.task_outputs import build_task_output
from mineru.parser.base import ParseResult
from mineru.types import MiddleJson, PageInfo, Tier
from mineru.integrations.docvortex import build_metadata
from docvortex.schema import Producer


def middle() -> MiddleJson:
    """生成严格协议数据，不依赖推理模型。"""
    return MiddleJson(
        pages=[PageInfo(page_idx=0)],
        is_full_document=True,
        metadata={"file_suffix": "pdf", "producer": Producer(name="mineru", version="4.0")},
        extensions=build_metadata(effort="flash", parse_mode="txt"),
    )


def zip_output() -> bytes:
    """构造真实 Middle JSON ZIP，供流式下载和损坏包测试使用。"""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("middle_json.json", middle().model_dump_json(by_alias=True))
    return stream.getvalue()


def task_snapshot(
    status: str = "completed", *, upload: bool = False, file_name: str = "file.pdf", size: int = 3
) -> dict[str, Any]:
    """提供逐文件内嵌上传描述，不伪造已移除的查询链接或顶层档位。"""
    return {
        "task_id": "task-one",
        "status": status,
        "output_formats": ["default"],
        "files": [
            {
                "file_id": "file-one",
                "file_name": file_name,
                "bytes": size,
                "status": status,
                "upload": {
                    "url": "https://storage.test/input",
                    "headers": {},
                    "expires_at": 9999999999,
                }
                if upload
                else None,
                "parse": {"tier": "flash", "duration_ms": 250, "parser_version": "test"} if status == "completed" else None,
                "result_url": "https://storage.test/result?signature=original" if status == "completed" else None,
            }
        ],
    }


def bind_transport(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    """同时注入同步/异步 HTTP 传输，保留真实 HTTPX 流和响应处理。"""
    client, async_client = httpx.Client, httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(api_client.httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs))
    monkeypatch.setattr(api_client.httpx, "AsyncClient", lambda **kwargs: async_client(transport=transport, **kwargs))
    monkeypatch.setattr(api_client, "_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(api_client, "_transport_retry_delay", lambda _: 0)


def parse(parser: MinerUApiParser, source: Path, asynchronous: bool, **kwargs: Any) -> ParseResult:
    """统一驱动两个 SDK 公开入口，避免只覆盖其中一条传输路径。"""
    return asyncio.run(parser.parse_async(source, **kwargs)) if asynchronous else parser.parse(source, **kwargs)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("prefix", ["", "/gateway/api"])
@pytest.mark.parametrize("tier", [None, "basic", "advanced"])
def test_task_query_uses_configured_prefix_without_status_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, prefix: str, tier: Tier | None
) -> None:
    """响应不含查询链接时仍能轮询，代理前缀和编码后的任务 ID 均不丢失。"""
    source = tmp_path / "file.pdf"
    source.write_bytes(b"pdf")
    queries: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        """区分固定 API 路径和跨源结果下载，禁止向对象存储泄漏凭据。"""
        if request.url.host == "storage.test":
            assert request.headers.get("authorization") is None
            return httpx.Response(200, content=zip_output())
        assert request.headers.get("authorization") == "Bearer secret"
        if request.method == "POST":
            assert request.url.path == prefix + "/v1/tasks"
            body = json.loads(request.content)
            if tier is None:
                assert "tier" not in body
            else:
                assert body["tier"] == tier
            snapshot = task_snapshot("pending")
        else:
            queries.append(request.url.raw_path.decode())
            assert request.url.raw_path.decode() == prefix + "/v1/tasks/task%20one"
            snapshot = task_snapshot()
        snapshot["task_id"] = "task one"
        assert "status_url" not in snapshot
        return httpx.Response(200, json=snapshot)

    bind_transport(monkeypatch, handle)
    result = parse(MinerUApiParser(api_url="https://api.test" + prefix, api_key="secret", tier=tier), source, asynchronous)
    assert len(result.pages) == 1 and len(queries) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_tasks_sdk_accepts_missing_upload_and_polls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    """创建回执省略上传动作时只轮询，成功后用详情下载，不触发重复提交或上传。"""
    source = tmp_path / "file.pdf"
    source.write_bytes(b"pdf")
    methods: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        """分别提供精简创建回执、查询详情和真实 ZIP 字节。"""
        methods.append(request.method)
        if request.method == "POST":
            return httpx.Response(
                202,
                json={
                    "task_id": "task-one",
                    "status": "waiting-file",
                    "files": [{"file_id": "file-one", "file_name": "file.pdf", "status": "waiting-file"}],
                },
            )
        if request.url.host == "api.test":
            return httpx.Response(200, json=task_snapshot())
        return httpx.Response(200, content=zip_output())

    bind_transport(monkeypatch, handle)
    assert len(parse(MinerUApiParser(api_url="https://api.test"), source, asynchronous).pages) == 1
    assert methods == ["POST", "GET", "GET"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("lost_upload_response", [False, True])
def test_tasks_sdk_upload_cache_and_lost_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, cached: bool, lost_upload_response: bool
) -> None:
    """同机也提交 hash；命中不 PUT，上传响应丢失后查询 ready 且不重传或重建任务。"""
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-example")
    calls: list[httpx.Request] = []
    state = task_snapshot(
        "completed" if cached else "waiting-file", upload=not cached, file_name=source.name, size=source.stat().st_size
    )

    def handle(request: httpx.Request) -> httpx.Response:
        """记录线上协议，不接受任何独立上传或 Jobs 路由。"""
        calls.append(request)
        if request.method == "POST":
            assert request.url.path == "/v1/tasks"
            body = json.loads(request.content)
            assert body["files"][0]["sha256sum"] == hashlib.sha256(source.read_bytes()).hexdigest()
            assert body["files"][0]["page_range"] == "1"
            assert body["ocr_mode"] == "txt" and body["tier"] == "flash"
            return httpx.Response(202, json=copy.deepcopy(state))
        if request.method == "PUT":
            assert request.content == source.read_bytes()
            assert request.headers["content-length"] == str(source.stat().st_size)
            assert "transfer-encoding" not in request.headers
            assert "authorization" not in request.headers
            state.update(task_snapshot(file_name=source.name, size=source.stat().st_size))
            if lost_upload_response:
                raise httpx.ReadError("response lost")
            return httpx.Response(200, json=state)
        if request.url.host == "api.test":
            return httpx.Response(200, json=state)
        assert request.url.query == b"signature=original"
        assert "authorization" not in request.headers
        return httpx.Response(200, content=zip_output())

    bind_transport(monkeypatch, handle)
    statuses, durations = [], []
    result = parse(
        MinerUApiParser(api_url="https://api.test", api_key="secret", tier="flash", ocr_mode="txt"),
        source,
        asynchronous,
        page_range="1",
        status_callback=statuses.append,
        duration_callback=durations.append,
    )
    assert len(result.pages) == 1 and durations == [250.0]
    assert sum(r.method == "POST" for r in calls) == 1
    assert sum(r.method == "PUT" for r in calls) == (0 if cached else 1)
    assert statuses == (["completed"] if cached else ["waiting-file", "completed"])


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "failure", ["submission", "query", "upload", "download", "expired_signature", "corrupt_zip", "redirect"]
)
def test_tasks_sdk_transport_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, failure: str
) -> None:
    """覆盖创建不重试、查询/传输有界重试、签名刷新、ZIP 校验和跨源重定向认证。"""
    source = tmp_path / "file.pdf"
    source.write_bytes(b"pdf")
    counts = {"POST": 0, "PUT": 0, "query": 0, "download": 0}
    state = task_snapshot("waiting-file" if failure == "upload" else "pending", upload=failure == "upload")

    def handle(request: httpx.Request) -> httpx.Response:
        """在指定阶段只注入一次故障，观测是否重复创建或泄露认证。"""
        if request.method == "POST":
            counts["POST"] += 1
            if failure == "submission":
                raise httpx.ReadError("unknown submission outcome")
            return httpx.Response(202, json=copy.deepcopy(state))
        if request.method == "PUT":
            counts["PUT"] += 1
            if counts["PUT"] == 1:
                raise httpx.WriteError("upload interrupted")
            state.update(task_snapshot())
            return httpx.Response(200, json=state)
        if request.url.host == "api.test":
            counts["query"] += 1
            if failure == "query" and counts["query"] == 1:
                raise httpx.ReadError("query interrupted")
            if failure != "upload":
                state.update(task_snapshot())
            return httpx.Response(200, json=copy.deepcopy(state))
        counts["download"] += 1
        assert "authorization" not in request.headers
        if counts["download"] == 1:
            if failure == "download":
                raise httpx.ReadError("download interrupted")
            if failure == "expired_signature":
                return httpx.Response(403, json={"error": {"code": "expired", "message": "expired"}})
            if failure == "corrupt_zip":
                return httpx.Response(200, content=b"invalid zip")
            if failure == "redirect":
                return httpx.Response(302, headers={"location": "https://cdn.test/result"})
        return httpx.Response(200, content=zip_output())

    bind_transport(monkeypatch, handle)
    parser = MinerUApiParser(api_url="https://api.test", api_key="secret", tier="flash")
    if failure == "submission":
        with pytest.raises(api_client._APITransportError):
            parse(parser, source, asynchronous)
    else:
        assert len(parse(parser, source, asynchronous).pages) == 1
    assert counts["POST"] == 1
    if failure == "upload":
        assert counts["PUT"] == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("task_id", "other"),
        ("task_id", ".."),
        ("output_formats", ["html"]),
        ("file_id", "file-another"),
        ("file_name", "another.pdf"),
        ("bytes", 4),
    ],
)
def test_tasks_sdk_rejects_changed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, field: str, value: Any
) -> None:
    """拒绝改变身份或输出协议的响应，不尝试读取不匹配产物。"""
    source = tmp_path / "file.pdf"
    source.write_bytes(b"pdf")

    def handle(request: httpx.Request) -> httpx.Response:
        """提交合法，后续查询注入字段变化。"""
        response = task_snapshot("pending") if request.method == "POST" else task_snapshot()
        if request.method != "POST":
            if field in {"file_id", "file_name", "bytes"}:
                response["files"][0][field] = value
            else:
                response[field] = value
        return httpx.Response(200, json=response)

    bind_transport(monkeypatch, handle)
    with pytest.raises(api_client._V1APIError, match="invalid_response"):
        parse(MinerUApiParser(api_url="https://api.test"), source, asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("state", ["failed", "canceled"])
@pytest.mark.parametrize("fields", [{}, {"param": None}, {"param": "tier"}])
def test_tasks_sdk_preserves_file_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, state: str, fields: dict[str, Any]
) -> None:
    """优先保留逐文件错误码与字段，不统一改写为 parse_failed。"""
    source = tmp_path / "file.pdf"
    source.write_bytes(b"pdf")
    snapshot = task_snapshot(state)
    if state == "failed":
        snapshot["files"][0]["error"] = {"code": "engine_dead", "message": "engine stopped", **fields}
    bind_transport(monkeypatch, lambda _: httpx.Response(202, json=snapshot))
    with pytest.raises(api_client._V1APIError, match="engine_dead" if state == "failed" else "canceled") as error:
        parse(MinerUApiParser(api_url="https://api.test"), source, asynchronous)
    assert error.value.param == (fields.get("param") if state == "failed" else None)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("fields", [{}, {"param": None}, {"param": "api_key"}])
def test_tasks_sdk_preserves_http_errors_without_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, fields: dict[str, Any]
) -> None:
    """同步和异步入口都只依赖错误码和说明，可选字段缺省或为空时保持可用。"""
    source = tmp_path / "file.pdf"
    source.write_bytes(b"pdf")
    detail = {"code": "invalid_api_key", "message": "Invalid API key", **fields}
    bind_transport(monkeypatch, lambda _: httpx.Response(401, json={"error": detail}))
    with pytest.raises(api_client._V1APIError) as error:
        parse(MinerUApiParser(api_url="https://api.test"), source, asynchronous)
    assert error.value.code == detail["code"]
    assert error.value.message == detail["message"]
    assert error.value.param == fields.get("param")


def test_task_output_preserves_middle_without_model_output(tmp_path: Path) -> None:
    """本地保存和远端 ZIP 往返均只交付语义结果，不暴露内部模型数据。"""
    from mineru.kit.common import save_parse_result

    result = ParseResult(middle_json=middle())
    output = build_task_output(result, ["default"], tmp_path / "api")
    restored = api_client._parse_result_from_zip_bytes(output, include_images=True)
    assert restored.middle_json == result.middle_json
    assert not hasattr(result, "_model_output") and not hasattr(restored, "_model_output")
    local_output = tmp_path / "local.zip"
    save_parse_result(result, local_output, "zip")
    for target in (output, local_output):
        with zipfile.ZipFile(target) as archive:
            assert {"middle_json.json", "markdown.md", "structured_content.json"}.issubset(archive.namelist())
            assert "model_output.json" not in archive.namelist()


@pytest.mark.parametrize("enabled", [False, True])
def test_removed_model_output_option_is_rejected_by_sdk_and_schema(enabled: bool) -> None:
    """SDK 构造和任务请求模型都拒绝已删除选项，不提供旧调用兼容。"""
    with pytest.raises(TypeError, match="include_model_output"):
        MinerUApiParser(include_model_output=enabled)
    with pytest.raises(ValueError, match="include_model_output"):
        CreateTaskRequest(files=[{"file_name": "a.pdf", "bytes": 1}], include_model_output=enabled)
    assert ApiJobStatus is ApiTaskStatus


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["query", "download", "length", "corrupt_zip"])
def test_tasks_sdk_retries_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, failure: str
) -> None:
    """持续失败最多三次，不以无限重试掩盖断连或损坏产物。"""
    source = tmp_path / "file.pdf"
    source.write_bytes(b"pdf")
    attempts = []

    def handle(request: httpx.Request) -> httpx.Response:
        """按阶段产生持续故障，创建请求仍只提交一次。"""
        if request.method == "POST":
            return httpx.Response(202, json=task_snapshot("pending" if failure == "query" else "completed"))
        if request.url.host == "api.test":
            if failure == "query":
                attempts.append(request)
                raise httpx.ReadTimeout("unreachable")
            return httpx.Response(200, json=task_snapshot())
        attempts.append(request)
        if failure == "download":
            raise httpx.ReadTimeout("unreachable")
        if failure == "length":
            return httpx.Response(200, content=zip_output(), headers={"content-length": str(len(zip_output()) + 10)})
        return httpx.Response(200, content=b"broken zip")

    bind_transport(monkeypatch, handle)
    with pytest.raises((api_client._APITransportError, api_client._V1APIError)):
        parse(MinerUApiParser(api_url="https://api.test"), source, asynchronous)
    assert len(attempts) == 3


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("field,value", [("task_id", None), ("status", {}), ("files", []), ("task_id", ".")])
def test_tasks_sdk_rejects_malformed_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, field: str, value: Any
) -> None:
    """非法成功响应立即失败，不进行上传、轮询或未知地址下载。"""
    source = tmp_path / "file.pdf"
    source.write_bytes(b"pdf")
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        """只允许一次创建请求。"""
        calls.append(request)
        return httpx.Response(202, json={**task_snapshot(), field: value})

    bind_transport(monkeypatch, handle)
    with pytest.raises(api_client._V1APIError, match="invalid_response"):
        parse(MinerUApiParser(api_url="https://api.test"), source, asynchronous)
    assert len(calls) == 1


def test_file_parse_identity_uses_only_current_options() -> None:
    """恢复身份由当前有效参数计算，已删除选项在模型入口明确拒绝。"""
    from mineru.kit.file_parse.models import ParseOptions, fingerprint

    options = ParseOptions()
    assert "include_model_output" not in options.model_dump()
    identity = {"filename": "a.pdf", "size": 3, "sha256sum": "a" * 64, "options": options.model_dump()}
    assert (
        fingerprint("a.pdf", 3, "a" * 64, options)
        == hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    )
    for enabled in (False, True):
        with pytest.raises(ValueError, match="include_model_output"):
            ParseOptions(include_model_output=enabled)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status,failures,expected_queries", [(429, 1, 2), (503, 1, 2), (429, 5, 3), (503, 5, 3), (400, 1, 1)])
def test_tasks_sdk_retries_transient_poll_responses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    status: int,
    failures: int,
    expected_queries: int,
) -> None:
    """轮询按 Retry-After 重试临时响应，耗尽后保留错误，永久错误不重试。"""
    source = tmp_path / "input.pdf"
    source.write_bytes(b"pdf")
    counts = {"post": 0, "query": 0}
    delays: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        """首次创建成功，按预算注入轮询故障，之后返回可读取的完整结果。"""
        if request.method == "POST":
            counts["post"] += 1
            return httpx.Response(202, json=task_snapshot("pending", file_name=source.name))
        if request.url.host == "api.test":
            counts["query"] += 1
            if counts["query"] <= failures:
                return httpx.Response(
                    status, headers={"Retry-After": "2"}, json={"error": {"code": "temporary", "message": "retry later"}}
                )
            return httpx.Response(200, json=task_snapshot(file_name=source.name))
        return httpx.Response(200, content=zip_output())

    async def sleep(delay: float) -> None:
        """记录异步退避时间，不让协议测试实际等待。"""
        delays.append(delay)

    bind_transport(monkeypatch, handle)
    monkeypatch.setattr(api_client.time, "sleep", delays.append)
    monkeypatch.setattr(api_client.asyncio, "sleep", sleep)
    parser = MinerUApiParser(api_url="https://api.test", tier="flash")
    if status == 400 or failures > 3:
        with pytest.raises(api_client._V1APIError) as error:
            parse(parser, source, asynchronous)
        assert error.value.code == "temporary"
    else:
        assert len(parse(parser, source, asynchronous).pages) == 1
    assert counts == {"post": 1, "query": expected_queries}
    assert [delay for delay in delays if delay] == [2] * (expected_queries - 1)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_tasks_sdk_does_not_retry_http_submission_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    """即使服务端建议重试，非幂等创建的 HTTP 503 也不能触发重复任务。"""
    source = tmp_path / "input.pdf"
    source.write_bytes(b"pdf")
    calls: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        """模拟创建响应不确定，记录是否重复发送 POST。"""
        calls.append(request.method)
        return httpx.Response(503, headers={"Retry-After": "0"}, json={"error": {"code": "busy", "message": "busy"}})

    bind_transport(monkeypatch, handle)
    with pytest.raises(api_client._V1APIError):
        parse(MinerUApiParser(api_url="https://api.test"), source, asynchronous)
    assert calls == ["POST"]


@pytest.mark.parametrize(
    "header,expected",
    [("2", 2), ("Thu, 22 Oct 2026 00:00:05 GMT", 5), ("bad", 0.25), ("-2", 0.25), ("61", None), ("9" * 5000, None)],
)
def test_tasks_sdk_retry_after_formats(monkeypatch: pytest.MonkeyPatch, header: str, expected: float | None) -> None:
    """支持 HTTP 日期与秒数；非法值退避，过长等待返回原错误且不提前重试。"""
    from datetime import datetime, timezone

    monkeypatch.setattr(api_client.time, "time", lambda: datetime(2026, 10, 22, tzinfo=timezone.utc).timestamp())
    monkeypatch.setattr(api_client, "_transport_retry_delay", lambda _: 0.25)
    assert api_client._http_retry_delay(httpx.Response(429, headers={"Retry-After": header}), 1) == expected
