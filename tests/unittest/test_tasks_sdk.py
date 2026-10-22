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
from mineru.parser.task_api import CreateTaskRequest
from mineru.parser.task_outputs import build_task_output
from mineru.parser.base import ParseResult
from mineru.types import MiddleJson, ModelJson, PageInfo
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


def task_snapshot(status: str = "completed", *, upload: bool = False) -> dict[str, Any]:
    """提供可独立调整状态和上传准备情况的单文件任务快照。"""
    return {
        "task_id": "task-one",
        "status_url": "https://api.test/v1/tasks/task-one",
        "status": status,
        "output_formats": ["default"],
        "include_model_output": False,
        "files": [
            {
                "client_file_id": "file",
                "status": status,
                "upload_status": "pending" if upload else "ready",
                "parse": {"duration_ms": 250},
                "result_url": "https://storage.test/result?signature=original" if status == "completed" else None,
            }
        ],
        "uploads": (
            [
                {
                    "client_file_id": "file",
                    "upload_method": "PUT",
                    "upload_url": "https://storage.test/input",
                    "upload_headers": {},
                    "expires_at": 9999999999,
                }
            ]
            if upload
            else []
        ),
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
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("lost_upload_response", [False, True])
def test_tasks_sdk_upload_cache_and_lost_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, cached: bool, lost_upload_response: bool
) -> None:
    """同机也提交 hash；命中不 PUT，上传响应丢失后查询 ready 且不重传或重建任务。"""
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-example")
    calls: list[httpx.Request] = []
    state = task_snapshot("completed" if cached else "waiting_upload", upload=not cached)

    def handle(request: httpx.Request) -> httpx.Response:
        """记录线上协议，不接受任何独立上传或 Jobs 路由。"""
        calls.append(request)
        if request.method == "POST":
            assert request.url.path == "/v1/tasks"
            body = json.loads(request.content)
            assert body["manifest"][0]["sha256sum"] == hashlib.sha256(source.read_bytes()).hexdigest()
            assert body["manifest"][0]["page_range"] == "1"
            assert body["ocr_mode"] == "txt" and body["tier"] == "flash"
            return httpx.Response(202, json=copy.deepcopy(state))
        if request.method == "PUT":
            assert request.content == source.read_bytes()
            assert "authorization" not in request.headers
            state.update(task_snapshot())
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
    assert statuses == (["completed"] if cached else ["waiting_upload", "completed"])


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
    state = task_snapshot("waiting_upload" if failure == "upload" else "queued", upload=failure == "upload")

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
        ("status_url", "https://evil.test/metadata"),
        ("output_formats", ["html"]),
        ("include_model_output", True),
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
        response = task_snapshot("queued") if request.method == "POST" else {**task_snapshot(), field: value}
        return httpx.Response(200, json=response)

    bind_transport(monkeypatch, handle)
    with pytest.raises(api_client._V1APIError, match="invalid_response"):
        parse(MinerUApiParser(api_url="https://api.test"), source, asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("state", ["failed", "canceled"])
def test_tasks_sdk_preserves_file_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, state: str
) -> None:
    """优先保留逐文件错误码与字段，不统一改写为 parse_failed。"""
    source = tmp_path / "file.pdf"
    source.write_bytes(b"pdf")
    snapshot = task_snapshot(state)
    if state == "failed":
        snapshot["files"][0]["error"] = {"code": "engine_dead", "message": "engine stopped", "param": "tier"}
    bind_transport(monkeypatch, lambda _: httpx.Response(202, json=snapshot))
    with pytest.raises(api_client._V1APIError, match="engine_dead" if state == "failed" else "canceled"):
        parse(MinerUApiParser(api_url="https://api.test"), source, asynchronous)


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("has_model", [False, True])
def test_task_model_output_contract(tmp_path: Path, include: bool, has_model: bool) -> None:
    """模型输出显式启用且真实存在时才写入，默认包不变。"""
    model = (
        ModelJson(
            pages=[[{"raw": "model"}]],
            page_index_map=[0],
            metadata={"file_suffix": "pdf", "producer": Producer(name="mineru", version="4.0")},
            extensions=build_metadata(effort="flash", parse_mode="txt"),
        )
        if has_model
        else None
    )
    result = ParseResult(middle_json=middle(), _model_output=model)
    output = build_task_output(result, ["default"], tmp_path, include_model_output=include)
    restored = api_client._parse_result_from_zip_bytes(output, include_images=True, include_model_output=include)
    assert (restored._model_output is not None) == (include and has_model)
    with zipfile.ZipFile(output) as archive:
        assert ("model_output.json" in archive.namelist()) == (include and has_model)


def test_task_model_output_requires_default() -> None:
    """不把原本单文件的 HTML/DOCX 请求隐式改成 ZIP。"""
    with pytest.raises(ValueError, match="requires default"):
        CreateTaskRequest(
            manifest=[{"client_file_id": "file", "filename": "a.pdf", "bytes": 1}],
            output_formats=["html"],
            include_model_output=True,
        )
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
            return httpx.Response(202, json=task_snapshot("queued" if failure == "query" else "completed"))
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
@pytest.mark.parametrize(
    "field,value", [("task_id", None), ("status", {}), ("files", []), ("status_url", "file:///tmp/result")]
)
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


def test_task_model_output_changes_wrapper_identity() -> None:
    """同步包装服务区分模型输出选项，不能复用另一种结果的 request_id。"""
    from mineru.kit.file_parse.models import ParseOptions, fingerprint

    ordinary = ParseOptions()
    with_model = ParseOptions(include_model_output=True)
    assert ordinary.model_dump()["include_model_output"] is False
    prior_options = ordinary.model_dump(exclude={"include_model_output"})
    prior_value = {"filename": "a.pdf", "size": 3, "sha256sum": "a" * 64, "options": prior_options}
    assert (
        fingerprint("a.pdf", 3, "a" * 64, ordinary)
        == hashlib.sha256(json.dumps(prior_value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    )
    assert fingerprint("a.pdf", 3, "a" * 64, ordinary) != fingerprint("a.pdf", 3, "a" * 64, with_model)
    with pytest.raises(ValueError, match="requires default"):
        ParseOptions(output_formats=["docx"], include_model_output=True)
