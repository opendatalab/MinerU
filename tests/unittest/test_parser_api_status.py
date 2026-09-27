"""同步和异步 V1 parser 的状态观察回调合同。"""

from docvortex.schema import Producer
from mineru.integrations.docvortex import build_metadata

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from mineru.parser import ApiJobStatus, MinerUApiParser, ParseResult
from mineru.parser import api_client
from mineru.types import MiddleJson


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("callback_raises", [False, True])
@pytest.mark.parametrize(
    "statuses",
    [
        ("queued", "running", "running", "completed"),
        ("completed",),
        ("queued", "completed"),
        ("queued", "partial"),
        ("queued", "running", "failed"),
        ("queued", "canceled"),
    ],
)
def test_parse_notifies_submission_and_polls_before_result_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_async: bool,
    callback_raises: bool,
    statuses: tuple[ApiJobStatus, ...],
) -> None:
    """验证六种真实状态、固定轮询、直接完成和回调异常隔离，通知不延迟到结果下载之后。"""
    callback_errors: list[str] = []
    monkeypatch.setattr(api_client.logger, "exception", callback_errors.append)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="flash")
    parser._source_features = {"local"}
    notifications: list[ApiJobStatus] = []
    durations: list[float] = []
    delays: list[float] = []
    replies = iter(statuses)
    methods: list[str] = []
    result = ParseResult(
        middle_json=MiddleJson(
            pages=[],
            is_full_document=True,
            metadata={"file_suffix": "pdf", "producer": Producer(name="mineru", version="4.0.0")},
            extensions=build_metadata(
                effort="flash",
                parse_mode="txt",
            ),
        )
    )

    def respond(request: httpx.Request) -> httpx.Response:
        """只模拟创建和轮询任务，保留正式 HTTP 调用及状态传播路径。"""
        methods.append(request.method)
        return httpx.Response(
            200, json={"job_id": "job_test", "status": next(replies), "files": [{"parse": {"duration_ms": 250}}]}
        )

    transport = httpx.MockTransport(respond)
    sync_client, async_client = httpx.Client, httpx.AsyncClient

    def create_sync_client(**kwargs: Any) -> httpx.Client:
        """注入同步 HTTP transport，禁止测试访问真实服务。"""
        return sync_client(transport=transport, **kwargs)

    def create_async_client(**kwargs: Any) -> httpx.AsyncClient:
        """注入异步 HTTP transport，禁止测试访问真实服务。"""
        return async_client(transport=transport, **kwargs)

    async def sleep_async(delay: float) -> None:
        """记录轮询间隔而不真实等待。"""
        delays.append(delay)

    def on_status(status: ApiJobStatus) -> None:
        """记录本次调用的状态，按参数模拟不可靠的观察者。"""
        notifications.append(status)
        if callback_raises:
            raise RuntimeError("observer failed")

    def build_result(job: dict[str, Any], file_name: str) -> ParseResult:
        """确认所有状态在开始下载/构建结果之前已经通知。"""
        assert notifications == list(statuses)
        assert durations == ([250.0] if statuses[-1] in {"completed", "partial"} else [])
        assert file_name == source.name
        api_client._raise_for_terminal_job_error(job)
        return result

    async def build_result_async(job: dict[str, Any], file_name: str) -> ParseResult:
        """为异步解析复用相同的终态和通知时序断言。"""
        return build_result(job, file_name)

    monkeypatch.setattr(httpx, "Client", create_sync_client)
    monkeypatch.setattr(httpx, "AsyncClient", create_async_client)
    monkeypatch.setattr(api_client.time, "sleep", delays.append)
    monkeypatch.setattr(api_client.asyncio, "sleep", sleep_async)
    monkeypatch.setattr(parser, "_build_result", build_result)
    monkeypatch.setattr(parser, "_async_build_result", build_result_async)

    def parse() -> ParseResult:
        """统一执行两个公共入口，确保回调参数一路透传。"""
        if use_async:
            return asyncio.run(parser.parse_async(source, status_callback=on_status, duration_callback=durations.append))
        return parser.parse(source, status_callback=on_status, duration_callback=durations.append)

    if statuses[-1] in ("failed", "canceled"):
        with pytest.raises(Exception, match=f"ended with status {statuses[-1]}"):
            parse()
    else:
        assert parse() is result
    assert notifications == list(statuses)
    assert methods == ["POST", *(["GET"] * (len(statuses) - 1))]
    assert delays == [1] * (len(statuses) - 1)
    assert callback_errors == (["Parse job status callback failed"] * len(statuses) if callback_raises else [])


@pytest.mark.parametrize("value", [None, True, "250", -1, float("nan"), float("inf"), 10**400])
def test_invalid_or_missing_duration_does_not_invent_zero(value: object) -> None:
    """旧服务缺失字段或非法数值不伪造零耗时，也不破坏结果下载。"""
    durations: list[float] = []
    api_client._notify_job_duration({"status": "completed", "files": [{"parse": {"duration_ms": value}}]}, durations.append)
    assert durations == []


def test_duration_callback_failure_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """耗时观察者异常仅写日志，不让已完成的解析任务失败。"""
    errors: list[str] = []
    monkeypatch.setattr(api_client.logger, "exception", errors.append)

    def fail(_duration: float) -> None:
        """模拟显示层的耗时处理异常。"""
        raise RuntimeError("observer failed")

    api_client._notify_job_duration({"status": "completed", "files": [{"parse": {"duration_ms": 250}}]}, fail)
    assert errors == ["Parse job duration callback failed"]
