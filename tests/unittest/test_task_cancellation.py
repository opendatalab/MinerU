"""协程取消到 Tasks 取消接口的身份、竞争和有界清理回归。"""

from __future__ import annotations

import asyncio
import copy
import logging
from pathlib import Path

import httpx
import pytest

from mineru.parser import api_client
from mineru.parser.api_client import MinerUApiParser
from tests.unittest.test_tasks_sdk import bind_transport, task_snapshot


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stage", ["create", "query", "upload", "pending", "running", "download"])
def test_cancel_targets_original_task_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool, stage: str) -> None:
    """从真实异步入口取消各个传输阶段，验证默认不取消、前缀鉴权和重复取消隔离。"""
    source = tmp_path / "file.pdf"
    source.write_bytes(b"pdf")

    async def scenario() -> None:
        """用可控网络响应阻塞指定阶段，并保证所有受保护任务在退出前收尾。"""
        entered, release, cancel_entered, cancel_release = (asyncio.Event() for _ in range(4))
        requests: list[httpx.Request] = []
        queries = 0
        snapshot = task_snapshot("waiting-file" if stage == "upload" else "pending", upload=stage == "upload")
        snapshot["task_id"] = "task one"

        async def handle(request: httpx.Request) -> httpx.Response:
            """只阻塞选中的请求；取消响应保留到第二次本地取消之后。"""
            nonlocal queries
            requests.append(request)
            if request.url.path.endswith("/cancel"):
                assert request.url.raw_path == b"/gateway/api/v1/tasks/task%20one/cancel"
                assert request.headers["authorization"] == "Bearer secret"
                cancel_entered.set()
                await cancel_release.wait()
                return httpx.Response(200, json={"task_id": "task one", "status": "canceled"})
            if request.method == "POST":
                if stage == "create":
                    entered.set()
                    await release.wait()
                return httpx.Response(202, json=copy.deepcopy(snapshot))
            if request.method == "PUT" or request.url.host == "storage.test":
                entered.set()
                await release.wait()
                raise AssertionError("canceled transfer resumed")
            queries += 1
            if stage == "query" or (stage in {"pending", "running"} and queries > 1):
                entered.set()
                await release.wait()
                raise AssertionError("canceled query resumed")
            current = copy.deepcopy(snapshot)
            if stage in {"running", "download"}:
                current = task_snapshot("completed" if stage == "download" else "running")
                current["task_id"] = "task one"
            return httpx.Response(200, json=current)

        bind_transport(monkeypatch, handle)
        operation = asyncio.create_task(
            MinerUApiParser(api_url="https://api.test/gateway/api", api_key="secret").parse_async(
                source, cancel_on_cancel=enabled
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            operation.cancel()
            await asyncio.sleep(0)
            if enabled and stage == "create":
                assert not operation.done()
                release.set()
            if enabled and stage != "download":
                await asyncio.wait_for(cancel_entered.wait(), 3)
                operation.cancel()
                await asyncio.sleep(0)
                assert not operation.done()
            cancel_release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(operation, 3)
            assert sum(request.method == "POST" and not request.url.path.endswith("/cancel") for request in requests) == 1
            cancels = [request for request in requests if request.url.path.endswith("/cancel")]
            assert len(cancels) == int(enabled and stage != "download")
            assert all(request.method != "DELETE" for request in requests)
        finally:
            release.set()
            cancel_release.set()
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["404", "409", "bad409", "500", "timeout", "wrong_identity"])
def test_cancel_failure_preserves_original_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, outcome: str
) -> None:
    """确认取消仅一次、短超时生效，正常竞争不告警，失败不会替换原始 CancelledError。"""
    source = tmp_path / "file.pdf"
    source.write_bytes(b"pdf")
    monkeypatch.setattr(api_client, "_TASK_CANCEL_TIMEOUT_SECONDS", 0.03)

    async def scenario() -> None:
        """先取得合法任务身份，再在首次查询时取消。"""
        entered = asyncio.Event()
        cancellations = 0

        async def handle(request: httpx.Request) -> httpx.Response:
            """注入取消接口的终态竞争、错误和网络无响应。"""
            nonlocal cancellations
            if request.url.path.endswith("/cancel"):
                cancellations += 1
                if outcome == "timeout":
                    await asyncio.Future()
                if outcome == "wrong_identity":
                    return httpx.Response(200, json={"task_id": "another", "status": "canceled"})
                code = "other_conflict" if outcome == "bad409" else "task_already_terminal"
                return httpx.Response(409 if outcome == "bad409" else int(outcome), json={"error": {"code": code}})
            if request.method == "POST":
                return httpx.Response(202, json=task_snapshot("pending"))
            entered.set()
            await asyncio.Future()
            raise AssertionError("unreachable")

        bind_transport(monkeypatch, handle)
        operation = asyncio.create_task(MinerUApiParser(api_url="https://api.test").parse_async(source, cancel_on_cancel=True))
        await asyncio.wait_for(entered.wait(), 3)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, 1)
        assert cancellations == 1

    with caplog.at_level(logging.WARNING, logger="mineru.api_client"):
        asyncio.run(scenario())
    assert ("could not be confirmed" in caplog.text) == (outcome not in {"404", "409"})


@pytest.mark.parametrize("failure", ["timeout", "transport", "invalid_response"])
def test_cancel_during_failed_creation_has_no_orphan_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure: str
) -> None:
    """创建失败或截止后无法取得身份时记录未确认，不重建任务且回收受保护的提交。"""
    source = tmp_path / "file.pdf"
    source.write_bytes(b"pdf")
    monkeypatch.setattr(api_client, "_TASK_CREATE_TIMEOUT_SECONDS", 0.05)

    async def scenario() -> None:
        """让调用方先取消，再触发原创建请求的失败。"""
        entered, release = asyncio.Event(), asyncio.Event()
        requests = []

        async def handle(request: httpx.Request) -> httpx.Response:
            """阻止重试，并观察超时确实结束原来的传输。"""
            requests.append(request)
            entered.set()
            await release.wait()
            if failure == "transport":
                raise httpx.ReadError("lost receipt", request=request)
            return httpx.Response(202, json={})

        bind_transport(monkeypatch, handle)
        operation = asyncio.create_task(MinerUApiParser(api_url="https://api.test").parse_async(source, cancel_on_cancel=True))
        await asyncio.wait_for(entered.wait(), 3)
        operation.cancel()
        if failure != "timeout":
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, 1)
        assert len(requests) == 1
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]

    asyncio.run(scenario())
    assert "could not be confirmed for unknown task" in caplog.text


def test_concurrent_calls_on_same_parser_keep_separate_task_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """同一解析器的两个并发调用只能取消各自服务端任务，不能复用实例上的旧身份。"""
    sources = [tmp_path / name for name in ("a.pdf", "b.pdf")]
    for source in sources:
        source.write_bytes(b"pdf")

    async def scenario() -> None:
        """分别阻塞两个已提交任务，先取消一个并确认另一个仍可独立等待。"""
        waiting: asyncio.Queue[str] = asyncio.Queue()
        canceled = []

        async def handle(request: httpx.Request) -> httpx.Response:
            """用文件名生成不同公共任务 ID，避免用调度顺序推测调用身份。"""
            if request.url.path.endswith("/cancel"):
                task_id = request.url.path.split("/")[-2]
                canceled.append(task_id)
                return httpx.Response(200, json={"task_id": task_id, "status": "canceled"})
            if request.method == "POST":
                import json

                file_name = json.loads(request.content)["files"][0]["file_name"]
                snapshot = task_snapshot("pending", file_name=file_name)
                snapshot["task_id"] = "task-" + Path(file_name).stem
                return httpx.Response(202, json=snapshot)
            waiting.put_nowait(request.url.path.split("/")[-1])
            await asyncio.Future()
            raise AssertionError("unreachable")

        bind_transport(monkeypatch, handle)
        parser = MinerUApiParser(api_url="https://api.test")
        operations = [asyncio.create_task(parser.parse_async(source, cancel_on_cancel=True)) for source in sources]
        try:
            assert {await asyncio.wait_for(waiting.get(), 3) for _ in operations} == {"task-a", "task-b"}
            operations[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(operations[0], 3)
            assert canceled == ["task-a"]
            assert not operations[1].done()
            operations[1].cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(operations[1], 3)
            assert canceled == ["task-a", "task-b"]
        finally:
            for operation in operations:
                operation.cancel()
            await asyncio.gather(*operations, return_exceptions=True)

    asyncio.run(scenario())
