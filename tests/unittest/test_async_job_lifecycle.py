"""验证解析 API 任务取消、排队和关闭的真实生命周期。"""

import asyncio
from pathlib import Path

import pytest

from mineru.parser.api_server import ApiServerError, JobStore, _JobRecord


def test_queued_job_cancel_never_starts_and_does_not_cancel_other_job() -> None:
    """排队任务取消不进入解析，另一份文档继续完成。"""

    async def run() -> None:
        """在一个并发名额下验证队列取消和状态不可被覆盖。"""
        store = JobStore(concurrency=1)
        first, queued = _JobRecord("first"), _JobRecord("queued")
        store._jobs = {first.id: first, queued.id: queued}
        entered, release = asyncio.Event(), asyncio.Event()
        queued_started = False

        async def work() -> None:
            """占住唯一名额，直到测试释放。"""
            first.status = "running"
            entered.set()
            await release.wait()
            first.status = "completed"

        async def forbidden() -> None:
            """记录不应开始的排队任务。"""
            nonlocal queued_started
            queued_started = True

        store.start_task(first, work)
        await entered.wait()
        store.start_task(queued, forbidden)
        store.cancel(queued.id)
        with pytest.raises(ApiServerError):
            store.cancel(queued.id)
        release.set()
        await asyncio.gather(*list(store._tasks.values()), return_exceptions=True)
        await asyncio.sleep(0)
        assert first.status == "completed"
        assert queued.status == "canceled"
        assert not queued_started
        assert not store._tasks
        await store.shutdown()

    asyncio.run(run())


def test_running_cancel_keeps_slot_until_cleanup_and_shutdown_waits() -> None:
    """运行中取消必须等清理退出，不能让下一份文档提前取得名额。"""

    async def run() -> None:
        """用受控清理阶段验证取消、排队和服务关闭顺序。"""
        store = JobStore(1)
        first, second = _JobRecord("first"), _JobRecord("second")
        store._jobs = {first.id: first, second.id: second}
        entered, cleaning, finish_cleanup, second_started = (asyncio.Event() for _ in range(4))

        async def work() -> None:
            """模拟取消之后仍需完成的引擎资源清理。"""
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish_cleanup.wait()

        async def followup() -> None:
            """仅在前一任务清理完成后才允许开始。"""
            second_started.set()

        store.start_task(first, work)
        await entered.wait()
        store.start_task(second, followup)
        store.cancel(first.id)
        await cleaning.wait()
        assert not second_started.is_set()
        shutdown = asyncio.create_task(store.shutdown())
        await asyncio.sleep(0)
        assert not shutdown.done()
        assert not second_started.is_set()
        finish_cleanup.set()
        await shutdown
        assert not second_started.is_set()
        assert first.status == second.status == "canceled"
        assert store.runtime_owner.closed

    asyncio.run(run())


def test_parser_input_cancellation_waits_before_tempfile_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """取消输入准备后不能让 API 删除仍被后台读取的临时文件。"""
    import threading
    from mineru.parser.mineru_parser import MinerUParser

    path = tmp_path / "input.pdf"
    path.write_bytes(b"pdf")
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def prepare(*args: object) -> None:
        """模拟持有临时文件路径的同步预处理。"""
        entered.set()
        assert release.wait(3)
        assert path.read_bytes() == b"pdf"
        finished.set()

    parser = MinerUParser(tier="flash")
    monkeypatch.setattr(parser, "_prepare_input", prepare)

    async def run() -> None:
        """确认取消返回之前同步预处理已经退出。"""
        task = asyncio.create_task(parser.parse_async(path))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    try:
        asyncio.run(run())
    finally:
        release.set()


def test_late_parser_result_is_not_published_after_cancel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """即使下层吞掉取消并返回结果，API 也不得发布已取消任务的产物。"""
    from mineru.parser import api_server

    source = tmp_path / "late.pdf"
    source.write_bytes(b"pdf")
    rendered = []

    class Result:
        """模拟取消之后仍迟到返回的有效解析结果。"""

        pages = []

        def to_dict(self, *, skip_defaults: bool = True) -> dict:
            """若 API 错误地继续导出则留下可观测记录。"""
            rendered.append(True)
            return {"pages": []}

    async def run() -> None:
        """取消实际 JobStore 任务，并通过真实文件处理逻辑检查结果发布边界。"""
        entered = asyncio.Event()

        async def parse(*args: object, **kwargs: object) -> Result:
            """模拟一个不正确地吞掉 CancelledError 的第三方解析器。"""
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return Result()

        monkeypatch.setattr(api_server, "parse_async", parse)
        store = JobStore()
        files = api_server.FileStore(tmp_path / "storage")
        request = api_server.CreateJobRequest.model_validate(
            {
                "tier": "standard",
                "output_formats": ["middle_json"],
                "files": [{"source": {"type": "local", "path": str(source)}}],
            }
        )
        rec = store.create(request, files)

        async def work() -> None:
            """执行真实任务循环，避免只测试状态赋值。"""
            await api_server._run_job(rec, request, files, image_analysis=True, allow_local_source=True)

        store.start_task(rec, work)
        await entered.wait()
        task = store._tasks[rec.id]
        store.cancel(rec.id)
        await task
        assert rec.status == "canceled"
        assert rec.files[0].output_files is None
        assert rec.progress.completed == 0
        assert not rendered
        await store.shutdown()

    asyncio.run(run())
