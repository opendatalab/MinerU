"""步骤卡片的计时、状态生命周期与安全文本回归。"""

import asyncio
from contextlib import aclosing

import pytest

from mineru.kit.gradio.status import (
    STATUS_COMPLETED,
    STATUS_DOWNLOADING_RESULT,
    STATUS_PROCESSING_ON_SERVER,
    STATUS_PROCESSING_OUTPUT,
    STATUS_QUEUED_LOCALLY,
    STATUS_QUEUED_ON_SERVER,
    StatusPanelState,
    status_html,
    stream_status_updates,
)


def test_processing_clock_excludes_queue_and_download_and_keeps_completed_duration() -> None:
    """模拟长时间排队和下载，确保完成耗时仅保留两次真实运行通知之间的时间。"""
    now = [0.0]
    state = StatusPanelState(clock=lambda: now[0])
    state.append(STATUS_QUEUED_ON_SERVER)
    now[0] = 100.0
    state.append(STATUS_PROCESSING_ON_SERVER)
    now[0] = 101.2
    assert "Processing on server (1.20s)" in state.render()
    assert 'data-mineru-processing-start="100.000000000"' in state.render()
    assert 'data-mineru-processing-elapsed="1.200000"' in state.render()
    assert not state.append(STATUS_PROCESSING_ON_SERVER)
    now[0] = 104.6
    state.append(STATUS_DOWNLOADING_RESULT, at=104.5)
    assert state.processing_elapsed == pytest.approx(4.5)
    assert "status-step is-active" in state.render()
    assert state.render().count("status-step is-done") == 5
    now[0] = 200.0
    state.append(STATUS_PROCESSING_OUTPUT)
    state.append(STATUS_COMPLETED)
    assert "Completed (4.50s)" in state.render()
    assert "data-mineru-processing-start" not in state.render()
    assert state.render().count("status-step is-done") == 8


@pytest.mark.parametrize("message", [STATUS_QUEUED_LOCALLY, STATUS_QUEUED_ON_SERVER])
def test_queued_status_stays_stable_without_timer_updates(message: str) -> None:
    """验证服务端排队 HTML 保持静态，并携带供浏览器绘制动画的状态标记。"""
    now = [5.0]
    state = StatusPanelState(clock=lambda: now[0])
    state.append(message)
    initial = state.render()
    queue_key = "queued_locally" if message == STATUS_QUEUED_LOCALLY else "queued_on_server"
    assert f'data-mineru-queue-key="{queue_key}"' in initial
    for seconds in (0, 3, 9, 10):
        now[0] = 5.0 + seconds
        assert not state.append(message)
        assert state.render() == initial
        assert f'data-mineru-i18n-en="{message}"' in state.render()


def test_fast_completion_shows_zero_elapsed_time_and_reset_has_eight_pending_steps() -> None:
    """验证未观察到解析阶段时完成状态显示 0.00 秒，重置后仍保留完整双语步骤。"""
    state = StatusPanelState()
    state.append(STATUS_QUEUED_ON_SERVER)
    state.append(STATUS_DOWNLOADING_RESULT)
    state.append(STATUS_COMPLETED)
    assert 'data-mineru-i18n-en="Completed (0.00s)"' in state.render()
    assert 'data-mineru-i18n-zh="已完成（0.00 秒）"' in state.render()
    idle = status_html()
    assert idle.count("status-step is-pending") == 8
    assert 'data-mineru-i18n-en="Waiting"' in idle
    assert 'data-mineru-i18n-zh="排队"' in idle
    assert "status-steps-panel" in idle


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [(0.0, "0.00"), (0.004, "0.00"), (0.005, "0.01"), (0.04, "0.04"), (0.05, "0.05"), (0.125, "0.13")],
)
def test_completed_duration_uses_decimal_half_up_rounding(elapsed: float, expected: str) -> None:
    """验证完成耗时保留两位小数，并在 0.005 秒等边界按四舍五入显示。"""
    state = StatusPanelState(clock=lambda: 0.0)
    state.append(STATUS_PROCESSING_ON_SERVER, at=0.0)
    state.append(STATUS_DOWNLOADING_RESULT, at=elapsed)
    state.append(STATUS_COMPLETED)
    assert state.processing_elapsed == pytest.approx(elapsed)
    assert f'data-mineru-i18n-en="Completed ({expected}s)"' in state.render()


@pytest.mark.parametrize("error", ["server task failed", "server task canceled", '<script>alert("x")</script>'])
def test_failure_stops_timer_and_renders_escaped_error(error: str) -> None:
    """验证失败或取消停止计时，并使用红色失败节点展示转义后的消息。"""
    state = StatusPanelState(clock=lambda: 10.0)
    state.append(STATUS_PROCESSING_ON_SERVER, at=1.0)
    state.append(f"Failed: {error}", at=3.0)
    rendered = state.render()
    assert state.processing_elapsed == 2.0
    assert "status-step is-active is-error" in rendered
    assert 'data-mineru-i18n-en="Failed"' in rendered
    assert "<script>" not in rendered


def test_status_stream_waits_for_stage_changes_and_cleans_up_waiter() -> None:
    """验证排队和解析期间均无服务端计时帧，真实阶段仍及时更新。"""

    async def scenario() -> None:
        """运行可手动推进时钟的解析等待场景。"""
        now = [1.0]
        state = StatusPanelState(clock=lambda: now[0])
        events: asyncio.Queue[tuple[str, float]] = asyncio.Queue()
        events.put_nowait((STATUS_QUEUED_ON_SERVER, 1.0))
        task = asyncio.create_task(asyncio.Event().wait())
        baseline = set(asyncio.all_tasks())
        async with aclosing(stream_status_updates(task, events, state)) as stream:
            assert "Queued on server" in await anext(stream)
            now[0] = 2.3
            next_update = asyncio.create_task(anext(stream))
            await asyncio.sleep(0.03)
            assert not next_update.done()
            events.put_nowait((STATUS_PROCESSING_ON_SERVER, 2.3))
            assert "(0.00s)" in await asyncio.wait_for(next_update, timeout=1)
            next_update = asyncio.create_task(anext(stream))
            await asyncio.sleep(0.03)
            assert not next_update.done()
            events.put_nowait((STATUS_DOWNLOADING_RESULT, 2.6))
            assert "Task completed, downloading result" in await asyncio.wait_for(next_update, timeout=1)
            assert set(asyncio.all_tasks()) == baseline
        assert set(asyncio.all_tasks()) == baseline
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("message", ["Preparing request...", STATUS_QUEUED_ON_SERVER, STATUS_PROCESSING_ON_SERVER])
def test_status_heartbeat_survives_duplicate_notifications(message: str) -> None:
    """重复阶段通知不能饿死每秒快照，心跳也不能改变任务或阶段身份。"""

    async def scenario() -> None:
        """持续发送同阶段通知并检查周期快照及即时阶段切换。"""
        state = StatusPanelState()
        state.append(message)
        identity = (state.run_id, state.phase_id, state._processing_started)
        events: asyncio.Queue[tuple[str, float]] = asyncio.Queue()
        finish = asyncio.Event()
        task = asyncio.create_task(finish.wait())

        async def repeat() -> None:
            """模拟每次查询都返回相同阶段。"""
            while not finish.is_set():
                events.put_nowait((message, state.clock()))
                await asyncio.sleep(0.05)

        sender = asyncio.create_task(repeat())
        try:
            async with aclosing(stream_status_updates(task, events, state)) as stream:
                started = asyncio.get_running_loop().time()
                first = await asyncio.wait_for(anext(stream), 2)
                assert 0.9 <= asyncio.get_running_loop().time() - started < 2
                assert 'data-mineru-status-seq="1"' in first
                assert identity == (state.run_id, state.phase_id, state._processing_started)
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
                events.put_nowait((STATUS_PROCESSING_OUTPUT, state.clock()))
                update = await asyncio.wait_for(anext(stream), 0.5)
                assert "Preparing outputs" in update
                assert 'data-mineru-status-seq="2"' in update
                finish.set()
                assert [item async for item in stream] == []
        finally:
            finish.set()
            sender.cancel()
            await asyncio.gather(task, sender, return_exceptions=True)

    asyncio.run(scenario())


def test_local_queue_can_return_to_preparation() -> None:
    """拿到本地槽位后高亮准备步骤，避免底部阶段与高亮不一致。"""
    state = StatusPanelState()
    state.append(STATUS_QUEUED_LOCALLY)
    phase = state.phase_id
    state.append("Preparing request...")
    assert state.step_index == 0
    assert state.phase_id == phase + 1
    assert "data-mineru-queue-key" not in state.render()
