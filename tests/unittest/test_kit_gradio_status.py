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
    assert state.refresh_interval == 0.1
    now[0] = 101.2
    assert "Processing on server (1.2s)" in state.render()
    assert not state.append(STATUS_PROCESSING_ON_SERVER)
    now[0] = 104.6
    state.append(STATUS_DOWNLOADING_RESULT, at=104.5)
    assert state.refresh_interval is None
    assert state.processing_elapsed == pytest.approx(4.5)
    assert "status-step is-active" in state.render()
    assert state.render().count("status-step is-done") == 5
    now[0] = 200.0
    state.append(STATUS_PROCESSING_OUTPUT)
    state.append(STATUS_COMPLETED)
    assert "Completed (4.5s)" in state.render()
    assert state.render().count("status-step is-done") == 8


@pytest.mark.parametrize("message", [STATUS_QUEUED_LOCALLY, STATUS_QUEUED_ON_SERVER])
def test_queue_animation_cycles_without_restarting_on_repeated_status(message: str) -> None:
    """验证本地和远端队列都按每秒一个圆点循环，重复轮询不重置动画。"""
    now = [5.0]
    state = StatusPanelState(clock=lambda: now[0])
    state.append(message)
    assert state.refresh_interval == 1.0
    for seconds, dots in [(0, 1), (3, 4), (9, 10), (10, 1)]:
        now[0] = 5.0 + seconds
        assert not state.append(message)
        assert f'<div class="status-latest">{message}{"." * dots}</div>' in state.render()


def test_fast_completion_has_no_invented_elapsed_time_and_reset_has_eight_pending_steps() -> None:
    """验证没有观察到运行的任务不显示零秒耗时，重置后保留完整双语步骤。"""
    state = StatusPanelState()
    state.append(STATUS_QUEUED_ON_SERVER)
    state.append(STATUS_DOWNLOADING_RESULT)
    state.append(STATUS_COMPLETED)
    assert '<div class="status-latest">Completed</div>' in state.render()
    idle = status_html()
    assert idle.count("status-step is-pending") == 8
    assert 'data-mineru-i18n-en="Waiting"' in idle
    assert 'data-mineru-i18n-zh="排队"' in idle
    assert "status-steps-panel" in idle


@pytest.mark.parametrize("error", ["server task failed", "server task canceled", '<script>alert("x")</script>'])
def test_failure_stops_timer_and_renders_escaped_error(error: str) -> None:
    """验证失败或取消停止计时，并使用红色失败节点展示转义后的消息。"""
    state = StatusPanelState(clock=lambda: 10.0)
    state.append(STATUS_PROCESSING_ON_SERVER, at=1.0)
    state.append(f"Failed: {error}", at=3.0)
    rendered = state.render()
    assert state.processing_elapsed == 2.0
    assert state.refresh_interval is None
    assert "status-step is-active is-error" in rendered
    assert 'data-mineru-i18n-en="Failed"' in rendered
    assert "<script>" not in rendered


def test_status_stream_ticks_while_job_waits_and_cleans_up_waiter() -> None:
    """验证无网络通知时仍刷新本地时钟，关闭流后不残留队列等待任务。"""

    async def scenario() -> None:
        """运行可手动推进时钟的解析等待场景。"""
        now = [1.0]
        state = StatusPanelState(clock=lambda: now[0])
        events: asyncio.Queue[tuple[str, float]] = asyncio.Queue()
        events.put_nowait((STATUS_PROCESSING_ON_SERVER, 1.0))
        task = asyncio.create_task(asyncio.Event().wait())
        baseline = set(asyncio.all_tasks())
        async with aclosing(stream_status_updates(task, events, state)) as stream:
            assert "(0.0s)" in await anext(stream)
            now[0] = 2.3
            assert "(1.3s)" in await asyncio.wait_for(anext(stream), timeout=1)
            assert set(asyncio.all_tasks()) == baseline
        assert set(asyncio.all_tasks()) == baseline
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
