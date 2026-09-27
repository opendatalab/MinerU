"""步骤卡片的计时、状态生命周期与安全文本回归。"""

import json

import pytest

from mineru.kit.gradio.conversion import ConversionRun, SessionConversions

from mineru.kit.gradio.status import (
    STATUS_COMPLETED,
    STATUS_DOWNLOADING_RESULT,
    STATUS_PROCESSING_ON_SERVER,
    STATUS_PROCESSING_OUTPUT,
    STATUS_QUEUED_LOCALLY,
    STATUS_QUEUED_ON_SERVER,
    ParseStatusUpdate,
    StatusPanelState,
    status_html,
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


@pytest.mark.parametrize("queued", [False, True])
def test_fast_completion_uses_server_duration_without_running(queued: bool) -> None:
    """直接完成或轮询漏过 running 时仍显示服务端亚秒耗时，重置后保留完整双语步骤。"""
    state = StatusPanelState()
    if queued:
        state.append(STATUS_QUEUED_ON_SERVER)
    state.append(ParseStatusUpdate(STATUS_DOWNLOADING_RESULT, 250))
    state.append(STATUS_COMPLETED)
    assert state.processing_elapsed is None
    assert 'data-mineru-i18n-en="Completed (0.25s)"' in state.render()
    assert 'data-mineru-i18n-zh="已完成（0.25 秒）"' in state.render()
    assert "data-mineru-processing-start" not in state.render()
    assert state.render().count("status-step is-done") == 8
    idle = status_html()
    assert idle.count("status-step is-pending") == 8
    assert 'data-mineru-i18n-en="Waiting"' in idle
    assert 'data-mineru-i18n-zh="排队"' in idle
    assert "status-steps-panel" in idle


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        (0.0, "0.01"),
        (0.004, "0.01"),
        (0.005, "0.01"),
        (0.04, "0.04"),
        (0.05, "0.05"),
        (0.125, "0.13"),
        (0.25, "0.25"),
        (0.999, "1.00"),
        (1.25, "1.25"),
    ],
)
def test_completed_duration_uses_decimal_half_up_rounding(elapsed: float, expected: str) -> None:
    """直接函数调用的阶段计时保留两位小数，极短任务按 0.01 秒显示下限处理。"""
    state = StatusPanelState(clock=lambda: 0.0)
    state.append(STATUS_PROCESSING_ON_SERVER, at=0.0)
    state.append(STATUS_DOWNLOADING_RESULT, at=elapsed)
    state.append(STATUS_COMPLETED)
    assert state.processing_elapsed == pytest.approx(elapsed)
    assert f'data-mineru-i18n-en="Completed ({expected}s)"' in state.render()


@pytest.mark.parametrize(("duration_ms", "expected"), [(0, "0.01"), (1, "0.01"), (125, "0.13"), (1750, "1.75")])
def test_server_duration_overrides_observed_polling_time(duration_ms: float, expected: str) -> None:
    """最终数值来自服务端，长轮询间隔和下载等待不会覆盖它。"""
    state = StatusPanelState(clock=lambda: 1000.0)
    state.append(STATUS_PROCESSING_ON_SERVER, at=10.0)
    state.append(ParseStatusUpdate(STATUS_DOWNLOADING_RESULT, duration_ms), at=20.0)
    state.append(STATUS_PROCESSING_OUTPUT, at=900.0)
    state.append(STATUS_COMPLETED, at=1000.0)
    assert state.processing_elapsed == 10.0
    assert f"Completed ({expected}s)" in state.render()


def test_duration_only_snapshot_preserves_phase_and_rejects_late_updates() -> None:
    """仅补齐耗时也增加快照序号；重复、终态和取消后的通知不能覆盖当前结果。"""
    run = ConversionRun("timed")
    run.publish(STATUS_DOWNLOADING_RESULT)
    phase, sequence = run.state.phase_id, json.loads(run.snapshot)["sequence"]
    run.publish(ParseStatusUpdate(STATUS_DOWNLOADING_RESULT, 250))
    assert run.state.phase_id == phase
    assert json.loads(run.snapshot)["sequence"] == sequence + 1
    snapshot = run.snapshot
    run.publish(ParseStatusUpdate(STATUS_DOWNLOADING_RESULT, 250))
    assert run.snapshot == snapshot
    run.publish(STATUS_COMPLETED)
    final = run.snapshot
    run.publish(ParseStatusUpdate(STATUS_DOWNLOADING_RESULT, 9000))
    assert run.snapshot == final
    assert "Completed (0.25s)" in final
    canceled = ConversionRun("canceled")
    canceled.cancel()
    canceled.publish(ParseStatusUpdate(STATUS_DOWNLOADING_RESULT, 250))
    assert canceled.state.server_elapsed is None


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


def test_complete_snapshots_stay_stable_and_reject_late_notifications() -> None:
    """重复轮询不改变内容，失败或完成后拒绝迟到阶段。"""
    run = ConversionRun("run-1")
    run.publish(STATUS_PROCESSING_ON_SERVER, at=1.0)
    snapshot = run.snapshot
    for _ in range(150):
        run.publish(STATUS_PROCESSING_ON_SERVER, at=2.0)
        assert run.snapshot == snapshot
    run.publish(STATUS_DOWNLOADING_RESULT, at=3.0)
    run.publish(STATUS_COMPLETED, at=4.0)
    final = json.loads(run.snapshot)
    assert final["run_id"] == "run-1" and final["terminal"] is True
    assert "Completed (2.00s)" in final["html"]
    assert final["sequence"] > json.loads(snapshot)["sequence"]
    run.publish(STATUS_PROCESSING_OUTPUT, at=5.0)
    assert json.loads(run.snapshot) == final


def test_session_run_identity_and_revision_reject_stale_submissions() -> None:
    """同会话旧任务失效，其他会话保留自己的状态，旧修订无法重新启动。"""
    sessions = SessionConversions()
    old = sessions.start("a", "old", revision=1)
    other = sessions.start("b", "other", revision=1)
    current = sessions.start("a", "new", revision=2)
    assert old.cancelled and sessions.current("a", "old") is None
    assert sessions.current("a", "new") is current
    assert sessions.current("b", "other") is other
    assert sessions.start("a", "old", revision=1) is None
    sessions.cancel("a", "old")
    assert sessions.current("a", "new") is current
    sessions.cancel("a", "new")
    assert sessions.current("a", "new") is None
    assert sessions.current("b", "other") is other


def test_local_queue_can_return_to_preparation() -> None:
    """拿到本地槽位后高亮准备步骤，避免底部阶段与高亮不一致。"""
    state = StatusPanelState()
    state.append(STATUS_QUEUED_LOCALLY)
    phase = state.phase_id
    state.append("Preparing request...")
    assert state.step_index == 0
    assert state.phase_id == phase + 1
    assert "data-mineru-queue-key" not in state.render()
