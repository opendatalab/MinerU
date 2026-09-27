"""Gradio 步骤卡片与单次任务的真实阶段计时。"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from .i18n import localized_message, localized_text as _localized_text

DEFAULT_STATUS = "Upload a file and start conversion."
STATUS_PREPARING_REQUEST = "Preparing request..."
STATUS_CHECKING_SERVER = "Checking server status..."
STATUS_SUBMITTING_TASK = "Submitting task..."
STATUS_QUEUED_LOCALLY = "Queued locally"
STATUS_QUEUED_ON_SERVER = "Queued on server"
STATUS_PROCESSING_ON_SERVER = "Processing on server..."
STATUS_DOWNLOADING_RESULT = "Task completed, downloading result..."
STATUS_PROCESSING_OUTPUT = "Preparing outputs..."
STATUS_COMPLETED = "Completed"

_STEPS = ("prepare", "check", "submit", "queue", "process", "download", "outputs", "done")
_MESSAGE_STEPS = {
    STATUS_PREPARING_REQUEST: 0,
    STATUS_CHECKING_SERVER: 1,
    STATUS_SUBMITTING_TASK: 2,
    STATUS_QUEUED_LOCALLY: 3,
    STATUS_QUEUED_ON_SERVER: 3,
    STATUS_PROCESSING_ON_SERVER: 4,
    STATUS_DOWNLOADING_RESULT: 5,
    STATUS_PROCESSING_OUTPUT: 6,
    STATUS_COMPLETED: 7,
}


@dataclass(frozen=True)
class ParseStatusUpdate:
    """在现有阶段通知中携带可选的服务端文件处理耗时，单位为毫秒。"""

    message: str
    duration_ms: float | None = None


@dataclass
class StatusPanelState:
    """保存一次转换的阶段和单调时钟，不在会话之间共享状态。"""

    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    message: str = DEFAULT_STATUS
    step_index: int = -1
    processing_elapsed: float | None = None
    server_elapsed: float | None = None
    _processing_started: float | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    phase_id: int = 0
    sequence: int = 0

    def append(self, message: str | ParseStatusUpdate, *, at: float | None = None) -> bool:
        """阶段通知维持本地计时；服务端耗时只更新完成数值，不重启同阶段动画。"""
        duration_changed = False
        if isinstance(message, ParseStatusUpdate):
            if message.duration_ms is not None:
                elapsed = message.duration_ms / 1000
                duration_changed = elapsed != self.server_elapsed
                self.server_elapsed = elapsed
            message = message.message
        if not message or message == self.message:
            if duration_changed:
                self.sequence += 1
            return duration_changed
        now = self.clock() if at is None else at
        if self._processing_started is not None:
            self.processing_elapsed = max(0.0, now - self._processing_started)
            self._processing_started = None
        if message == STATUS_PROCESSING_ON_SERVER:
            self._processing_started = now
            self.processing_elapsed = 0.0
        self.message = message
        self.phase_id += 1
        self.sequence += 1
        # 本地排队结束后可以重新准备请求，高亮必须与实际阶段一致。
        self.step_index = _MESSAGE_STEPS.get(message, self.step_index)
        if message.startswith("Failed:"):
            self.step_index = len(_STEPS) - 1
        return True

    def render(self) -> str:
        """按 3.4.5 的两列卡片结构渲染当前状态，并转义外部错误文本。"""
        now = self.clock()
        failed = self.message.startswith("Failed:")
        completed = self.message == STATUS_COMPLETED
        items: list[str] = []
        for index, key in enumerate(_STEPS):
            if failed and index == self.step_index:
                state = "is-active is-error"
                key = "failed"
            elif completed or index < self.step_index:
                state = "is-done"
            elif index == self.step_index:
                state = "is-active"
            else:
                state = "is-pending"
            items.append(
                f'<div class="status-step {state}"><span class="status-dot"></span>'
                f'<span class="status-label">{_localized_text("status_step_" + key)}</span></div>'
            )
        latest = self.message
        timer_attributes = ""
        if self._processing_started is not None:
            elapsed = max(0.0, now - self._processing_started)
            latest = f"Processing on server ({elapsed:.2f}s)"
            timer_attributes = (
                f' data-mineru-processing-start="{self._processing_started:.9f}" data-mineru-processing-elapsed="{elapsed:.6f}"'
            )
        elif self.message in (STATUS_QUEUED_LOCALLY, STATUS_QUEUED_ON_SERVER):
            queue_key = "queued_locally" if self.message == STATUS_QUEUED_LOCALLY else "queued_on_server"
            timer_attributes = f' data-mineru-queue-key="{queue_key}"'
        elif completed:
            elapsed = self.server_elapsed if self.server_elapsed is not None else self.processing_elapsed
            if elapsed is not None:
                # API 优先使用含打包的文件耗时；HF 等直接调用方继续使用阶段间隔，极短任务显示下限为 0.01 秒。
                display_elapsed = max(Decimal("0.01"), Decimal(str(elapsed)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
                latest = f"{STATUS_COMPLETED} ({display_elapsed:.2f}s)"
        if self.message == DEFAULT_STATUS:
            title = _localized_text("status_idle_title")
            latest_html = _localized_text("status_idle_hint")
        else:
            title = _localized_text("status_latest")
            latest_html = localized_message(latest)
        return (
            '<div class="status-steps-panel">'
            f'<div class="status-panel-title">{title}</div>'
            f'<div class="status-steps-list">{"".join(items)}</div>'
            f'<div class="status-latest" data-mineru-run-id="{self.run_id}" '
            f'data-mineru-phase-id="{self.phase_id}" data-mineru-status-seq="{self.sequence}"'
            f"{timer_attributes}>{latest_html}</div></div>"
        )


def status_html(message: str = DEFAULT_STATUS) -> str:
    """为初始化、输入校验和重置生成没有历史计时的状态卡片。"""
    state = StatusPanelState()
    state.append(message)
    return state.render()


__all__ = [
    "DEFAULT_STATUS",
    "STATUS_CHECKING_SERVER",
    "STATUS_COMPLETED",
    "STATUS_DOWNLOADING_RESULT",
    "STATUS_PREPARING_REQUEST",
    "STATUS_PROCESSING_ON_SERVER",
    "STATUS_PROCESSING_OUTPUT",
    "STATUS_QUEUED_LOCALLY",
    "STATUS_QUEUED_ON_SERVER",
    "STATUS_SUBMITTING_TASK",
    "ParseStatusUpdate",
    "StatusPanelState",
    "status_html",
]
