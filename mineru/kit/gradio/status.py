"""Gradio 步骤卡片、单次任务计时与流式状态等待。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

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


@dataclass
class StatusPanelState:
    """保存一次转换的阶段和单调时钟，不在会话之间共享状态。"""

    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    message: str = DEFAULT_STATUS
    step_index: int = -1
    processing_elapsed: float | None = None
    _processing_started: float | None = None

    def append(self, message: str, *, at: float | None = None) -> bool:
        """接收真实阶段变化；重复通知不会重置解析计时。"""
        if not message or message == self.message:
            return False
        now = self.clock() if at is None else at
        if self._processing_started is not None:
            self.processing_elapsed = max(0.0, now - self._processing_started)
            self._processing_started = None
        if message == STATUS_PROCESSING_ON_SERVER:
            self._processing_started = now
            self.processing_elapsed = 0.0
        self.message = message
        # 本地等待可能发生在上传之前；已展示的步骤不因后续准备通知倒退。
        self.step_index = max(self.step_index, _MESSAGE_STEPS.get(message, -1))
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
            display_elapsed = Decimal(str(self.processing_elapsed or 0.0)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
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
            f'<div class="status-latest"{timer_attributes}>{latest_html}</div></div>'
        )


def status_html(message: str = DEFAULT_STATUS) -> str:
    """为初始化、输入校验和重置生成没有历史计时的状态卡片。"""
    state = StatusPanelState()
    state.append(message)
    return state.render()


async def stream_status_updates(
    task: asyncio.Task[Any],
    events: asyncio.Queue[tuple[str, float]],
    state: StatusPanelState,
) -> AsyncIterator[str]:
    """只等待真实阶段通知或任务结束，并及时回收临时等待任务。"""
    while True:
        while not events.empty():
            message, at = events.get_nowait()
            if state.append(message, at=at):
                yield state.render()
        if task.done():
            return
        waiter = asyncio.create_task(events.get())
        updated_html: str | None = None
        try:
            done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
            if waiter in done:
                message, at = waiter.result()
                if state.append(message, at=at):
                    updated_html = state.render()
        finally:
            if not waiter.done():
                waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        # 向 Gradio 交回控制权前已回收临时 waiter，避免暂停在 yield 时残留后台等待。
        if updated_html is not None:
            yield updated_html


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
    "StatusPanelState",
    "status_html",
    "stream_status_updates",
]
