"""Gradio 步骤卡片、单次任务计时与流式状态等待。"""

from __future__ import annotations

import asyncio
import html
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

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

_STEPS = (
    ("prepare", "Prepare", "准备请求"),
    ("check", "Check service", "检查服务"),
    ("submit", "Submit", "提交任务"),
    ("queue", "Queue", "排队"),
    ("process", "Parse", "解析中"),
    ("download", "Download", "下载结果"),
    ("outputs", "Build outputs", "整理输出"),
    ("done", "Done", "完成"),
)
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


def _localized_text(key: str, english: str, chinese: str) -> str:
    """输出中英文属性，复用页面现有的浏览器语言本地化逻辑。"""
    return (
        f'<span data-mineru-i18n-key="{html.escape(key, quote=True)}"'
        f' data-mineru-i18n-en="{html.escape(english, quote=True)}"'
        f' data-mineru-i18n-zh="{html.escape(chinese, quote=True)}">{html.escape(chinese)}</span>'
    )


@dataclass
class StatusPanelState:
    """保存一次转换的阶段和单调时钟，不在会话之间共享状态。"""

    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    message: str = DEFAULT_STATUS
    step_index: int = -1
    processing_elapsed: float | None = None
    _processing_started: float | None = None
    _queue_started: float | None = None

    def append(self, message: str, *, at: float | None = None) -> bool:
        """接收真实阶段变化；重复通知不会重置解析计时或排队动画。"""
        if not message or message == self.message:
            return False
        now = self.clock() if at is None else at
        if self._processing_started is not None:
            self.processing_elapsed = max(0.0, now - self._processing_started)
            self._processing_started = None
        if message == STATUS_PROCESSING_ON_SERVER:
            self._processing_started = now
            self.processing_elapsed = 0.0
        if message in (STATUS_QUEUED_LOCALLY, STATUS_QUEUED_ON_SERVER):
            if self._queue_started is None:
                self._queue_started = now
        else:
            self._queue_started = None
        self.message = message
        # 本地等待可能发生在上传之前；已展示的步骤不因后续准备通知倒退。
        self.step_index = max(self.step_index, _MESSAGE_STEPS.get(message, -1))
        if message.startswith("Failed:"):
            self.step_index = len(_STEPS) - 1
        return True

    @property
    def refresh_interval(self) -> float | None:
        """只在解析或排队期间启用旧版刷新频率。"""
        if self._processing_started is not None:
            return 0.1
        if self._queue_started is not None:
            return 1.0
        return None

    def render(self) -> str:
        """按 3.4.5 的两列卡片结构渲染当前状态，并转义外部错误文本。"""
        now = self.clock()
        failed = self.message.startswith("Failed:")
        completed = self.message == STATUS_COMPLETED
        items: list[str] = []
        for index, (key, english, chinese) in enumerate(_STEPS):
            if failed and index == self.step_index:
                state = "is-active is-error"
                key, english, chinese = "failed", "Failed", "失败"
            elif completed or index < self.step_index:
                state = "is-done"
            elif index == self.step_index:
                state = "is-active"
            else:
                state = "is-pending"
            items.append(
                f'<div class="status-step {state}"><span class="status-dot"></span>'
                f'<span class="status-label">{_localized_text("status_step_" + key, english, chinese)}</span></div>'
            )
        latest = self.message
        if self._processing_started is not None:
            elapsed = max(0.0, now - self._processing_started)
            latest = f"Processing on server ({elapsed:.1f}s)"
        elif self._queue_started is not None:
            latest += "." * (int(max(0.0, now - self._queue_started)) % 10 + 1)
        elif completed and self.processing_elapsed is not None:
            latest = f"{STATUS_COMPLETED} ({self.processing_elapsed:.1f}s)"
        if self.message == DEFAULT_STATUS:
            title = _localized_text("status_idle_title", "Waiting", "等待任务")
            latest_html = _localized_text("status_idle_hint", DEFAULT_STATUS, "上传文件后开始转换。")
        else:
            title = _localized_text("status_latest", "Latest status", "最新状态")
            latest_html = html.escape(latest)
        return (
            '<div class="status-steps-panel">'
            f'<div class="status-panel-title">{title}</div>'
            f'<div class="status-steps-list">{"".join(items)}</div>'
            f'<div class="status-latest">{latest_html}</div></div>'
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
    """同时等待任务、状态通知和本地动画时钟，并及时回收临时等待任务。"""
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
            done, _ = await asyncio.wait({task, waiter}, timeout=state.refresh_interval, return_when=asyncio.FIRST_COMPLETED)
            if waiter in done:
                message, at = waiter.result()
                if state.append(message, at=at):
                    updated_html = state.render()
            elif not done:
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
