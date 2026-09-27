"""WebUI 会话任务、完整状态快照和同步输出工作的取消边界。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypeVar

from .status import STATUS_COMPLETED, STATUS_PREPARING_REQUEST, ParseStatusUpdate, StatusPanelState

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class ConversionRun:
    """只保留当前任务的状态和素材路径，不在会话缓存中持有解析结果。"""

    run_id: str
    state: StatusPanelState = field(init=False)
    snapshot: str = ""
    cancelled: bool = False
    terminal: bool = False
    task: asyncio.Task[tuple[Any, ...]] | None = None
    artifacts: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """使用浏览器或 API 分配的唯一标识初始化准备阶段。"""
        self.state = StatusPanelState(run_id=self.run_id)
        self.publish(STATUS_PREPARING_REQUEST)

    def publish(self, message: str | ParseStatusUpdate, *, at: float | None = None) -> None:
        """阶段或耗时变化生成新快照，终态拒绝迟到的后台通知。"""
        if self.cancelled or self.terminal or not self.state.append(message, at=at):
            return
        self.terminal = self.state.message == STATUS_COMPLETED or self.state.message.startswith("Failed:")
        self.snapshot = json.dumps(
            {
                "run_id": self.run_id,
                "sequence": self.state.sequence,
                "terminal": self.terminal,
                "html": self.state.render(),
            },
            ensure_ascii=False,
        )

    def cancel(self) -> asyncio.Task[tuple[Any, ...]] | None:
        """立即失效界面回执，实际任务在同步工作退出后释放执行槽。"""
        self.cancelled = True
        self.artifacts = None
        if self.task is not None and not self.task.done():
            self.task.cancel()
        return self.task


class SessionConversions:
    """按 Gradio 会话隔离任务，并让所有实际解析共享一个执行槽。"""

    def __init__(self) -> None:
        """在构建 WebUI 时创建注册表，不在模块导入期间分配运行资源。"""
        self.runs: dict[str, ConversionRun] = {}
        self.revisions: dict[str, int] = {}
        self.slot = asyncio.Semaphore(1)

    def start(self, session: str, run_id: str, *, revision: int | None = None) -> ConversionRun | None:
        """拒绝晚到的旧提交，替换会话任务时先取消其后台工作。"""
        if revision is not None:
            if revision <= self.revisions.get(session, 0):
                return None
            self.revisions[session] = revision
        previous = self.runs.get(session)
        if previous is not None:
            previous.cancel()
        run = ConversionRun(run_id)
        self.runs[session] = run
        return run

    def current(self, session: str, run_id: str) -> ConversionRun | None:
        """只能读取当前会话仍有效的指定任务，避免跨任务状态混入。"""
        run = self.runs.get(session)
        return run if run is not None and run.run_id == run_id and not run.cancelled else None

    def cancel(self, session: str, run_id: str | None = None) -> asyncio.Task[tuple[Any, ...]] | None:
        """按回执身份取消任务，迟到的文件切换请求不能取消新任务。"""
        run = self.runs.get(session)
        if run is None or (run_id is not None and run.run_id != run_id):
            return None
        self.runs.pop(session, None)
        return run.cancel()


async def await_task_completion(task: asyncio.Task[T]) -> T:
    """收到取消后仍等待实际工作退出，再传播取消以保持资源和槽位有效。"""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()


async def run_sync_output(function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """同步输出线程不可强制取消，必须等它退出后才能释放外层执行槽。"""
    return await await_task_completion(asyncio.create_task(asyncio.to_thread(function, *args, **kwargs)))


__all__ = ["ConversionRun", "SessionConversions", "await_task_completion", "run_sync_output"]
