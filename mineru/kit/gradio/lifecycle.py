"""为 Gradio 同步组装提供有明确归属的临时事件循环。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

__all__ = ["with_setup_loop"]

_P = ParamSpec("_P")
_T = TypeVar("_T")


def with_setup_loop(setup: Callable[_P, _T]) -> Callable[_P, _T]:
    """复用宿主运行中的循环；同步入口只创建并关闭自己拥有的组装循环。"""

    @wraps(setup)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        """避免 Gradio 的锁和事件工厂在同步调用中反复创建无主循环。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return setup(*args, **kwargs)

        async def invoke() -> _T:
            """在运行中的循环内执行同步组装，异步服务仍由 Gradio 自行托管。"""
            return setup(*args, **kwargs)

        # 显式工厂不会覆盖宿主尚未运行的当前循环；Runner 负责异常时的排空和关闭。
        with asyncio.Runner(loop_factory=asyncio.new_event_loop) as runner:
            # 不使用 Runner.run()，保留阻塞式 launch 对首次 Ctrl+C 的原有处理。
            return runner.get_loop().run_until_complete(invoke())

    return wrapped
