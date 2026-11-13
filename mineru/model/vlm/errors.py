# Copyright (c) Opendatalab. All rights reserved.
"""仅识别推理引擎明确报告的不可恢复错误，不依赖模糊日志关键词。"""

from __future__ import annotations

from ...errors import MineruError

_FATAL_ENGINE_TYPES: dict[str, frozenset[str]] = {
    "vllm": frozenset({"EngineDeadError", "AsyncEngineDeadError"}),
    "mineru_vl_utils": frozenset({"EngineUnavailableError"}),
}


class EngineNotReadyError(MineruError):
    """健康探测暂时不可用，拒绝新推理但不把实例永久标记为死亡。"""

    def __init__(self, message: str) -> None:
        """保留现有 API 错误码供本地、远程和批处理一致处理。"""
        super().__init__("engine_unavailable", message)


class EngineDeadError(RuntimeError):
    """表示本地引擎已死亡，当前进程不能继续接受推理。"""


def is_fatal_engine_error(error: BaseException) -> bool:
    """遍历异常链和异常组，仅匹配后端明确报告的死亡类型。"""
    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, EngineDeadError):
            return True
        for cls in type(current).__mro__:
            if cls.__name__ in _FATAL_ENGINE_TYPES.get(cls.__module__.split(".", 1)[0], ()):
                return True
        if any(cls.__name__ == "BaseExceptionGroup" for cls in type(current).__mro__):
            pending.extend(item for item in getattr(current, "exceptions", ()) if isinstance(item, BaseException))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif current.__context__ is not None and not current.__suppress_context__:
            pending.append(current.__context__)
    return False


__all__ = ["EngineDeadError", "EngineNotReadyError", "is_fatal_engine_error"]
