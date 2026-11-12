# Copyright (c) Opendatalab. All rights reserved.
"""仅识别推理引擎明确报告的不可恢复错误，不依赖模糊日志关键词。"""

from __future__ import annotations

# 按“模块根名 + 类名”匹配，避免为一个可选异常给后端客户端加硬依赖。
_FATAL_ENGINE_TYPES: dict[str, frozenset[str]] = {
    "vllm": frozenset({"EngineDeadError", "AsyncEngineDeadError"}),
    # mineru_vl_utils 在 LMDeploy 引擎停止（error_code=engine_unavailable）时抛出。
    "mineru_vl_utils": frozenset({"EngineUnavailableError"}),
}


class EngineDeadError(RuntimeError):
    """表示本地引擎已死亡，当前进程不能继续接受推理。"""


def _is_fatal_engine_type(error: BaseException) -> bool:
    """按模块根名匹配后端报告的死亡类型，同名但来源不同的异常不参与匹配。"""
    for cls in type(error).__mro__:
        names = _FATAL_ENGINE_TYPES.get(cls.__module__.split(".", 1)[0])
        if names is not None and cls.__name__ in names:
            return True
    return False


def is_fatal_engine_error(error: BaseException) -> bool:
    """遍历异常链和异常组，优先匹配后端报告的明确死亡类型。"""
    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, EngineDeadError) or _is_fatal_engine_type(current):
            return True
        if any(cls.__name__ == "BaseExceptionGroup" for cls in type(current).__mro__):
            pending.extend(item for item in getattr(current, "exceptions", ()) if isinstance(item, BaseException))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif current.__context__ is not None and not current.__suppress_context__:
            pending.append(current.__context__)
    return False


__all__ = ["EngineDeadError", "is_fatal_engine_error"]
