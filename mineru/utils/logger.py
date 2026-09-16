# Copyright (c) Opendatalab. All rights reserved.
"""MinerU 进程内 Loguru 日志级别配置。"""

from __future__ import annotations

import sys
import threading

from ..config import LogLevel

_managed_sink_ids: set[int] = set()
_configured_level: LogLevel | None = None
_explicit_level: LogLevel | None = None
_lock = threading.Lock()


def _active_managed_sink_ids() -> set[int]:
    """返回仍然存在的 MinerU 托管 sink，避免其它入口清理后使用失效 ID。"""
    # Loguru 未公开枚举 handler 的稳定 API；这里只做只读检查，不修改私有状态。
    from loguru import logger

    return _managed_sink_ids.intersection(logger._core.handlers)


def configure_global_log_level(level: LogLevel | str | None = None) -> LogLevel:
    """配置当前进程的 Loguru 默认 stderr sink，并返回归一化后的级别。

    ``level`` 为 ``None`` 时使用 ``config.log.level``；显式传入后作为进程级
    局部覆盖，后续默认调用不会把它改回全局配置。函数只移除 Loguru 初始
    sink 和 MinerU 自己创建的 sink，不会影响宿主程序显式添加的自定义 sink。
    """
    from loguru import logger

    from ..config import GlobalLogConfig, config

    global _configured_level, _explicit_level

    with _lock:
        if level is None:
            normalized = _explicit_level or config.log.level
        else:
            normalized = GlobalLogConfig(level=level).level
            _explicit_level = normalized
        active_ids = _active_managed_sink_ids()
        has_initial_sink = 0 in logger._core.handlers
        if _configured_level == normalized and (active_ids or not has_initial_sink):
            return normalized
        if not active_ids and not has_initial_sink:
            # 宿主或 Doclib 可能已刻意移除默认 stderr sink；此时尊重现有配置，不重新添加。
            _configured_level = normalized
            return normalized

        for sink_id in active_ids:
            logger.remove(sink_id)
        _managed_sink_ids.clear()

        # Loguru 的初始 sink 固定使用 ID 0；如果宿主已经移除它，则忽略即可。
        try:
            logger.remove(0)
        except ValueError:
            pass

        managed_sink_id = logger.add(sys.stderr, level=normalized.upper())
        _managed_sink_ids.add(managed_sink_id)
        _configured_level = normalized
        return normalized


__all__ = ["LogLevel", "configure_global_log_level"]
