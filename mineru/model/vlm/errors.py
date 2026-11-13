# Copyright (c) Opendatalab. All rights reserved.
"""仅识别推理引擎明确报告的不可恢复错误，不依赖模糊日志关键词。"""

from __future__ import annotations

from mineru_vl_utils.vlm_client.base_client import EngineUnavailableError

from ...errors import MineruError


class EngineNotReadyError(MineruError):
    """健康探测暂时不可用，拒绝新推理但不把实例永久标记为死亡。"""

    def __init__(self, message: str) -> None:
        """保留现有 API 错误码供本地、远程和批处理一致处理。"""
        super().__init__("engine_unavailable", message)


class EngineDeadError(EngineUnavailableError):
    """表示本地引擎已死亡，当前进程不能继续接受推理。"""


def is_fatal_engine_error(error: BaseException) -> bool:
    """只消费依赖库公共故障契约，不维护第二份后端类型识别规则。"""
    return EngineUnavailableError.find_in(error) is not None


__all__ = ["EngineDeadError", "EngineNotReadyError", "is_fatal_engine_error"]
