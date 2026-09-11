# Copyright (c) Opendatalab. All rights reserved.
"""共享确定性文档后处理，并在 MinerU 边界执行可选智能增强。"""

from __future__ import annotations
from docvortex.postprocess.document import model_json_to_middle_json as build_middle_json
from ...config import LLMAidedConfig
from ...types import MiddleJson, ModelJson
from ...utils.async_utils import run_sync
from .llm_aided import aio_apply_llm_aided_postprocess, apply_llm_aided_postprocess


def model_json_to_middle_json(model_json: ModelJson, *, llm_aided_config: LLMAidedConfig) -> MiddleJson:
    """只由宿主显式启用 LLM，DocVortex 的默认后处理完整且独立。"""
    middle_json = build_middle_json(model_json)
    if model_json.metadata.file_suffix == "pdf":
        apply_llm_aided_postprocess(middle_json, llm_aided_config)
    return middle_json


async def aio_model_json_to_middle_json(model_json: ModelJson, *, llm_aided_config: LLMAidedConfig) -> MiddleJson:
    """确定性处理在线程执行，可选 LLM 增强直接异步等待。"""
    middle_json = await run_sync(build_middle_json, model_json)
    if model_json.metadata.file_suffix == "pdf":
        await aio_apply_llm_aided_postprocess(middle_json, llm_aided_config)
    return middle_json


__all__ = ["model_json_to_middle_json", "aio_model_json_to_middle_json"]
