# Copyright (c) Opendatalab. All rights reserved.
"""共享确定性文档后处理，并在 MinerU 边界执行可选智能增强。"""

from __future__ import annotations
from docvortex.postprocess.document import model_json_to_middle_json as build_middle_json
from ...config import LLMAidedConfig
from ...types import MiddleJson, ModelJson
from .llm_aided import apply_llm_aided_postprocess


def model_json_to_middle_json(model_json: ModelJson, *, llm_aided_config: LLMAidedConfig) -> MiddleJson:
    """只由宿主显式启用 LLM，DocVortex 的默认后处理完整且独立。"""
    middle_json = build_middle_json(model_json)
    if model_json.file_suffix == "pdf":
        apply_llm_aided_postprocess(middle_json, llm_aided_config)
    return middle_json


__all__ = ["model_json_to_middle_json"]
