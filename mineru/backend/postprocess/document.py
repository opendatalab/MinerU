# Copyright (c) Opendatalab. All rights reserved.
"""严格 ModelJson 到 MiddleJson 的文档级后处理编排。"""

from __future__ import annotations

from ...config import LLMAidedConfig
from ...types import MiddleJson, ModelJson

from .llm_aided import apply_llm_aided_postprocess
from .pages import model_json_to_pages


def model_json_to_middle_json(
    model_json: ModelJson,
    *,
    llm_aided_config: LLMAidedConfig,
) -> MiddleJson:
    """从严格 ModelJson 构造 MiddleJson，并执行适用的文档级增强。"""
    middle_json = MiddleJson(
        pages=model_json_to_pages(model_json),
        is_full_document=model_json.is_full_document,
        file_suffix=model_json.file_suffix,
        effort=model_json.effort,
        parse_mode=model_json.parse_mode,
        mineru_version=model_json.mineru_version,
    )
    if model_json.file_suffix == "pdf":
        apply_llm_aided_postprocess(middle_json, llm_aided_config)
    return middle_json
