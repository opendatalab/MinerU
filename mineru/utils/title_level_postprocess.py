# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from copy import deepcopy
from typing import Any

from mineru.utils.config_reader import get_llm_aided_config
from mineru.utils.llm_aided import llm_aided_title


SUPPORTED_PDF_BACKENDS = {"pipeline", "vlm", "hybrid"}


def _resolve_title_aided_config() -> dict[str, Any] | None:
    """从本地配置解析标题分级开关。"""
    llm_aided_config = get_llm_aided_config()
    title_aided_config = (
        llm_aided_config.get("title_aided")
        if isinstance(llm_aided_config, dict)
        else None
    )
    if not isinstance(title_aided_config, dict) or not title_aided_config:
        return None
    if not title_aided_config.get("enable", False):
        return None
    return deepcopy(title_aided_config)


def apply_title_leveling_to_pdf_info(pdf_info: list[dict[str, Any]]):
    title_aided_config = _resolve_title_aided_config()
    if title_aided_config:
        llm_aided_title(pdf_info, title_aided_config)


def finalize_client_side_middle_json(middle_json: dict[str, Any]) -> dict[str, Any]:
    """根据 staged middle json 的后端类型，在客户端执行完整 finalize。"""
    if not isinstance(middle_json, dict):
        raise ValueError("middle_json must be a dict.")

    backend = middle_json.get("_backend")
    pdf_info = middle_json.get("pdf_info")

    if backend == "pipeline":
        from mineru.backend.pipeline.model_json_to_middle_json import (
            finalize_middle_json_from_preproc,
        )

        finalize_middle_json_from_preproc(
            pdf_info,
        )
    elif backend == "vlm":
        from mineru.backend.vlm.model_output_to_middle_json import (
            finalize_middle_json_from_preproc,
        )

        finalize_middle_json_from_preproc(
            pdf_info,
        )
    elif backend == "hybrid":
        from mineru.backend.hybrid.hybrid_model_output_to_middle_json import (
            finalize_middle_json_from_preproc,
        )

        finalize_middle_json_from_preproc(
            pdf_info,
        )

    return middle_json
