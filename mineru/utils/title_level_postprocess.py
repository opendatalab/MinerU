# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from loguru import logger

from ..types import PageInfo
from .backend_options import DEFAULT_HYBRID_EFFORT, validate_effort
from .config_reader import get_llm_aided_config
from .llm_aided import llm_aided_title


def _resolve_title_aided_config() -> dict[str, Any] | None:
    """从本地配置解析标题分级开关。"""
    llm_aided_config = get_llm_aided_config()
    title_aided_config = llm_aided_config.get("title_aided") if isinstance(llm_aided_config, dict) else None
    if not isinstance(title_aided_config, dict) or not title_aided_config:
        return None
    if not title_aided_config.get("enable", False):
        return None
    return deepcopy(title_aided_config)


def apply_title_leveling_to_pdf_info(pdf_info: list[PageInfo]) -> None:
    title_aided_config = _resolve_title_aided_config()
    if title_aided_config:
        start_time = time.perf_counter()
        success = False
        try:
            llm_aided_title(pdf_info, title_aided_config)
            success = True
        finally:
            elapsed = time.perf_counter() - start_time
            status = "finished" if success else "failed"
            logger.info(f"title leveling {status}, cost: {elapsed:.2f}s")


def finalize_client_side_pages(pages: list[PageInfo], backend: str, effort: str = DEFAULT_HYBRID_EFFORT) -> None:
    """按调用方传入的后端类型，对 pages 原地执行客户端可完成的 finalize。"""
    if backend == "pipeline":
        from mineru.backend.pipeline.model_output_to_middle_json import finalize_middle_json_from_preproc

        finalize_middle_json_from_preproc(pages)
    elif backend == "hybrid":
        from mineru.backend.hybrid.model_output_to_middle_json import finalize_middle_json_from_preproc

        finalize_middle_json_from_preproc(pages, effort=validate_effort(effort))
