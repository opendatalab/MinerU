# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from copy import deepcopy
from typing import Any

from mineru.utils.config_reader import get_llm_aided_config
from mineru.utils.llm_aided import llm_aided_title


SUPPORTED_PDF_BACKENDS = {"pipeline", "vlm", "hybrid"}


def _resolve_title_aided_config(
    title_aided_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """解析标题分级配置，显式参数优先，本地配置作为兜底。"""
    if title_aided_config is not None:
        resolved_config = title_aided_config
    else:
        llm_aided_config = get_llm_aided_config()
        resolved_config = (
            llm_aided_config.get("title_aided")
            if isinstance(llm_aided_config, dict)
            else None
        )

    if not isinstance(resolved_config, dict) or not resolved_config:
        raise ValueError("Missing llm-aided-config.title_aided for title leveling.")

    if resolved_config.get("enable", True) is False:
        raise ValueError("llm-aided-config.title_aided is disabled.")

    return resolved_config


def _validate_pdf_middle_json(middle_json: dict[str, Any]) -> list[dict[str, Any]]:
    """校验外置标题分级只处理 PDF 三后端生成的 middle json。"""
    if not isinstance(middle_json, dict):
        raise ValueError("middle_json must be a dict.")

    backend = middle_json.get("_backend")
    if backend not in SUPPORTED_PDF_BACKENDS:
        raise ValueError(f"Unsupported middle json backend for title leveling: {backend}")

    pdf_info = middle_json.get("pdf_info")
    if not isinstance(pdf_info, list):
        raise ValueError("middle_json must contain a list field named pdf_info.")

    return pdf_info


def apply_title_leveling_to_middle_json(
    middle_json: dict[str, Any],
    title_aided_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """对 PDF middle json 执行外置标题分级，并返回写回后的 middle json。"""
    pdf_info = _validate_pdf_middle_json(middle_json)
    resolved_config = _resolve_title_aided_config(title_aided_config)

    llm_aided_title(pdf_info, deepcopy(resolved_config))
    return middle_json
