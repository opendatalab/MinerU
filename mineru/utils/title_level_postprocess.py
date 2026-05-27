# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from copy import deepcopy
from typing import Any

from mineru.utils.config_reader import get_llm_aided_config
from mineru.utils.enum_class import BlockType
from mineru.utils.llm_aided import llm_aided_title


SUPPORTED_PDF_BACKENDS = {"pipeline", "vlm", "hybrid"}
PENDING_TITLE_ROLE_LEVELS = {1, 2}


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


def _iter_finalized_para_title_blocks(pdf_info: list[dict[str, Any]]):
    """遍历 finalized middle json 中 para_blocks 里的普通标题块。"""
    for page_info in pdf_info:
        for block in page_info.get("para_blocks", []):
            if block.get("type") == BlockType.TITLE:
                yield block


def _collect_finalized_title_levels(pdf_info: list[dict[str, Any]]) -> list[Any]:
    """收集 finalized title 的 level，用于判断是否仍处于待分级角色态。"""
    return [
        block.get("level")
        for block in _iter_finalized_para_title_blocks(pdf_info)
    ]


def _should_skip_llm_title_leveling(pdf_info: list[dict[str, Any]]) -> bool:
    """判断客户端是否应跳过 LLM 标题分级，仅保留产物重生流程。"""
    title_levels = _collect_finalized_title_levels(pdf_info)
    if len(title_levels) == 0:
        return True

    return any(
        level is not None and level not in PENDING_TITLE_ROLE_LEVELS
        for level in title_levels
    )


def _restore_title_roles_from_finalized_levels(
    pdf_info: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """根据 finalized level 临时恢复 para title 的 doc/paragraph 角色。"""
    restored_blocks = []
    for block in _iter_finalized_para_title_blocks(pdf_info):
        level = block.get("level")
        if level == 1:
            block["type"] = BlockType.DOC_TITLE
        elif level == 2:
            block["type"] = BlockType.PARAGRAPH_TITLE
        else:
            continue
        restored_blocks.append(block)

    return restored_blocks


def _normalize_restored_para_title_types(blocks: list[dict[str, Any]]) -> None:
    """将临时恢复过角色的 para title 兜底归一化回普通 title。"""
    for block in blocks:
        if block.get("type") in (BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE):
            block["type"] = BlockType.TITLE


def apply_title_leveling_to_middle_json(
    middle_json: dict[str, Any],
    title_aided_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """对 PDF middle json 执行外置标题分级，并返回写回后的 middle json。"""
    pdf_info = _validate_pdf_middle_json(middle_json)

    if _should_skip_llm_title_leveling(pdf_info):
        return middle_json

    resolved_config = _resolve_title_aided_config(title_aided_config)

    restored_blocks = _restore_title_roles_from_finalized_levels(pdf_info)
    try:
        llm_aided_title(pdf_info, deepcopy(resolved_config))
    finally:
        _normalize_restored_para_title_types(restored_blocks)

    return middle_json
