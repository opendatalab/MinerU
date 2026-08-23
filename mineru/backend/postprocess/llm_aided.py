# Copyright (c) Opendatalab. All rights reserved.
"""编排 PDF 的后置 LLM 标题与跨页单元格增强。"""

from __future__ import annotations

import asyncio

from loguru import logger

from mineru.config import LLMAidedConfig
from mineru.types import MiddleJson

from .llm_client import LLMAidedClient
from .table_merge.llm_cell_merge import apply_llm_cross_page_cell_merge
from .title_leveling import apply_llm_title_leveling


def _resolve_enabled_features(
    config: LLMAidedConfig,
    middle_json: MiddleJson,
) -> tuple[bool, bool]:
    """根据整本输入约束解析本次实际启用的标题与表格功能。"""
    title_enabled = config.features.title_leveling and middle_json.is_full_document
    table_enabled = config.features.cross_page_table_cell_merge
    if config.features.title_leveling and not middle_json.is_full_document:
        logger.info("Skipping LLM title leveling because the input is not a full document")
    return title_enabled, table_enabled


async def _apply_llm_aided_postprocess(
    middle_json: MiddleJson,
    config: LLMAidedConfig,
    *,
    title_enabled: bool,
    table_enabled: bool,
    client: LLMAidedClient | None,
) -> None:
    """使用同一异步客户端并发执行当前实际启用的 LLM 后处理。"""
    resolved_client = client or LLMAidedClient(config)
    try:
        tasks = []
        if title_enabled:
            tasks.append(apply_llm_title_leveling(middle_json.pages, resolved_client))
        if table_enabled:
            tasks.append(apply_llm_cross_page_cell_merge(middle_json.pages, resolved_client))
        await asyncio.gather(*tasks)
    finally:
        if client is None:
            await resolved_client.close()


def apply_llm_aided_postprocess(
    middle_json: MiddleJson,
    config: LLMAidedConfig,
    *,
    client: LLMAidedClient | None = None,
) -> None:
    """按配置和整本输入约束同步桥接异步 LLM 后处理。"""
    title_enabled, table_enabled = _resolve_enabled_features(config, middle_json)
    if not title_enabled and not table_enabled:
        return

    asyncio.run(
        _apply_llm_aided_postprocess(
            middle_json,
            config,
            title_enabled=title_enabled,
            table_enabled=table_enabled,
            client=client,
        )
    )


__all__ = ["apply_llm_aided_postprocess"]
