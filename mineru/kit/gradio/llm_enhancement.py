"""WebUI 输出阶段的有界 LLM 增强，不改变解析服务和文档协议。"""

from __future__ import annotations

import asyncio
import time

from loguru import logger

from ...config import LLMAidedConfig, LLMAidedFeaturesConfig
from ...types import MiddleJson
from ...utils.async_utils import drain_future

ENHANCEMENT_TIMEOUT_SECONDS = 60.0


async def enhance_outputs(
    middle_json: MiddleJson,
    config: LLMAidedConfig,
    *,
    title_leveling: bool | None,
    cross_page_table_cell_merge: bool | None,
) -> tuple[str, ...]:
    """按本次选择执行增强，失败返回本地化提示键，用户取消仍向上传播。"""
    title_enabled = config.features.title_leveling if title_leveling is None else title_leveling
    table_enabled = (
        config.features.cross_page_table_cell_merge if cross_page_table_cell_merge is None else cross_page_table_cell_merge
    )
    if not title_enabled and not table_enabled:
        return ()
    if not config.api_key.strip():
        return ("llm_unavailable",)

    from ...backend.postprocess.llm_aided import aio_apply_llm_aided_postprocess
    from ...backend.postprocess.llm_client import LLMAidedClient
    from ...backend.postprocess.llm_report import LLMAidedReport

    request_config = config.model_copy(
        update={
            "features": LLMAidedFeaturesConfig(
                title_leveling=title_enabled,
                cross_page_table_cell_merge=table_enabled,
            )
        }
    )
    report = LLMAidedReport()
    client: LLMAidedClient | None = None
    warnings: list[str] = []
    started = time.monotonic()
    try:
        async with asyncio.timeout(ENHANCEMENT_TIMEOUT_SECONDS):
            client = LLMAidedClient(request_config, request_timeout=20.0, transport_retries=0)
            await aio_apply_llm_aided_postprocess(
                middle_json,
                request_config,
                client=client,
                allow_partial_title_leveling=True,
                report=report,
            )
    except TimeoutError:
        warnings.append("llm_timeout")
    except Exception as exc:
        logger.warning("WebUI LLM enhancement failed: {}", type(exc).__name__)
        warnings.append("llm_failed")
    finally:
        if client is not None:
            closing = asyncio.create_task(client.close())
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                await drain_future(closing)
                raise
            except Exception as exc:
                logger.warning("WebUI LLM client cleanup failed: {}", type(exc).__name__)
        logger.debug("WebUI LLM outputs elapsed={:.3f}s report={}", time.monotonic() - started, report)
    for feature, key in (
        (report.title_leveling, "llm_title_incomplete"),
        (report.cross_page_table_cell_merge, "llm_cells_incomplete"),
    ):
        if feature.failed or feature.pending:
            warnings.append(key)
    return tuple(warnings)


__all__ = ["enhance_outputs"]
