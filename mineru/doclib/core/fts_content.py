"""Doclib 检索文本构造与缓存重建，不把展示用图片载荷写入索引。"""

from __future__ import annotations

import logging
from typing import cast

from docvortex.options import LatexDelimitersConfig

from ...errors import MineruError
from ...parser.page_range import parse_page_range_set
from ...types import TIER_ORDER, BlockBase, MiddleJson, Tier
from ..rows import ParseRow
from ..types import PARSE_STATUS_DONE
from .db import DatabaseManager
from .fts import FTSManager, truncate_head_tail
from .middle_json import read_cached_middle_json
from .parse_artifacts import load_pages_from_done_batches, parse_batch_is_readable, parse_batch_json_path

logger = logging.getLogger("mineru.doclib.search_index")


def _omit_image(_block: BlockBase) -> str:
    """索引只保留视觉块的文字内容，图片通过 Doclib read 按需读取。"""
    return ""


def render_fts_text(middle_json: MiddleJson, *, latex_delimiters: LatexDelimitersConfig) -> str:
    """沿用正文渲染和跨页续接规则，抑制图片后才应用索引字符预算。"""
    from docvortex.render.markdown import render_markdown

    return truncate_head_tail(render_markdown(middle_json, latex_delimiters=latex_delimiters, image_renderer=_omit_image))


async def rebuild_fts_from_cache(
    db: DatabaseManager,
    fts: FTSManager,
    *,
    data_dir: str,
    sha256: str,
    latex_delimiters: LatexDelimitersConfig,
    require_complete_cache: bool = False,
) -> None:
    """按最高可读档位重建索引；升级迁移要求完整缓存，且不依赖源文件仍然存在。"""
    doc = await db.fetchone("SELECT title, author FROM docs WHERE sha256=?", (sha256,))
    rows = cast(
        list[ParseRow],
        await db.fetchall(
            "SELECT * FROM parses WHERE sha256=? AND status=? ORDER BY done_at DESC, id DESC",
            (sha256, PARSE_STATUS_DONE),
        ),
    )
    tiers: list[Tier] = sorted({row["tier"] for row in rows}, key=lambda tier: TIER_ORDER[tier], reverse=True)
    for tier in tiers:
        tier_rows = [row for row in rows if row["tier"] == tier]
        readable = [row for row in tier_rows if parse_batch_is_readable(data_dir, sha256, tier, row)]
        pages = load_pages_from_done_batches(data_dir, sha256, tier, readable)
        if require_complete_cache:
            expected = set().union(*(parse_page_range_set(row["page_range"]) for row in tier_rows))
            if not pages or expected - {page.page_idx + 1 for page in pages}:
                raise MineruError("stale_cache", "Search index rebuild requires readable cached parse pages.")
        if not pages:
            continue
        # 保留真实批次的文档身份与元数据，用按最新记录合并的页树进行整本文本渲染。
        newest = readable[0]
        template = read_cached_middle_json(
            parse_batch_json_path(data_dir, sha256, tier, newest["page_range"], newest["done_at"])
        )
        middle_json = template.model_copy(update={"pages": pages})
        await fts.replace(
            sha256=sha256,
            tier=tier,
            text=render_fts_text(middle_json, latex_delimiters=latex_delimiters),
            title=(doc or {}).get("title") or "",
            author=(doc or {}).get("author") or "",
        )
        return
    if require_complete_cache:
        raise MineruError("stale_cache", "Search index rebuild requires a completed parse cache.")
    await fts.delete(sha256)


async def migrate_fts_contents(
    db: DatabaseManager, fts: FTSManager, *, data_dir: str, latex_delimiters: LatexDelimitersConfig
) -> None:
    """启动接单前分批迁移旧索引；成功记录随替换原子清除，失败保留原因供下次重试。"""
    cursor = ""
    while rows := await db.fetchall("SELECT sha256 FROM fts_rebuild_queue WHERE sha256>? ORDER BY sha256 LIMIT 100", (cursor,)):
        for row in rows:
            sha256 = row["sha256"]
            try:
                await rebuild_fts_from_cache(
                    db,
                    fts,
                    data_dir=data_dir,
                    sha256=sha256,
                    latex_delimiters=latex_delimiters,
                    require_complete_cache=True,
                )
            except (OSError, ValueError, TypeError, MineruError) as exc:
                await db.execute("UPDATE fts_rebuild_queue SET error=? WHERE sha256=?", (str(exc)[:500], sha256))
                logger.warning(
                    "Search index rebuild deferred for doc=%s: %s. The old index is excluded from search; "
                    "restore or reparse its cache and restart Doclib to retry.",
                    sha256,
                    exc,
                )
        cursor = rows[-1]["sha256"]


__all__ = ["render_fts_text", "rebuild_fts_from_cache", "migrate_fts_contents"]
