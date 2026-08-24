"""Compaction — merges overlapping / adjacent done parse batches to keep the parses table lean."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Sequence
from typing import Any, cast

from ...parser.base import MIDDLE_JSON_SCHEMA_VERSION
from ...types import Tier
from ..core.db import DatabaseManager
from ..rows import ParseBatchRow, ParseGroupRow, ParseRow
from ..services.parse_svc import parse_batch_json_path, parse_page_range_set
from ..types import PARSE_STATUS_DONE, PARSE_STATUS_SUPERSEDED

logger = logging.getLogger("mineru.compaction")


def _normalize_batch_pages(batch_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """把 batch JSON 的 pages 统一为 2.0 schema 的 page dict 列表。

    2.0 batch：page 只有 page_idx + blocks，直接用。
    1.0 batch：page 含 preproc_blocks/para_blocks/discarded_blocks/page_size，
    走 legacy_schema_adapter 回推为 raw model_list，再走 model_list_to_pages 转 2.0。
    """
    raw_pages = batch_payload.get("pages", [])
    if batch_payload.get("schema_version") == MIDDLE_JSON_SCHEMA_VERSION:
        return raw_pages

    # 1.0 batch：走 legacy 适配器转换
    from ...backend.postprocess.legacy_schema_adapter import legacy_page_to_model_list
    from ...backend.postprocess.pages import model_list_to_pages

    model_list = [legacy_page_to_model_list(page) for page in raw_pages]
    if not model_list or not any(model_list):
        return []
    pages = model_list_to_pages(model_list)
    return [page.to_dict() for page in pages]


class Compaction:
    def __init__(self, db: DatabaseManager, interval_sec: int, data_dir: str) -> None:
        self.db = db
        self.interval_sec = interval_sec
        self.data_dir = os.path.expanduser(data_dir)
        self.running = False

    async def run(self) -> None:
        self.running = True
        while self.running:
            await asyncio.sleep(self.interval_sec)
            if not self.running:
                break
            try:
                merged = await self._compact()
                if merged > 0:
                    logger.info(f"Compaction merged {merged} parse batches")
            except Exception as exc:
                logger.error(f"Compaction error: {exc}")

    async def stop(self) -> None:
        self.running = False

    async def _compact(self) -> int:
        """Scan all (sha256, tier) pairs with multiple done batches and merge them."""
        rows = cast(
            list[ParseGroupRow],
            await self.db.fetchall(
                "SELECT sha256, tier FROM parses WHERE status=? GROUP BY sha256, tier HAVING COUNT(*) > 1",
                (PARSE_STATUS_DONE,),
            ),
        )
        total_merged = 0
        for r in rows:
            merged = await self._compact_doc_tier(r["sha256"], r["tier"])
            total_merged += merged
        return total_merged

    async def _compact_doc_tier(self, sha256: str, tier: Tier) -> int:
        rows = cast(
            list[ParseRow],
            await self.db.fetchall(
                "SELECT * FROM parses WHERE sha256=? AND tier=? AND status=? ORDER BY done_at DESC",
                (sha256, tier, PARSE_STATUS_DONE),
            ),
        )
        if len(rows) <= 1:
            return 0

        # collect all done page numbers
        all_page_numbers: set[int] = set()
        max_done_at = 0
        for r in rows:
            all_page_numbers |= parse_page_range_set(r["page_range"])
            if r["done_at"] and r["done_at"] > max_done_at:
                max_done_at = r["done_at"]

        # merge contiguous ranges
        sorted_page_numbers = sorted(all_page_numbers)
        merged_ranges: list[str] = []
        start = sorted_page_numbers[0]
        end = start
        for page_no in sorted_page_numbers[1:]:
            if page_no == end + 1:
                end = page_no
            else:
                merged_ranges.append(f"{start}~{end}" if start != end else str(start))
                start = page_no
                end = page_no
        merged_ranges.append(f"{start}~{end}" if start != end else str(start))

        # check if merge actually reduced row count
        if len(merged_ranges) >= len(rows):
            return 0  # no benefit

        # atomic replace
        now = int(time.time() * 1000)
        await self.db.execute(
            "DELETE FROM parses WHERE sha256=? AND tier=? AND status=?",
            (sha256, tier, PARSE_STATUS_DONE),
        )
        await self.db.execute(
            "DELETE FROM parses WHERE sha256=? AND tier=? AND status=?",
            (sha256, tier, PARSE_STATUS_SUPERSEDED),
        )
        for page_range in merged_ranges:
            await self.db.execute(
                "INSERT INTO parses (sha256, tier, page_range, status, done_at, priority, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
                (sha256, tier, page_range, PARSE_STATUS_DONE, max_done_at, now, now),
            )

        # compact JSON files (only from done batches, not superseded)
        await self._compact_json(sha256, tier, merged_ranges, rows, max_done_at)

        return len(rows) - len(merged_ranges)

    async def _compact_json(
        self,
        sha256: str,
        tier: Tier,
        merged_ranges: list[str],
        done_rows: Sequence[ParseBatchRow],
        max_done_at: int,
    ) -> None:
        """Merge per-batch JSON files to match compacted parses rows.
        Only reads files belonging to *done_rows* — ignores superseded files."""
        tier_dir = os.path.join(self.data_dir, "parsed", sha256[:2], sha256, tier)
        if not os.path.isdir(tier_dir):
            return

        # only read files from done batches (exclude superseded)
        # process oldest first → newest overwrites (done_rows is sorted by done_at DESC)
        pages_by_page_idx: dict[int, dict] = {}
        envelope: dict[str, Any] = {}
        for row in reversed(done_rows):
            fpath = parse_batch_json_path(self.data_dir, sha256, tier, row["page_range"], row["done_at"])
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, encoding="utf-8") as f:
                    batch_payload = json.load(f)
            except Exception:
                continue
            batch_pages = _normalize_batch_pages(batch_payload)
            for p in batch_pages:
                pages_by_page_idx[p["page_idx"]] = p
            # 继承源 batch JSON 的 envelope 元数据（2.0 schema 字段）
            if not envelope:
                envelope = {
                    k: batch_payload[k] for k in ("file_suffix", "effort", "parse_mode", "mineru_version") if k in batch_payload
                }

        if not pages_by_page_idx:
            return

        # delete old files
        for fname in os.listdir(tier_dir):
            if fname.endswith(".json"):
                try:
                    os.unlink(os.path.join(tier_dir, fname))
                except OSError:
                    pass

        # write one compacted JSON per merged range
        for page_range in merged_ranges:
            page_numbers = parse_page_range_set(page_range)
            json_pages = [
                pages_by_page_idx[page_no - 1] for page_no in sorted(page_numbers) if page_no - 1 in pages_by_page_idx
            ]
            if not json_pages:
                continue
            json_path = parse_batch_json_path(self.data_dir, sha256, tier, page_range, max_done_at)
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    # compacted payload 用 2.0 schema，继承源 batch 的 envelope 元数据
                    payload: dict[str, Any] = {"schema_version": MIDDLE_JSON_SCHEMA_VERSION, "pages": json_pages}
                    payload.update(envelope)
                    json.dump(payload, f, ensure_ascii=False, indent=4)
            except Exception:
                pass
