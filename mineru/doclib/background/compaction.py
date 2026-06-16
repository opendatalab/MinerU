"""Compaction — merges overlapping / adjacent done parse batches to keep the parses table lean."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Sequence
from typing import cast

from ...types import Tier
from ..core.db import DatabaseManager
from ..rows import ParseBatchRow, ParseGroupRow, ParseRow
from ..services.parse_svc import parse_batch_json_path, parse_page_range_set
from ..types import PARSE_STATUS_DONE, PARSE_STATUS_SUPERSEDED

logger = logging.getLogger("mineru.compaction")


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
        for row in reversed(done_rows):
            fpath = parse_batch_json_path(self.data_dir, sha256, tier, row["page_range"], row["done_at"])
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, encoding="utf-8") as f:
                    for p in json.load(f).get("pages", []):
                        pages_by_page_idx[p["page_idx"]] = p
            except Exception:
                pass

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
            json_pages = [pages_by_page_idx[page_no - 1] for page_no in sorted(page_numbers) if page_no - 1 in pages_by_page_idx]
            if not json_pages:
                continue
            json_path = parse_batch_json_path(self.data_dir, sha256, tier, page_range, max_done_at)
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({"pages": json_pages}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
