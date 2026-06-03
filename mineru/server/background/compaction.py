"""Compaction — merges overlapping / adjacent done parse batches to keep the parses table lean."""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import time

from mineru.constants import DATA_DIR
from ..services.parse_svc import parse_range_set

logger = logging.getLogger("mineru.compaction")


class Compaction:
    def __init__(self, db, interval_sec: int = 600) -> None:
        self.db = db
        self.interval_sec = interval_sec
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
        rows = await self.db.fetchall(
            "SELECT sha256, tier FROM parses WHERE status='done' "
            "GROUP BY sha256, tier HAVING COUNT(*) > 1"
        )
        total_merged = 0
        for r in rows:
            merged = await self._compact_doc_tier(r["sha256"], r["tier"])
            total_merged += merged
        return total_merged

    async def _compact_doc_tier(self, sha256: str, tier: str) -> int:
        rows = await self.db.fetchall(
            "SELECT * FROM parses WHERE sha256=? AND tier=? AND status='done' "
            "ORDER BY done_at DESC",
            (sha256, tier),
        )
        if len(rows) <= 1:
            return 0

        # collect all done pages
        all_pages: set[int] = set()
        max_done_at = 0
        for r in rows:
            all_pages |= parse_range_set(r["pages"])
            if r["done_at"] and r["done_at"] > max_done_at:
                max_done_at = r["done_at"]

        # merge contiguous ranges
        sorted_pages = sorted(all_pages)
        merged_ranges: list[str] = []
        start = sorted_pages[0]
        end = start
        for p in sorted_pages[1:]:
            if p == end + 1:
                end = p
            else:
                merged_ranges.append(f"{start}~{end}" if start != end else str(start))
                start = p
                end = p
        merged_ranges.append(f"{start}~{end}" if start != end else str(start))

        # check if merge actually reduced row count
        if len(merged_ranges) >= len(rows):
            return 0  # no benefit

        # atomic replace
        now = int(time.time() * 1000)
        await self.db.execute(
            "DELETE FROM parses WHERE sha256=? AND tier=? AND status='done'",
            (sha256, tier),
        )
        for pages_str in merged_ranges:
            await self.db.execute(
                "INSERT INTO parses (sha256, tier, pages, status, done_at, priority, "
                "created_at, updated_at) VALUES (?, ?, ?, 'done', ?, 0, ?, ?)",
                (sha256, tier, pages_str, max_done_at, now, now),
            )

        # compact JSON files
        await self._compact_json(sha256, tier, merged_ranges)

        return len(rows) - len(merged_ranges)

    async def _compact_json(self, sha256: str, tier: str, merged_ranges: list[str]) -> None:
        """Merge per-batch middle_*.json files to match compacted parses rows."""
        data_dir = os.path.expanduser(DATA_DIR)
        tier_dir = os.path.join(data_dir, "parsed", sha256[:2], sha256, tier)
        if not os.path.isdir(tier_dir):
            return

        # collect all pages from existing JSON files
        pages_by_idx: dict[int, dict] = {}
        for fname in os.listdir(tier_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(tier_dir, fname), encoding="utf-8") as f:
                    for p in _json.load(f).get("pdf_info", []):
                        pages_by_idx[p["page_idx"]] = p
            except Exception:
                pass

        if not pages_by_idx:
            return

        # delete old JSON files
        for fname in os.listdir(tier_dir):
            if fname.endswith(".json"):
                try:
                    os.unlink(os.path.join(tier_dir, fname))
                except OSError:
                    pass

        # write one compacted JSON per merged range
        for pages_str in merged_ranges:
            page_set = parse_range_set(pages_str)
            json_pages = [pages_by_idx[i] for i in sorted(page_set) if i in pages_by_idx]
            if not json_pages:
                continue
            json_path = os.path.join(tier_dir, f"{_safe_filename(pages_str)}.json")
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    _json.dump({"pdf_info": json_pages}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass


def _safe_filename(s: str) -> str:
    return s  # pages string is already safe: "1~5,43~47"
