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
    """统一读取 2.0 batch，或把 3.4.5/1.0 页面转换为当前 page 字典。"""
    schema_version = batch_payload.get("schema_version")
    raw_pages = batch_payload.get("pages")
    if schema_version == MIDDLE_JSON_SCHEMA_VERSION:
        return raw_pages if isinstance(raw_pages, list) else []

    if schema_version == "1.0":
        pass
    elif schema_version is None:
        raw_pages = batch_payload.get("pdf_info")
    else:
        raw_pages = None
    if not isinstance(raw_pages, list) or any(not isinstance(page, dict) for page in raw_pages):
        raise ValueError("stale Middle JSON cache requires source reparse")

    from ...backend.postprocess.legacy_schema_adapter import legacy_page_to_model_list
    from ...backend.postprocess.pages import model_json_to_pages
    from ...parser.base import (
        _legacy_effort,
        _legacy_file_suffix,
        _legacy_page_index_map,
        _legacy_parse_mode,
    )
    from ...types import ModelJson
    from ...version import __version__ as current_mineru_version

    source_version = batch_payload.get("_version_name", batch_payload.get("mineru_version"))
    mineru_version = (
        source_version.strip() if isinstance(source_version, str) and source_version.strip() else current_mineru_version
    )
    model_json = ModelJson(
        pages=[legacy_page_to_model_list(page) for page in raw_pages],
        page_index_map=_legacy_page_index_map(raw_pages),
        file_suffix=_legacy_file_suffix(batch_payload),
        effort=_legacy_effort(batch_payload),
        parse_mode=_legacy_parse_mode(batch_payload),
        mineru_version=mineru_version,
    )
    return [page.to_dict() for page in model_json_to_pages(model_json)]


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

        loaded = self._load_batch_payloads(sha256, tier, rows)
        if loaded is None:
            return 0
        pages_by_page_idx, envelope = loaded

        # collect all done page numbers
        all_page_numbers = {page_idx + 1 for page_idx in pages_by_page_idx}
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

        compacted_paths = self._write_compacted_json_files(
            sha256,
            tier,
            merged_ranges,
            max_done_at,
            pages_by_page_idx,
            envelope,
        )
        if compacted_paths is None:
            return 0

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

        self._delete_obsolete_json_files(sha256, tier, compacted_paths)

        actual_page_count = max(pages_by_page_idx) + 1
        await self.db.execute(
            "UPDATE docs SET page_count=?, updated_at=? WHERE sha256=? AND (file_type IS NULL OR file_type<>?)",
            (actual_page_count, now, sha256, "pdf"),
        )

        return len(rows) - len(merged_ranges)

    def _load_batch_payloads(
        self,
        sha256: str,
        tier: Tier,
        done_rows: Sequence[ParseBatchRow],
    ) -> tuple[dict[int, dict[str, Any]], dict[str, Any]] | None:
        """完整读取 done batch；任一源文件损坏时放弃本轮压缩。"""
        pages_by_page_idx: dict[int, dict[str, Any]] = {}
        envelope: dict[str, Any] = {}
        for row in reversed(done_rows):
            fpath = parse_batch_json_path(self.data_dir, sha256, tier, row["page_range"], row["done_at"])
            if not os.path.isfile(fpath):
                return None
            try:
                with open(fpath, encoding="utf-8") as f:
                    batch_payload = json.load(f)
                batch_pages = _normalize_batch_pages(batch_payload)
            except Exception:
                return None
            if not batch_pages:
                return None
            for page in batch_pages:
                page_idx = page.get("page_idx")
                if type(page_idx) is not int or page_idx < 0:
                    return None
                pages_by_page_idx[page_idx] = page
            if not envelope:
                envelope = {
                    key: batch_payload[key]
                    for key in ("is_full_document", "file_suffix", "effort", "parse_mode", "mineru_version")
                    if key in batch_payload
                }
        return (pages_by_page_idx, envelope) if pages_by_page_idx else None

    def _write_compacted_json_files(
        self,
        sha256: str,
        tier: Tier,
        merged_ranges: Sequence[str],
        max_done_at: int,
        pages_by_page_idx: dict[int, dict[str, Any]],
        envelope: dict[str, Any],
    ) -> set[str] | None:
        """先完整写入临时 JSON，再原子提升所有目标文件。"""
        prepared: list[tuple[str, str]] = []
        try:
            for page_range in merged_ranges:
                page_numbers = parse_page_range_set(page_range)
                json_pages = [
                    pages_by_page_idx[page_no - 1] for page_no in sorted(page_numbers) if page_no - 1 in pages_by_page_idx
                ]
                if not json_pages:
                    raise ValueError(f"Compacted page range has no source pages: {page_range}")
                final_path = parse_batch_json_path(self.data_dir, sha256, tier, page_range, max_done_at)
                temp_path = f"{final_path}.tmp-{time.time_ns()}"
                payload: dict[str, Any] = {"schema_version": MIDDLE_JSON_SCHEMA_VERSION, "pages": json_pages}
                payload.update(envelope)
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                prepared.append((temp_path, final_path))
            for temp_path, final_path in prepared:
                os.replace(temp_path, final_path)
            return {final_path for _, final_path in prepared}
        except Exception:
            for temp_path, _ in prepared:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            return None

    def _delete_obsolete_json_files(self, sha256: str, tier: Tier, keep_paths: set[str]) -> None:
        """仅在新缓存和数据库均就绪后删除已被替代的 JSON。"""
        tier_dir = os.path.join(self.data_dir, "parsed", sha256[:2], sha256, tier)
        for fname in os.listdir(tier_dir):
            path = os.path.join(tier_dir, fname)
            if fname.endswith(".json") and path not in keep_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass

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
        loaded = self._load_batch_payloads(sha256, tier, done_rows)
        if loaded is None:
            return
        pages_by_page_idx, envelope = loaded
        compacted_paths = self._write_compacted_json_files(
            sha256,
            tier,
            merged_ranges,
            max_done_at,
            pages_by_page_idx,
            envelope,
        )
        if compacted_paths is None:
            return
        self._delete_obsolete_json_files(sha256, tier, compacted_paths)
