"""Search service: content search and filename search."""

from __future__ import annotations

from typing import cast

from ...types import TIER_ORDER, Tier
from ..core.db import DatabaseManager
from ..core.fts import FTSManager, strip_sep
from ..rows import ContentSearchResultRow, FilenameSearchFileRow, FilenameSearchResultRow, SearchFileRow
from ..types import FILE_STATUS_ACTIVE


class SearchService:
    def __init__(self, db: DatabaseManager, fts: FTSManager) -> None:
        self.db = db
        self.fts = fts

    # ── content search ──────────────────────────────────────────

    async def search(
        self,
        query: str,
        file_type: str | None = None,
        tier: Tier | None = None,
        min_tier: Tier | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[ContentSearchResultRow], int]:
        """Full-text search. Returns (results, total_count)."""
        rows = await self.fts.search(query, limit=200)
        if not rows:
            return [], 0

        rows = [row for row in rows if _matches_tier(row.get("tier"), tier=tier, min_tier=min_tier)]
        if not rows:
            return [], 0

        # dedup by sha256
        sha256s = list(dict.fromkeys(r["sha256"] for r in rows))

        # join with docs + files
        placeholders = ",".join("?" * len(sha256s))
        sql = (
            "SELECT f.*, d.title, d.author, d.page_count, d.file_type "
            "FROM files f JOIN docs d ON f.sha256 = d.sha256 "
            f"WHERE f.sha256 IN ({placeholders})"
        )
        params = [*sha256s]

        if file_type:
            sql += " AND d.file_type = ?"
            params.append(file_type.lower())

        file_rows = cast(list[SearchFileRow], await self.db.fetchall(sql, tuple(params)))

        # group files by sha256
        files_by_doc: dict[str, list[SearchFileRow]] = {}
        for fr in file_rows:
            sha = fr["sha256"]
            if sha is None:
                continue
            if sha not in files_by_doc:
                files_by_doc[sha] = []
            files_by_doc[sha].append(fr)

        # build results
        results: list[ContentSearchResultRow] = []
        for row in rows:
            sha = row["sha256"]
            all_files = files_by_doc.get(sha, [])
            if not all_files:
                continue

            active_files = [file for file in all_files if file.get("status") == FILE_STATUS_ACTIVE]
            files = active_files or all_files
            fts_file = files[0]
            snippet = strip_sep(row.get("snippet", ""))
            result_tier = row["tier"]

            results.append(
                {
                    "sha256": sha,
                    "title": row.get("title") or fts_file.get("title"),
                    "author": row.get("author") or fts_file.get("author"),
                    "filename": row.get("filename") or fts_file.get("filename"),
                    "ext": fts_file.get("ext", ""),
                    "size_bytes": fts_file.get("size_bytes", 0),
                    "tier": result_tier,
                    "snippet": snippet,
                    "paths": [f["path"] for f in files],
                }
            )

        total = len(results)
        return results[offset : offset + limit], total

    # ── filename search ─────────────────────────────────────────

    async def search_filenames(
        self, query: str, ext: str | None = None, limit: int = 50
    ) -> tuple[list[FilenameSearchResultRow], int]:
        """Search filenames only. Returns (results, total_count)."""
        rows = await self.fts.search_filenames(query, limit=limit)
        if not rows:
            return [], 0

        file_ids = [r["file_id"] for r in rows]
        placeholders = ",".join("?" * len(file_ids))
        sql = (
            f"SELECT f.*, d.title "
            f"FROM files f LEFT JOIN docs d ON f.sha256 = d.sha256 "
            f"WHERE f.id IN ({placeholders}) AND f.status = ?"
        )
        params = [*file_ids, FILE_STATUS_ACTIVE]
        if ext:
            sql += " AND f.ext = ?"
            params.append(ext.lower().lstrip("."))
        file_rows = cast(list[FilenameSearchFileRow], await self.db.fetchall(sql, tuple(params)))

        files_by_id = {fr["id"]: fr for fr in file_rows}

        results: list[FilenameSearchResultRow] = []
        for row in rows:
            fr = files_by_id.get(row["file_id"])
            if not fr:
                continue
            results.append(
                {
                    "sha256": fr.get("sha256", ""),
                    "title": fr.get("title"),
                    "filename": fr["filename"],
                    "ext": fr.get("ext", ""),
                    "size_bytes": fr.get("size_bytes", 0),
                    "tier": "",
                    "snippet": strip_sep(row.get("snippet", "")),
                    "paths": [fr["path"]],
                }
            )

        total = len(results)
        return results, total


def _matches_tier(value: object, *, tier: Tier | None, min_tier: Tier | None) -> bool:
    if not isinstance(value, str) or value not in TIER_ORDER:
        return False
    if tier is not None and value != tier:
        return False
    if min_tier is not None and TIER_ORDER[value] < TIER_ORDER[min_tier]:
        return False
    return True
