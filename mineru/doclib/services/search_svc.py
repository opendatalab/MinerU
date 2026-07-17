"""Search service: content search and filename search."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, cast

from ...types import TIER_ORDER, Tier
from ..core.db import DatabaseManager
from ..core.fts import FTSManager, strip_sep
from ..rows import ContentSearchResultRow, DocRow, FileRow, FilenameSearchFileRow, FilenameSearchResultRow
from ..types import FILE_STATUS_ACTIVE

if TYPE_CHECKING:
    from .parse_svc import FileRefreshResult

FilenamePathProbe = Callable[[str], Awaitable["FileRefreshResult"]]


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

        # Load document metadata independently so indexed orphan documents remain searchable.
        placeholders = ",".join("?" * len(sha256s))
        doc_sql = f"SELECT * FROM docs WHERE sha256 IN ({placeholders})"
        params = [*sha256s]

        if file_type:
            doc_sql += " AND file_type = ?"
            params.append(file_type.lower())

        doc_rows = cast(list[DocRow], await self.db.fetchall(doc_sql, tuple(params)))
        docs_by_sha = {row["sha256"]: row for row in doc_rows}
        file_rows = cast(
            list[FileRow],
            await self.db.fetchall(
                f"SELECT * FROM files WHERE sha256 IN ({placeholders}) ORDER BY id DESC",
                tuple(sha256s),
            ),
        )

        # group files by sha256
        files_by_doc: dict[str, list[FileRow]] = {}
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
            doc = docs_by_sha.get(sha)
            if doc is None:
                continue
            all_files = files_by_doc.get(sha, [])
            snippet = strip_sep(row.get("snippet", ""))
            result_tier = row["tier"]

            results.append(
                {
                    "sha256": sha,
                    "short_id": doc["short_id"],
                    "title": row.get("title") or doc.get("title"),
                    "author": row.get("author") or doc.get("author"),
                    "size_bytes": doc["size_bytes"],
                    "page_count": doc.get("page_count"),
                    "tier": result_tier,
                    "snippet": snippet,
                    "files": [
                        {
                            "path": file["path"],
                            "filename": file["filename"],
                            "ext": file["ext"],
                            "status": file["status"],
                        }
                        for file in all_files
                    ],
                }
            )

        total = len(results)
        return results[offset : offset + limit], total

    # ── filename search ─────────────────────────────────────────

    async def search_filenames(
        self, query: str, ext: str | None = None, limit: int = 50, *, refresh_file: FilenamePathProbe | None = None
    ) -> tuple[list[FilenameSearchResultRow], int]:
        """Search filenames only. Returns (results, total_count)."""
        rows = await self.fts.search_filenames(query, limit=limit)
        if not rows:
            return [], 0

        file_ids = [r["file_id"] for r in rows]
        placeholders = ",".join("?" * len(file_ids))
        sql = (
            f"SELECT f.*, d.title, d.page_count "
            f"FROM files f LEFT JOIN docs d ON f.sha256 = d.sha256 "
            f"WHERE f.id IN ({placeholders})"
        )
        params = [*file_ids]
        if refresh_file is None:
            sql += " AND f.status = ?"
            params.append(FILE_STATUS_ACTIVE)
        if ext:
            sql += " AND f.ext = ?"
            params.append(ext.lower().lstrip("."))
        file_rows = cast(list[FilenameSearchFileRow], await self.db.fetchall(sql, tuple(params)))
        if refresh_file is not None:
            file_rows = await self._filter_probe_stale_filename_rows(file_rows, refresh_file)

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
                    "page_count": fr.get("page_count"),
                    "tier": "",
                    "snippet": strip_sep(row.get("snippet", "")),
                    "paths": [fr["path"]],
                }
            )

        total = len(results)
        return results, total

    async def _filter_probe_stale_filename_rows(
        self,
        file_rows: list[FilenameSearchFileRow],
        refresh_file: FilenamePathProbe,
    ) -> list[FilenameSearchFileRow]:
        current_rows: list[FilenameSearchFileRow] = []
        for file_row in file_rows:
            refreshed = await refresh_file(file_row["path"])
            if refreshed.status in {"missing", "deleted", "unreachable"}:
                continue
            if refreshed.file is not None and refreshed.file.status != FILE_STATUS_ACTIVE:
                continue
            current_row = cast(
                FilenameSearchFileRow | None,
                await self.db.fetchone(
                    "SELECT f.*, d.title, d.page_count "
                    "FROM files f LEFT JOIN docs d ON f.sha256 = d.sha256 "
                    "WHERE f.id = ? AND f.status = ?",
                    (file_row["id"], FILE_STATUS_ACTIVE),
                ),
            )
            if current_row is not None:
                current_rows.append(current_row)
        return current_rows


def _matches_tier(value: object, *, tier: Tier | None, min_tier: Tier | None) -> bool:
    if value is None:
        return tier is None and min_tier is None
    if not isinstance(value, str) or value not in TIER_ORDER:
        return False
    if tier is not None and value != tier:
        return False
    if min_tier is not None and TIER_ORDER[value] < TIER_ORDER[min_tier]:
        return False
    return True
