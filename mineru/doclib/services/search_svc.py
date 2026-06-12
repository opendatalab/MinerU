"""Search service: content search and filename search."""

from __future__ import annotations

from ..core.db import DatabaseManager
from ..core.fts import FTSManager, strip_sep
from ..types import SCAN_STATUS_ACTIVE


class SearchService:
    def __init__(self, db: DatabaseManager, fts: FTSManager) -> None:
        self.db = db
        self.fts = fts

    # ── content search ──────────────────────────────────────────

    async def search(
        self,
        query: str,
        file_type: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """Full-text search. Returns (results, total_count)."""
        rows = await self.fts.search(query, limit=200)
        if not rows:
            return [], 0

        # dedup by sha256
        sha256s = list({r["sha256"] for r in rows})

        # join with docs + files
        placeholders = ",".join("?" * len(sha256s))
        sql = (
            "SELECT f.*, d.title, d.author, d.page_count "
            "FROM files f JOIN docs d ON f.sha256 = d.sha256 "
            f"WHERE f.sha256 IN ({placeholders}) AND f.scan_status = ?"
        )
        params = [*sha256s, SCAN_STATUS_ACTIVE]

        if file_type:
            sql += " AND f.ext = ?"
            params.append(file_type.lower())

        file_rows = await self.db.fetchall(sql, params)

        # group files by sha256
        files_by_doc: dict[str, list[dict]] = {}
        for fr in file_rows:
            sha = fr["sha256"]
            if sha not in files_by_doc:
                files_by_doc[sha] = []
            files_by_doc[sha].append(fr)

        # build results
        results: list[dict] = []
        for row in rows:
            sha = row["sha256"]
            files = files_by_doc.get(sha, [])
            if not files:
                continue

            fts_file = files[0]
            snippet = strip_sep(row.get("snippet", ""))
            tier = row.get("tier") or ""

            results.append(
                {
                    "sha256": sha,
                    "title": row.get("title") or fts_file.get("title"),
                    "author": row.get("author") or fts_file.get("author"),
                    "filename": row.get("filename") or fts_file.get("filename"),
                    "ext": fts_file.get("ext", ""),
                    "size_bytes": fts_file.get("size_bytes", 0),
                    "tier": tier,
                    "snippet": snippet,
                    "paths": [f["path"] for f in files],
                }
            )

        total = len(results)
        return results[offset : offset + limit], total

    # ── filename search ─────────────────────────────────────────

    async def search_filenames(
        self, query: str, limit: int = 50
    ) -> tuple[list[dict], int]:
        """Search filenames only. Returns (results, total_count)."""
        rows = await self.fts.search_filenames(query, limit=limit)
        if not rows:
            return [], 0

        file_ids = [r["file_id"] for r in rows]
        placeholders = ",".join("?" * len(file_ids))
        file_rows = await self.db.fetchall(
            f"SELECT f.*, d.title "
            f"FROM files f LEFT JOIN docs d ON f.sha256 = d.sha256 "
            f"WHERE f.id IN ({placeholders}) AND f.scan_status = ?",
            [*file_ids, SCAN_STATUS_ACTIVE],
        )

        files_by_id = {fr["id"]: fr for fr in file_rows}

        results: list[dict] = []
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
