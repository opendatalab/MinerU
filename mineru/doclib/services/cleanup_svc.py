"""Cleanup service: orphan docs, deleted files, temp files."""

from __future__ import annotations

import os
import shutil
import time

from ..core.db import DatabaseManager
from ..types import SCAN_STATUS_ACTIVE, SCAN_STATUS_DELETED


class CleanupService:
    def __init__(self, db: DatabaseManager, data_dir: str) -> None:
        self.db = db
        self.data_dir = os.path.expanduser(data_dir)

    # ── orphan docs ─────────────────────────────────────────────

    async def find_orphan_docs(self) -> list[dict]:
        return await self.db.fetchall(
            "SELECT d.* FROM docs d WHERE NOT EXISTS ("
            "  SELECT 1 FROM files f WHERE f.sha256 = d.sha256 AND f.scan_status = ?"
            ")",
            (SCAN_STATUS_ACTIVE,),
        )

    async def cleanup_orphans(self, dry_run: bool = True) -> int:
        orphans = await self.find_orphan_docs()
        if dry_run:
            return len(orphans)

        count = 0
        for doc in orphans:
            sha256 = doc["sha256"]
            await self.db.execute("DELETE FROM fts_contents WHERE sha256=?", (sha256,))
            await self.db.execute("DELETE FROM parses WHERE sha256=?", (sha256,))
            await self.db.execute("DELETE FROM docs WHERE sha256=?", (sha256,))
            # remove parsed output dir
            parsed_dir = os.path.join(self.data_dir, "parsed", sha256[:2], sha256)
            if os.path.isdir(parsed_dir):
                shutil.rmtree(parsed_dir, ignore_errors=True)
            count += 1
        await self.db.commit()
        return count

    # ── deleted files ───────────────────────────────────────────

    async def cleanup_deleted(
        self, older_than_days: int = 30, dry_run: bool = True
    ) -> int:
        row = await self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM files WHERE scan_status=? "
            "AND deleted_at < ?",
            (SCAN_STATUS_DELETED, _days_ago_ms(older_than_days)),
        )
        count = row["cnt"] if row else 0

        if not dry_run and count > 0:
            await self.db.execute(
                "DELETE FROM files WHERE scan_status=? AND deleted_at < ?",
                (SCAN_STATUS_DELETED, _days_ago_ms(older_than_days)),
            )
            await self.db.commit()
            await self.cleanup_orphans(dry_run=False)

        return count

    # ── temp files ──────────────────────────────────────────────

    async def cleanup_temp_files(self, older_than_days: int = 7) -> int:
        """Remove process temp files (e.g. incomplete parse artifacts)."""
        temp_dir = os.path.join(self.data_dir, "temp")
        if not os.path.isdir(temp_dir):
            return 0

        threshold = _days_ago_ms(older_than_days) / 1000.0
        count = 0
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                fp = os.path.join(root, name)
                try:
                    if os.path.getmtime(fp) < threshold:
                        os.remove(fp)
                        count += 1
                except OSError:
                    pass
            for name in dirs:
                dp = os.path.join(root, name)
                try:
                    if not os.listdir(dp):
                        os.rmdir(dp)
                except OSError:
                    pass
        return count


def _days_ago_ms(days: int) -> int:
    return int((time.time() - days * 86400) * 1000)
