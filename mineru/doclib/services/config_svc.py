"""Configuration service: KV config, watch targets, rules."""

from __future__ import annotations

import fnmatch
import os
import time

import fnvhash

from ..core.db import DatabaseManager


class ConfigService:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    # ── KV config ───────────────────────────────────────────────

    async def get(self, key: str, default: str | None = None) -> str | None:
        row = await self.db.fetchone("SELECT value FROM config WHERE key=?", (key,))
        return row["value"] if row else default

    async def set(self, key: str, value: str) -> None:
        await self.db.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, value)
        )

    async def get_all(self) -> dict[str, str]:
        rows = await self.db.fetchall("SELECT key, value FROM config")
        return {r["key"]: r["value"] for r in rows}

    # ── watch targets ───────────────────────────────────────────

    async def add_watch(
        self, path: str, removable: bool = False, label: str | None = None
    ) -> dict:
        if not os.path.isabs(path):
            raise ValueError(f"Path must be absolute: {path}")
        if os.path.normpath(path) != path:
            raise ValueError(f"Path must be normalized: {path}")

        wid = fnvhash.fnv1a_64(path.encode())
        if wid >= 2**63:
            wid -= 2**64

        await self.db.execute(
            "INSERT OR REPLACE INTO watch_targets (id, path, label, removable) "
            "VALUES (?, ?, ?, ?)",
            (wid, path, label, int(removable)),
        )
        return await self.db.fetchone("SELECT * FROM watch_targets WHERE id=?", (wid,))

    async def list_watches(self) -> list[dict]:
        return await self.db.fetchall(
            "SELECT * FROM watch_targets WHERE enabled=1 ORDER BY path"
        )

    async def remove_watch(self, path: str) -> None:
        await self.db.execute("DELETE FROM watch_targets WHERE path=?", (path,))

    async def get_watches_by_status(self, watch_status: str) -> list[dict]:
        return await self.db.fetchall(
            "SELECT * FROM watch_targets WHERE watch_status=? AND enabled=1",
            (watch_status,),
        )

    async def update_watch_status(self, watch_id: int, status: str) -> None:
        await self.db.execute(
            "UPDATE watch_targets SET watch_status=? WHERE id=?", (status, watch_id)
        )

    async def update_watch_scan_stats(self, watch_id: int, file_count: int) -> None:
        now = int(time.time() * 1000)
        await self.db.execute(
            "UPDATE watch_targets SET last_scan_at=?, last_scan_files=? WHERE id=?",
            (now, file_count, watch_id),
        )

    # ── rules ───────────────────────────────────────────────────

    async def add_rule(
        self,
        name: str,
        rule_type: str,
        pattern: str,
        tier: str | None = None,
        pages: str | None = None,
        remote: bool = False,
        priority: int = 0,
    ) -> int:
        return await self.db.execute_insert(
            "INSERT INTO rules (name, rule_type, pattern, tier, pages, remote, priority) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, rule_type, pattern, tier, pages, int(remote), priority),
        )

    async def list_rules(self, rule_type: str | None = None) -> list[dict]:
        if rule_type:
            return await self.db.fetchall(
                "SELECT * FROM rules WHERE rule_type=? AND enabled=1 ORDER BY priority DESC",
                (rule_type,),
            )
        return await self.db.fetchall(
            "SELECT * FROM rules WHERE enabled=1 ORDER BY rule_type, priority DESC"
        )

    async def remove_rule(self, rule_id: int) -> None:
        await self.db.execute("DELETE FROM rules WHERE id=?", (rule_id,))

    async def match_rules(self, file_path: str, rule_type: str) -> list[dict]:
        rules = await self.db.fetchall(
            "SELECT * FROM rules WHERE rule_type=? AND enabled=1 ORDER BY priority DESC",
            (rule_type,),
        )
        matched: list[dict] = []
        for rule in rules:
            if fnmatch.fnmatch(file_path, rule["pattern"]):
                matched.append(rule)
                await self.db.execute(
                    "UPDATE rules SET hit_count=hit_count+1 WHERE id=?", (rule["id"],)
                )
        if matched:
            await self.db.commit()
        return matched

    async def is_path_excluded(self, file_path: str) -> bool:
        matched = await self.match_rules(file_path, "exclude")
        return len(matched) > 0
