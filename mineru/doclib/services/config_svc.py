"""Configuration service: KV config, watch targets, rules."""

from __future__ import annotations

import fnmatch
import os
import time
from typing import cast

import fnvhash

from ...errors import InvalidRequestError
from ...types import Tier
from ..config_defaults import CONFIG_DEFAULTS, CONFIG_SOURCE_DEFAULT, CONFIG_SOURCE_OVERRIDE, ConfigSource
from ..core.db import DatabaseManager
from ..rows import ConfigRow, RuleRow, WatchTargetRow
from ..types import (
    RULE_TYPE_EXCLUDE,
    RULE_TYPE_PARSING_RULE,
    FILE_STATUS_ACTIVE,
    FILE_STATUS_DELETED,
    FILE_STATUS_UNREACHABLE,
    WATCH_STATUS_UNREACHABLE,
    RuleType,
    WatchStatus,
)


class ConfigService:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    # ── KV config ───────────────────────────────────────────────

    async def get(self, key: str, default: str | None = None) -> str | None:
        if key in CONFIG_DEFAULTS and default is None:
            default = CONFIG_DEFAULTS[key]
        row = cast(ConfigRow | None, await self.db.fetchone("SELECT value FROM config WHERE key=?", (key,)))
        return row["value"] if row else default

    async def set(self, key: str, value: str) -> None:
        self._validate_config_key(key)
        if CONFIG_DEFAULTS[key] == value:
            await self.unset(key)
            return
        await self.db.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, value)
        )

    async def unset(self, key: str) -> bool:
        self._validate_config_key(key)
        cursor = await self.db.execute("DELETE FROM config WHERE key=?", (key,))
        return cursor.rowcount > 0

    async def get_all(self) -> dict[str, str]:
        config, _sources = await self.get_all_with_sources()
        return config

    async def get_source(self, key: str) -> ConfigSource:
        self._validate_config_key(key)
        row = cast(ConfigRow | None, await self.db.fetchone("SELECT value FROM config WHERE key=?", (key,)))
        return CONFIG_SOURCE_OVERRIDE if row else CONFIG_SOURCE_DEFAULT

    async def get_all_with_sources(self) -> tuple[dict[str, str], dict[str, ConfigSource]]:
        rows = cast(list[ConfigRow], await self.db.fetchall("SELECT key, value FROM config"))
        overrides = {r["key"]: r["value"] for r in rows}
        config = dict(CONFIG_DEFAULTS)
        config.update(overrides)
        sources: dict[str, ConfigSource] = {key: CONFIG_SOURCE_DEFAULT for key in CONFIG_DEFAULTS}
        for key in overrides:
            if key in CONFIG_DEFAULTS:
                sources[key] = CONFIG_SOURCE_OVERRIDE
        return config, sources

    def _validate_config_key(self, key: str) -> None:
        if key not in CONFIG_DEFAULTS:
            raise InvalidRequestError("invalid_config_key", f"Unknown config key: {key}", "key")

    # ── watch targets ───────────────────────────────────────────

    async def add_watch(
        self, path: str, removable: bool = False, label: str | None = None
    ) -> WatchTargetRow:
        if not os.path.isabs(path):
            raise ValueError(f"Path must be absolute: {path}")
        if os.path.normpath(path) != path:
            raise ValueError(f"Path must be normalized: {path}")

        wid = fnvhash.fnv1a_64(path.encode())
        if wid >= 2**63:
            wid -= 2**64

        now = int(time.time() * 1000)
        await self.db.execute(
            "INSERT INTO watches (id, path, label, removable, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET "
            "label=excluded.label, removable=excluded.removable, updated_at=excluded.updated_at",
            (wid, path, label, int(removable), now, now),
        )
        row = cast(WatchTargetRow | None, await self.db.fetchone("SELECT * FROM watches WHERE id=?", (wid,)))
        if row is None:
            raise RuntimeError(f"Watch target {wid} not found after insert.")
        return row

    async def list_watches(self) -> list[WatchTargetRow]:
        return cast(
            list[WatchTargetRow],
            await self.db.fetchall("SELECT * FROM watches WHERE enabled=1 ORDER BY path"),
        )

    async def remove_watch(self, path: str) -> None:
        watch = cast(WatchTargetRow | None, await self.db.fetchone("SELECT * FROM watches WHERE path=?", (path,)))
        if watch is None:
            return
        await self.remove_watch_by_id(watch["id"])

    async def remove_watch_by_id(self, watch_id: int) -> None:
        watch = cast(WatchTargetRow | None, await self.db.fetchone("SELECT * FROM watches WHERE id=?", (watch_id,)))
        if watch is None:
            return
        now = int(time.time() * 1000)
        await self.db.execute_atomic(
            [
                (
                    "UPDATE files SET watch_id=NULL, updated_at=? "
                    "WHERE watch_id=? AND status IN (?, ?)",
                    (now, watch["id"], FILE_STATUS_ACTIVE, FILE_STATUS_DELETED),
                ),
                (
                    "UPDATE files SET watch_id=NULL, status=?, deleted_at=?, updated_at=? "
                    "WHERE watch_id=? AND status=?",
                    (FILE_STATUS_DELETED, now, now, watch["id"], FILE_STATUS_UNREACHABLE),
                ),
                ("UPDATE scans SET watch_id=NULL, updated_at=? WHERE watch_id=?", (now, watch["id"])),
                ("DELETE FROM watches WHERE id=?", (watch["id"],)),
            ]
        )

    async def get_watches_by_status(self, status: WatchStatus) -> list[WatchTargetRow]:
        return cast(
            list[WatchTargetRow],
            await self.db.fetchall(
                "SELECT * FROM watches WHERE status=? AND enabled=1",
                (status,),
            ),
        )

    async def update_watch_status(self, watch_id: int, status: WatchStatus) -> None:
        now = int(time.time() * 1000)
        unreachable_at = now if status == WATCH_STATUS_UNREACHABLE else None
        await self.db.execute(
            "UPDATE watches SET status=?, unreachable_at=?, updated_at=? WHERE id=?",
            (status, unreachable_at, now, watch_id),
        )

    async def update_watch_scan_stats(self, watch_id: int, file_count: int) -> None:
        now = int(time.time() * 1000)
        await self.db.execute(
            "UPDATE watches SET last_scan_at=?, last_scan_files=?, updated_at=? WHERE id=?",
            (now, file_count, now, watch_id),
        )

    # ── rules ───────────────────────────────────────────────────

    async def add_rule(
        self,
        name: str,
        rule_type: RuleType,
        pattern: str,
        tier: Tier | None = None,
        pages: str | None = None,
        remote: bool = False,
        priority: int = 0,
    ) -> int:
        now = int(time.time() * 1000)
        if rule_type == RULE_TYPE_EXCLUDE:
            return await self.db.execute_insert(
                "INSERT INTO exclude_rules (name, pattern, priority, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, pattern, priority, now, now),
            )
        if rule_type == RULE_TYPE_PARSING_RULE:
            return await self.db.execute_insert(
                "INSERT INTO parsing_rules (name, pattern, tier, pages, remote, priority, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (name, pattern, tier, pages, int(remote), priority, now, now),
            )
        raise ValueError(f"Unsupported rule type: {rule_type}")

    async def list_rules(self, rule_type: RuleType | None = None) -> list[RuleRow]:
        if rule_type == RULE_TYPE_EXCLUDE:
            return cast(
                list[RuleRow],
                await self.db.fetchall(
                    "SELECT id, name, ? AS rule_type, pattern, NULL AS tier, NULL AS pages, "
                    "0 AS remote, enabled, priority, hit_count, created_at, updated_at "
                    "FROM exclude_rules WHERE enabled=1 ORDER BY priority DESC",
                    (RULE_TYPE_EXCLUDE,),
                ),
            )
        if rule_type == RULE_TYPE_PARSING_RULE:
            return cast(
                list[RuleRow],
                await self.db.fetchall(
                    "SELECT id, name, ? AS rule_type, pattern, tier, pages, remote, "
                    "enabled, priority, hit_count, created_at, updated_at "
                    "FROM parsing_rules WHERE enabled=1 ORDER BY priority DESC",
                    (RULE_TYPE_PARSING_RULE,),
                ),
            )
        if rule_type is not None:
            raise ValueError(f"Unsupported rule type: {rule_type}")
        return cast(
            list[RuleRow],
            await self.db.fetchall(
                "SELECT id, name, ? AS rule_type, pattern, NULL AS tier, NULL AS pages, "
                "0 AS remote, enabled, priority, hit_count, created_at, updated_at "
                "FROM exclude_rules WHERE enabled=1 "
                "UNION ALL "
                "SELECT id, name, ? AS rule_type, pattern, tier, pages, remote, "
                "enabled, priority, hit_count, created_at, updated_at "
                "FROM parsing_rules WHERE enabled=1 "
                "ORDER BY rule_type, priority DESC",
                (RULE_TYPE_EXCLUDE, RULE_TYPE_PARSING_RULE),
            ),
        )

    async def remove_rule(self, rule_id: int, rule_type: RuleType | None = None) -> None:
        if rule_type == RULE_TYPE_EXCLUDE:
            await self.db.execute("DELETE FROM exclude_rules WHERE id=?", (rule_id,))
            return
        if rule_type == RULE_TYPE_PARSING_RULE:
            await self.db.execute("DELETE FROM parsing_rules WHERE id=?", (rule_id,))
            return
        if rule_type is not None:
            raise ValueError(f"Unsupported rule type: {rule_type}")
        await self.db.execute("DELETE FROM exclude_rules WHERE id=?", (rule_id,))
        await self.db.execute("DELETE FROM parsing_rules WHERE id=?", (rule_id,))

    async def match_rules(self, file_path: str, rule_type: RuleType) -> list[RuleRow]:
        rules = await self.list_rules(rule_type)
        table = _rule_table(rule_type)
        matched: list[RuleRow] = []
        for rule in rules:
            if fnmatch.fnmatch(file_path, rule["pattern"]):
                matched.append(rule)
                await self.db.execute(
                    f"UPDATE {table} SET hit_count=hit_count+1, updated_at=? WHERE id=?",
                    (int(time.time() * 1000), rule["id"]),
                )
        if matched:
            await self.db.commit()
        return matched

    async def is_path_excluded(self, file_path: str) -> bool:
        matched = await self.match_rules(file_path, RULE_TYPE_EXCLUDE)
        return len(matched) > 0


def _rule_table(rule_type: RuleType) -> str:
    if rule_type == RULE_TYPE_EXCLUDE:
        return "exclude_rules"
    if rule_type == RULE_TYPE_PARSING_RULE:
        return "parsing_rules"
    raise ValueError(f"Unsupported rule type: {rule_type}")
