from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mineru.doclib.core import db as db_module
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.server import _doc_info, _parse_info, _parsing_rule_info
from mineru.doclib.services.config_svc import ConfigService


def test_fresh_database_records_all_schema_migrations(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))

        await db.initialize()

        rows = await db.fetchall("SELECT version FROM _migrations ORDER BY version")
        assert rows == [{"version": 1}, {"version": 2}]

    asyncio.run(_run())


def test_v2_migration_invalidates_legacy_cached_data_and_maps_user_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        monkeypatch.setattr(db_module, "SCHEMA_VERSION", 1)
        await db.initialize()

        now = 1
        legacy_sha = "a" * 64
        current_sha = "b" * 64
        await db.execute(
            "INSERT INTO docs (sha256, short_id, size_bytes, meta_tier, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (legacy_sha, "aaaaaaa", 1, "xhigh", now, now),
        )
        await db.execute(
            "INSERT INTO docs (sha256, short_id, size_bytes, meta_tier, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (current_sha, "bbbbbbb", 1, "standard", now, now),
        )
        for tier, status in (("medium", "done"), ("high", "pending"), ("xhigh", "done"), ("unknown", "done")):
            await db.execute(
                "INSERT INTO parses (sha256, tier, page_range, status, privacy, created_at, updated_at) "
                "VALUES (?, ?, '1', ?, 'local', ?, ?)",
                (legacy_sha, tier, status, now, now),
            )
        await db.execute(
            "INSERT INTO parses (sha256, tier, page_range, status, privacy, created_at, updated_at) "
            "VALUES (?, 'standard', '1', 'done', 'local', ?, ?)",
            (current_sha, now, now),
        )
        for tier, sha256 in (("medium", legacy_sha), ("unknown", legacy_sha), ("standard", current_sha)):
            await db.execute(
                "INSERT INTO fts_contents (sha256, tier, text, title, author) VALUES (?, ?, 'text', '', '')",
                (sha256, tier),
            )
        for index, tier in enumerate(("medium", "high", "xhigh", "unknown", "standard"), start=1):
            await db.execute(
                "INSERT INTO parsing_rules (id, pattern, tier, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (index, f"*.{index}.pdf", tier, now, now),
            )
        await db.execute(
            "INSERT INTO config (key, value) VALUES ('parse_server.local.managed_tier', 'xhigh')"
        )

        monkeypatch.setattr(db_module, "SCHEMA_VERSION", 2)
        await db.initialize()

        assert await db.fetchall("SELECT tier FROM parses ORDER BY id") == [{"tier": "standard"}]
        assert await db.fetchall("SELECT tier FROM fts_contents ORDER BY rowid") == [{"tier": "standard"}]
        assert await db.fetchall("SELECT short_id, meta_tier FROM docs ORDER BY short_id") == [
            {"short_id": "aaaaaaa", "meta_tier": None},
            {"short_id": "bbbbbbb", "meta_tier": "standard"},
        ]
        assert await db.fetchall("SELECT id, tier FROM parsing_rules ORDER BY id") == [
            {"id": 1, "tier": "basic"},
            {"id": 2, "tier": "standard"},
            {"id": 3, "tier": "advanced"},
            {"id": 4, "tier": None},
            {"id": 5, "tier": "standard"},
        ]
        assert await db.fetchone("SELECT value FROM config WHERE key='parse_server.local.managed_tier'") is None
        assert await ConfigService(db).get("parse_server.local.managed_tier") == "standard"
        assert await db.fetchall("SELECT version FROM _migrations ORDER BY version") == [
            {"version": 1},
            {"version": 2},
        ]

        doc_rows = await db.fetchall("SELECT * FROM docs ORDER BY short_id")
        parse_rows = await db.fetchall(
            "SELECT p.*, d.short_id FROM parses p JOIN docs d ON d.sha256=p.sha256 ORDER BY p.id"
        )
        rule_rows = await db.fetchall("SELECT * FROM parsing_rules ORDER BY id")
        assert [_doc_info(row).meta_tier for row in doc_rows] == [None, "standard"]
        assert [_parse_info(row).tier for row in parse_rows] == ["standard"]
        assert [_parsing_rule_info(row).tier for row in rule_rows] == [
            "basic",
            "standard",
            "advanced",
            None,
            "standard",
        ]

        snapshot = {
            "parses": await db.fetchall("SELECT * FROM parses ORDER BY id"),
            "fts": await db.fetchall("SELECT * FROM fts_contents ORDER BY rowid"),
            "docs": await db.fetchall("SELECT * FROM docs ORDER BY short_id"),
            "rules": await db.fetchall("SELECT * FROM parsing_rules ORDER BY id"),
            "config": await db.fetchall("SELECT * FROM config ORDER BY key"),
        }
        await db.initialize()
        assert await db.fetchall("SELECT * FROM parses ORDER BY id") == snapshot["parses"]
        assert await db.fetchall("SELECT * FROM fts_contents ORDER BY rowid") == snapshot["fts"]
        assert await db.fetchall("SELECT * FROM docs ORDER BY short_id") == snapshot["docs"]
        assert await db.fetchall("SELECT * FROM parsing_rules ORDER BY id") == snapshot["rules"]
        assert await db.fetchall("SELECT * FROM config ORDER BY key") == snapshot["config"]

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("legacy_tier", "expected_override", "expected_tier"),
    (
        ("medium", "basic", "basic"),
        ("high", None, "standard"),
        ("xhigh", None, "standard"),
        ("unknown", None, "standard"),
    ),
)
def test_v2_migration_normalizes_managed_server_tier(
    legacy_tier: str,
    expected_override: str | None,
    expected_tier: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        monkeypatch.setattr(db_module, "SCHEMA_VERSION", 1)
        await db.initialize()
        await db.execute(
            "INSERT INTO config (key, value) VALUES ('parse_server.local.managed_tier', ?)",
            (legacy_tier,),
        )

        monkeypatch.setattr(db_module, "SCHEMA_VERSION", 2)
        await db.initialize()

        row = await db.fetchone("SELECT value FROM config WHERE key='parse_server.local.managed_tier'")
        assert (row["value"] if row else None) == expected_override
        assert await ConfigService(db).get("parse_server.local.managed_tier") == expected_tier

    asyncio.run(_run())
