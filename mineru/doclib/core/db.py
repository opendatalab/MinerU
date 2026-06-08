from __future__ import annotations

import os
import time

import aiosqlite

from ..config import SQLiteConfig

SCHEMA_VERSION = 1

CREATE_TABLES_SQL = [
    # ── files ──────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS files (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        path            TEXT    NOT NULL UNIQUE,
        filename        TEXT    NOT NULL,
        ext             TEXT    NOT NULL,
        size_bytes      INTEGER NOT NULL,
        mtime_ms        INTEGER NOT NULL,
        birthtime_ms    INTEGER,
        sha256          TEXT    REFERENCES docs(sha256),
        watch_id        INTEGER REFERENCES watch_targets(id),
        scan_status     TEXT    NOT NULL DEFAULT 'active',
        locked_at       INTEGER,
        error_code      TEXT,
        error_msg       TEXT,
        deleted_at      INTEGER,
        first_seen_at   INTEGER NOT NULL,
        updated_at      INTEGER NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_files_sha256 ON files(sha256);",
    "CREATE INDEX IF NOT EXISTS idx_files_watch_id ON files(watch_id);",
    "CREATE INDEX IF NOT EXISTS idx_files_scan_status ON files(scan_status);",
    # ── docs ───────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS docs (
        sha256          TEXT    PRIMARY KEY,
        size_bytes      INTEGER NOT NULL,
        mime_type       TEXT,
        page_count      INTEGER,
        lang            TEXT,
        title           TEXT,
        author          TEXT,
        subject         TEXT,
        keywords        TEXT,
        is_encrypted    INTEGER NOT NULL DEFAULT 0,
        is_scanned      INTEGER NOT NULL DEFAULT 0,
        meta_tier       TEXT,
        first_seen_at   INTEGER NOT NULL,
        updated_at      INTEGER NOT NULL
    );
    """,
    # ── parses ─────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS parses (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        sha256      TEXT    NOT NULL REFERENCES docs(sha256),
        tier        TEXT    NOT NULL,
        pages       TEXT    NOT NULL,
        status      TEXT    NOT NULL DEFAULT 'pending',
        priority    INTEGER NOT NULL DEFAULT 0,
        locked_at   INTEGER,
        error_code  TEXT,
        error_msg   TEXT,
        privacy     TEXT    NOT NULL DEFAULT 'local',
        remote_url  TEXT,
        via         TEXT,
        output_path TEXT,
        done_at     INTEGER,
        created_at  INTEGER NOT NULL,
        updated_at  INTEGER NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_parses_status ON parses(status, priority DESC, created_at ASC);",
    "CREATE INDEX IF NOT EXISTS idx_parses_doc ON parses(sha256, tier);",
    # ── fts_contents ───────────────────────────────────────────────
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_contents USING fts5(
        sha256 UNINDEXED,
        tier UNINDEXED,
        text,
        title,
        author,
        filename,
        tokenize='unicode61'
    );
    """,
    # ── fts_filenames ──────────────────────────────────────────────
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_filenames USING fts5(
        file_id UNINDEXED,
        filename,
        ext,
        tokenize='unicode61'
    );
    """,
    # ── watch_targets ──────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS watch_targets (
        id              INTEGER PRIMARY KEY,
        path            TEXT    NOT NULL UNIQUE,
        label           TEXT,
        removable       INTEGER NOT NULL DEFAULT 0,
        enabled         INTEGER NOT NULL DEFAULT 1,
        recursive       INTEGER NOT NULL DEFAULT 0,
        watch_status    TEXT    NOT NULL DEFAULT 'active',
        unreachable_at  INTEGER,
        last_scan_at    INTEGER,
        last_scan_files INTEGER DEFAULT 0
    );
    """,
    # ── rules ──────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS rules (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        name            TEXT,
        rule_type       TEXT    NOT NULL,
        pattern         TEXT    NOT NULL,
        tier            TEXT,
        pages           TEXT,
        remote          INTEGER NOT NULL DEFAULT 0,
        enabled         INTEGER NOT NULL DEFAULT 1,
        priority        INTEGER NOT NULL DEFAULT 0,
        hit_count       INTEGER NOT NULL DEFAULT 0
    );
    """,
    # ── config ─────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS config (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """,
    # ── _migrations ────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS _migrations (
        version     INTEGER PRIMARY KEY,
        applied_at  INTEGER NOT NULL,
        description TEXT
    );
    """,
]

DEFAULT_CONFIG = {
    "data_dir": "~/MinerU",
    "default_tier": "flash",
    "scan_interval_sec": "300",
    "ingest_lock_timeout_sec": "60",
    "parse_lock_timeout_sec": "1800",
    "device_check_interval_sec": "5",
    "parse_server.local.mode": "disabled",
    "parse_server.local.managed_tier": "standard",
    "parse_server.remote.url": "https://mineru.net/api",
}

DEFAULT_EXCLUDE_RULES = [
    ("系统-*/Library/*", "*/Library/*"),
    ("系统-*/.git/*", "*/.git/*"),
    ("系统-*/node_modules/*", "*/node_modules/*"),
    ("系统-*/vendor/*", "*/vendor/*"),
    ("系统-*/go/pkg/*", "*/go/pkg/*"),
    ("系统-*/__pycache__/*", "*/__pycache__/*"),
    ("系统-*/.venv/*", "*/.venv/*"),
    ("系统-*/miniconda3/*", "*/miniconda3/*"),
    ("系统-*/.nvm/*", "*/.nvm/*"),
    ("系统-*/.docker/*", "*/.docker/*"),
    ("系统-*/target/*", "*/target/*"),
    ("系统-*/dist/*", "*/dist/*"),
    ("系统-*/build/*", "*/build/*"),
]


class DatabaseManager:
    """Per-operation connection manager — no pooling, no shared state."""

    def __init__(self, db_path: str, sqlite_cfg: SQLiteConfig | None = None) -> None:
        self.db_path = db_path
        self.sqlite_cfg = sqlite_cfg  # SQLiteConfig or None

    # ── lifecycle ──────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Create tables, apply migrations, seed default data.  Idempotent."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = await aiosqlite.connect(self.db_path)
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")

        if self.sqlite_cfg:
            await conn.execute(f"PRAGMA synchronous={self.sqlite_cfg.synchronous}")
            await conn.execute(f"PRAGMA mmap_size={self.sqlite_cfg.mmap_size}")
            await conn.execute(f"PRAGMA cache_size={self.sqlite_cfg.cache_size}")
            await conn.execute(f"PRAGMA temp_store={self.sqlite_cfg.temp_store}")
            await conn.execute(
                f"PRAGMA wal_autocheckpoint={self.sqlite_cfg.wal_autocheckpoint}"
            )
            await conn.execute(
                f"PRAGMA journal_size_limit={self.sqlite_cfg.journal_size_limit}"
            )
        else:
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA mmap_size=268435456")
            await conn.execute("PRAGMA cache_size=-20000")
            await conn.execute("PRAGMA temp_store=MEMORY")
            await conn.execute("PRAGMA wal_autocheckpoint=1000")
            await conn.execute("PRAGMA journal_size_limit=33554432")

        for sql in CREATE_TABLES_SQL:
            await conn.execute(sql)

        # ── migrations ──────────────────────────────────────────
        for version in range(1, SCHEMA_VERSION + 1):
            cursor = await conn.execute(
                "SELECT 1 FROM _migrations WHERE version=?", (version,)
            )
            if await cursor.fetchone() is None:
                await _apply_migration(conn, version)
                await conn.execute(
                    "INSERT INTO _migrations (version, applied_at) VALUES (?, ?)",
                    (version, _now_ms()),
                )

        # ── seed data ───────────────────────────────────────────
        for key, value in DEFAULT_CONFIG.items():
            await conn.execute(
                "INSERT OR IGNORE INTO config (key, value) VALUES (?, ?)", (key, value)
            )

        for name, pattern in DEFAULT_EXCLUDE_RULES:
            await conn.execute(
                "INSERT OR IGNORE INTO rules (name, rule_type, pattern, priority) "
                "SELECT ?, 'exclude', ?, 0 "
                "WHERE NOT EXISTS (SELECT 1 FROM rules WHERE rule_type='exclude' AND pattern=?)",
                (name, pattern, pattern),
            )

        await conn.commit()
        await conn.close()

    async def _connect(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self.db_path)
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = aiosqlite.Row
        return conn

    async def close(self) -> None:
        pass  # no cached connection

    # ── query helpers ──────────────────────────────────────────────

    async def execute(self, sql: str, params: tuple | None = None) -> aiosqlite.Cursor:
        conn = await self._connect()
        cursor = await conn.execute(sql, params or ())
        await conn.commit()
        await conn.close()
        return cursor

    async def execute_insert(self, sql: str, params: tuple | None = None) -> int:
        conn = await self._connect()
        cursor = await conn.execute(sql, params or ())
        await conn.commit()
        lastid = cursor.lastrowid
        await conn.close()
        return lastid

    async def fetchone(self, sql: str, params: tuple | None = None) -> dict | None:
        conn = await self._connect()
        cursor = await conn.execute(sql, params or ())
        row = await cursor.fetchone()
        result = dict(row) if row else None
        await conn.commit()
        await conn.close()
        return result

    async def fetchall(self, sql: str, params: tuple | None = None) -> list[dict]:
        conn = await self._connect()
        cursor = await conn.execute(sql, params or ())
        rows = await cursor.fetchall()
        result = [dict(row) for row in rows]
        await conn.commit()
        await conn.close()
        return result

    async def commit(self) -> None:
        pass

    async def execute_atomic(self, statements: list[tuple[str, tuple]]) -> int:
        """Execute multiple statements in a single transaction."""
        conn = await self._connect()
        try:
            for sql, params in statements:
                await conn.execute(sql, params)
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        finally:
            await conn.close()
        return len(statements)


# ── helpers ────────────────────────────────────────────────────────


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _apply_migration(conn: aiosqlite.Connection, version: int) -> None:
    """Apply a single schema migration.  Extend here as schema evolves."""
    # version 1: initial schema — already created by CREATE_TABLES_SQL
    pass
