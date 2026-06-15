from __future__ import annotations

import os
import time
from pathlib import Path

import aiosqlite

from ...config import SQLiteConfig
from ..config_defaults import CONFIG_DEFAULTS

SCHEMA_VERSION = 1
_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

_CREATE_TABLES_SQL: list[str] = []  # filled by _load_init_sql()


def _load_init_sql() -> list[str]:
    """Load the initial-schema SQL from the migration file so that
    ``initialize()`` is self-contained for fresh databases."""
    init_path = _MIGRATIONS_DIR / "001_init.sql"
    if not init_path.is_file():
        return []
    sql = init_path.read_text(encoding="utf-8")
    return [s.strip() for s in sql.split(";") if s.strip()]


_CREATE_TABLES_SQL = _load_init_sql()

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

        for sql in _CREATE_TABLES_SQL:
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
        # Config defaults are code-backed; the DB config table stores overrides only.
        # Clean legacy seeded rows whose values exactly match current defaults.
        for key, value in CONFIG_DEFAULTS.items():
            await conn.execute("DELETE FROM config WHERE key=? AND value=?", (key, value))

        now = _now_ms()
        for name, pattern in DEFAULT_EXCLUDE_RULES:
            await conn.execute(
                "INSERT INTO exclude_rules (name, pattern, priority, created_at, updated_at) "
                "SELECT ?, ?, 0, ?, ? "
                "WHERE NOT EXISTS (SELECT 1 FROM exclude_rules WHERE pattern=?)",
                (name, pattern, now, now, pattern),
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
        if lastid is None:
            raise RuntimeError("SQLite insert did not return a row id.")
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
    """Apply a single schema migration from its SQL file."""
    candidates = sorted(_MIGRATIONS_DIR.glob(f"{version:03d}_*.sql"))
    if not candidates:
        raise FileNotFoundError(f"No migration file found for version {version}")
    for p in candidates:
        raw = p.read_text(encoding="utf-8")
        # strip comment lines, then split into statements
        lines = [line for line in raw.splitlines() if not line.strip().startswith("--")]
        sql = "\n".join(lines)
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            try:
                await conn.execute(stmt)
            except Exception as exc:
                if "duplicate column name" in str(exc):
                    continue
                raise
