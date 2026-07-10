from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

import aiosqlite

from ...config import SQLiteConfig
from ...errors import ServerBusyError
from ..config_defaults import CONFIG_DEFAULTS

SCHEMA_VERSION = 1
_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

_T = TypeVar("_T")
_CREATE_TABLES_SQL: list[str] = []  # filled by _load_init_sql()
_DEFAULT_BUSY_TIMEOUT_MS = 5000
_DEFAULT_LOCK_RETRY_ATTEMPTS = 3
_DEFAULT_LOCK_RETRY_BASE_DELAY_MS = 50


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
    """Per-operation connections with concurrent reads and process-local serialized writes."""

    def __init__(self, db_path: str, sqlite_cfg: SQLiteConfig | None = None) -> None:
        self.db_path = db_path
        self.sqlite_cfg = sqlite_cfg  # SQLiteConfig or None
        self._write_lock = asyncio.Lock()

    # ── lifecycle ──────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Create tables, apply migrations, seed default data.  Idempotent."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = await self._connect()
        try:
            await conn.execute("PRAGMA journal_mode=WAL")

            if self.sqlite_cfg:
                await conn.execute(f"PRAGMA synchronous={self.sqlite_cfg.synchronous}")
                await conn.execute(f"PRAGMA mmap_size={self.sqlite_cfg.mmap_size}")
                await conn.execute(f"PRAGMA cache_size={self.sqlite_cfg.cache_size}")
                await conn.execute(f"PRAGMA temp_store={self.sqlite_cfg.temp_store}")
                await conn.execute(f"PRAGMA wal_autocheckpoint={self.sqlite_cfg.wal_autocheckpoint}")
                await conn.execute(f"PRAGMA journal_size_limit={self.sqlite_cfg.journal_size_limit}")
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
                cursor = await conn.execute("SELECT 1 FROM _migrations WHERE version=?", (version,))
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
        except Exception:
            await self._rollback_quietly(conn)
            raise
        finally:
            await conn.close()

    async def _connect(self) -> aiosqlite.Connection:
        busy_timeout_ms = self.sqlite_cfg.busy_timeout_ms if self.sqlite_cfg else _DEFAULT_BUSY_TIMEOUT_MS
        conn = await aiosqlite.connect(self.db_path, timeout=busy_timeout_ms / 1000)
        try:
            await conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            await conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = aiosqlite.Row
            return conn
        except Exception:
            await conn.close()
            raise

    async def close(self) -> None:
        pass  # no cached connection

    # ── query helpers ──────────────────────────────────────────────

    async def _with_connection(
        self,
        operation: Callable[[aiosqlite.Connection], Awaitable[_T]],
        *,
        write: bool = False,
    ) -> _T:
        if write:
            async with self._write_lock:
                return await self._run_with_retry(operation)
        return await self._run_with_retry(operation)

    async def _run_with_retry(self, operation: Callable[[aiosqlite.Connection], Awaitable[_T]]) -> _T:
        retry_attempts = self.sqlite_cfg.lock_retry_attempts if self.sqlite_cfg else _DEFAULT_LOCK_RETRY_ATTEMPTS
        base_delay_ms = self.sqlite_cfg.lock_retry_base_delay_ms if self.sqlite_cfg else _DEFAULT_LOCK_RETRY_BASE_DELAY_MS

        for attempt in range(retry_attempts + 1):
            conn: aiosqlite.Connection | None = None
            try:
                conn = await self._connect()
                result = await operation(conn)
                await conn.commit()
                return result
            except Exception as exc:
                if conn is not None:
                    await self._rollback_quietly(conn)
                if not _is_sqlite_lock_error(exc):
                    raise
                if attempt >= retry_attempts:
                    raise ServerBusyError("Doclib database is busy. Retry the request.") from exc
            finally:
                if conn is not None:
                    await conn.close()

            delay_seconds = (base_delay_ms / 1000) * (2**attempt)
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

        raise AssertionError("SQLite retry loop exited unexpectedly")

    @staticmethod
    async def _rollback_quietly(conn: aiosqlite.Connection) -> None:
        try:
            await conn.rollback()
        except Exception:
            pass

    async def execute(self, sql: str, params: tuple | None = None) -> aiosqlite.Cursor:
        async def _operation(conn: aiosqlite.Connection) -> aiosqlite.Cursor:
            return await conn.execute(sql, params or ())

        return await self._with_connection(_operation, write=True)

    async def execute_insert(self, sql: str, params: tuple | None = None) -> int:
        async def _operation(conn: aiosqlite.Connection) -> int:
            cursor = await conn.execute(sql, params or ())
            lastid = cursor.lastrowid
            if lastid is None:
                raise RuntimeError("SQLite insert did not return a row id.")
            return lastid

        return await self._with_connection(_operation, write=True)

    async def fetchone(self, sql: str, params: tuple | None = None) -> dict | None:
        async def _operation(conn: aiosqlite.Connection) -> dict | None:
            cursor = await conn.execute(sql, params or ())
            row = await cursor.fetchone()
            return dict(row) if row else None

        return await self._with_connection(_operation)

    async def fetchone_write(self, sql: str, params: tuple | None = None) -> dict | None:
        """Execute a write statement that returns at most one row."""

        async def _operation(conn: aiosqlite.Connection) -> dict | None:
            cursor = await conn.execute(sql, params or ())
            row = await cursor.fetchone()
            return dict(row) if row else None

        return await self._with_connection(_operation, write=True)

    async def fetchall(self, sql: str, params: tuple | None = None) -> list[dict]:
        async def _operation(conn: aiosqlite.Connection) -> list[dict]:
            cursor = await conn.execute(sql, params or ())
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

        return await self._with_connection(_operation)

    async def commit(self) -> None:
        pass

    async def execute_atomic(self, statements: list[tuple[str, tuple]]) -> int:
        """Execute multiple statements in a single transaction."""

        async def _operation(conn: aiosqlite.Connection) -> int:
            for sql, params in statements:
                await conn.execute(sql, params)
            return len(statements)

        return await self._with_connection(_operation, write=True)


# ── helpers ────────────────────────────────────────────────────────


def _now_ms() -> int:
    return int(time.time() * 1000)


def _is_sqlite_lock_error(exc: Exception) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    error_code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(error_code, int) and error_code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
        return True
    message = str(exc).lower()
    return "database is locked" in message or "database table is locked" in message


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
