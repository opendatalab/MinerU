import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from mineru.config import SQLiteConfig
from mineru.doclib.core.db import DatabaseManager
from mineru.errors import ServerBusyError


class _Cursor:
    def __init__(self, row: dict[str, Any] | None = None) -> None:
        self._row = row

    async def fetchone(self) -> dict[str, Any] | None:
        return self._row

    async def fetchall(self) -> list[dict[str, Any]]:
        return [self._row] if self._row is not None else []


class _FakeConnection:
    def __init__(self, *, locked: bool = False, delay: float = 0.0) -> None:
        self.locked = locked
        self.delay = delay
        self.statements: list[str] = []
        self.committed = False
        self.rolled_back = False
        self.closed = False

    async def execute(self, sql: str, _params: tuple[Any, ...] = ()) -> _Cursor:
        self.statements.append(sql)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.locked:
            raise sqlite3.OperationalError("database is locked")
        return _Cursor({"value": 1})

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True

    async def close(self) -> None:
        self.closed = True


def test_failed_write_rolls_back_and_closes_connection(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        await db.execute("CREATE TABLE child (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parent(id))")

        with pytest.raises(sqlite3.IntegrityError):
            await db.execute_insert("INSERT INTO child(parent_id) VALUES (?)", (999,))

        await db.execute("INSERT INTO parent(id) VALUES (?)", (1,))
        row = await db.fetchone("SELECT id FROM parent")
        assert row == {"id": 1}

    asyncio.run(_run())


def test_initialize_enables_wal_mode(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))

        await db.initialize()

        row = await db.fetchone("PRAGMA journal_mode")
        assert row == {"journal_mode": "wal"}

    asyncio.run(_run())


def test_operational_connection_sets_busy_timeout_without_resetting_wal(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        connection = _FakeConnection()
        connect_timeouts: list[float] = []

        async def _connect(_path: str, *, timeout: float) -> _FakeConnection:
            connect_timeouts.append(timeout)
            return connection

        monkeypatch.setattr("mineru.doclib.core.db.aiosqlite.connect", _connect)
        db = DatabaseManager("/tmp/doclib.db", SQLiteConfig(busy_timeout_ms=4321))

        await db._connect()

        assert connect_timeouts == [4.321]
        assert connection.statements == ["PRAGMA busy_timeout=4321", "PRAGMA foreign_keys=ON"]

    asyncio.run(_run())


def test_locked_operation_retries_with_a_fresh_connection() -> None:
    async def _run() -> None:
        db = DatabaseManager(
            "/tmp/doclib.db",
            SQLiteConfig(lock_retry_attempts=1, lock_retry_base_delay_ms=0),
        )
        connections = [_FakeConnection(locked=True), _FakeConnection()]

        async def _connect() -> _FakeConnection:
            return connections.pop(0)

        db._connect = _connect  # type: ignore[method-assign]

        row = await db.fetchone("SELECT 1 AS value")

        assert row == {"value": 1}
        assert not connections

    asyncio.run(_run())


def test_exhausted_lock_retries_raise_server_busy() -> None:
    async def _run() -> None:
        db = DatabaseManager(
            "/tmp/doclib.db",
            SQLiteConfig(lock_retry_attempts=2, lock_retry_base_delay_ms=0),
        )
        connect_count = 0

        async def _connect() -> _FakeConnection:
            nonlocal connect_count
            connect_count += 1
            return _FakeConnection(locked=True)

        db._connect = _connect  # type: ignore[method-assign]

        with pytest.raises(ServerBusyError) as exc_info:
            await db.fetchone("SELECT 1")

        assert exc_info.value.code == "server_busy"
        assert connect_count == 3

    asyncio.run(_run())


def test_write_returning_operations_are_serialized() -> None:
    async def _run() -> None:
        db = DatabaseManager("/tmp/doclib.db", SQLiteConfig(lock_retry_attempts=0))
        active_operations = 0
        max_active_operations = 0

        class _TrackedConnection(_FakeConnection):
            async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> _Cursor:
                nonlocal active_operations, max_active_operations
                active_operations += 1
                max_active_operations = max(max_active_operations, active_operations)
                try:
                    await asyncio.sleep(0.01)
                    return await super().execute(sql, params)
                finally:
                    active_operations -= 1

        async def _connect() -> _FakeConnection:
            return _TrackedConnection()

        db._connect = _connect  # type: ignore[method-assign]

        await asyncio.gather(
            db.fetchone_write("UPDATE tasks SET locked_at=1 WHERE id=1 RETURNING *"),
            db.fetchone_write("UPDATE tasks SET locked_at=1 WHERE id=2 RETURNING *"),
        )

        assert max_active_operations == 1

    asyncio.run(_run())


def test_read_operations_remain_concurrent() -> None:
    async def _run() -> None:
        db = DatabaseManager("/tmp/doclib.db", SQLiteConfig(lock_retry_attempts=0))
        active_operations = 0
        max_active_operations = 0

        class _TrackedConnection(_FakeConnection):
            async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> _Cursor:
                nonlocal active_operations, max_active_operations
                active_operations += 1
                max_active_operations = max(max_active_operations, active_operations)
                try:
                    await asyncio.sleep(0.01)
                    return await super().execute(sql, params)
                finally:
                    active_operations -= 1

        async def _connect() -> _FakeConnection:
            return _TrackedConnection()

        db._connect = _connect  # type: ignore[method-assign]

        await asyncio.gather(
            db.fetchone("SELECT 1"),
            db.fetchall("SELECT 2"),
        )

        assert max_active_operations == 2

    asyncio.run(_run())


def test_write_recovers_after_external_sqlite_lock_is_released(tmp_path: Path) -> None:
    async def _run() -> None:
        db_path = tmp_path / "doclib.db"
        db = DatabaseManager(
            str(db_path),
            SQLiteConfig(
                busy_timeout_ms=10,
                lock_retry_attempts=5,
                lock_retry_base_delay_ms=10,
            ),
        )
        await db.initialize()
        await db.execute("CREATE TABLE counters (id INTEGER PRIMARY KEY, value INTEGER NOT NULL)")
        await db.execute_insert("INSERT INTO counters(id, value) VALUES (1, 0)")

        blocker = sqlite3.connect(db_path, timeout=0)
        blocker.execute("BEGIN IMMEDIATE")
        blocker.execute("UPDATE counters SET value=1 WHERE id=1")
        try:
            update_task = asyncio.create_task(db.execute("UPDATE counters SET value=value+1 WHERE id=1"))
            await asyncio.sleep(0.05)
            blocker.commit()
            await update_task
        finally:
            blocker.close()

        row = await db.fetchone("SELECT value FROM counters WHERE id=1")
        assert row == {"value": 2}

    asyncio.run(_run())


def test_concurrent_reads_and_writes_complete_without_lock_errors(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(
            str(tmp_path / "doclib.db"),
            SQLiteConfig(
                busy_timeout_ms=100,
                lock_retry_attempts=3,
                lock_retry_base_delay_ms=5,
            ),
        )
        await db.initialize()
        await db.execute("CREATE TABLE counters (id INTEGER PRIMARY KEY, value INTEGER NOT NULL)")
        await db.execute_insert("INSERT INTO counters(id, value) VALUES (1, 0)")

        async def _reader() -> None:
            for _ in range(10):
                row = await db.fetchone("SELECT value FROM counters WHERE id=1")
                assert row is not None

        async def _writer() -> None:
            for _ in range(20):
                await db.fetchone_write("UPDATE counters SET value=value+1 WHERE id=1 RETURNING value")

        await asyncio.gather(
            *(_reader() for _ in range(32)),
            *(_writer() for _ in range(4)),
        )

        row = await db.fetchone("SELECT value FROM counters WHERE id=1")
        assert row == {"value": 80}

    asyncio.run(_run())
