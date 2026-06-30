import asyncio
import sqlite3

import pytest

from mineru.doclib.core.db import DatabaseManager


def test_failed_write_rolls_back_and_closes_connection(tmp_path) -> None:
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
