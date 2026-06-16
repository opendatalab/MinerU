import asyncio

import pytest

from mineru.doclib.app import _assert_required_schema
from mineru.doclib.core.db import DatabaseManager


def test_required_schema_check_fails_before_migration_and_passes_after(tmp_path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "mineru.db"))

        with pytest.raises(RuntimeError, match="missing tables"):
            await _assert_required_schema(db)

        await db.initialize()
        await _assert_required_schema(db)

        row = await db.fetchone("SELECT name FROM sqlite_master WHERE type='table' AND name='parses'")
        assert row == {"name": "parses"}

    asyncio.run(_run())
