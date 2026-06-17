import asyncio
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mineru.config import PatchedConfig
from mineru.doclib import app as doclib_app
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


def test_doclib_runtime_dependencies_are_in_base_install() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    dependency_names = {dependency.split(">", 1)[0].split("=", 1)[0].lower() for dependency in dependencies}

    assert "aiosqlite" in dependency_names
    assert "fnvhash" in dependency_names
    assert "watchfiles" in dependency_names


def test_server_status_reports_configured_socket_path(monkeypatch, tmp_path) -> None:
    cfg = PatchedConfig(
        doclib={
            "data_dir": str(tmp_path),
            "sqlite": {"path": str(tmp_path / "mineru.db")},
            "log": {"path": str(tmp_path / "mineru.log")},
            "uds": {"path": str(tmp_path / "mineru.sock")},
        }
    )

    def _skip_background_task(*args, **kwargs):
        return None

    monkeypatch.setattr(doclib_app, "_create_background_task", _skip_background_task)

    with TestClient(doclib_app.create_app(cfg)) as client:
        response = client.get("/api/v1/server/status")

    assert response.status_code == 200
    assert response.json()["socket_path"] == str(tmp_path / "mineru.sock")
