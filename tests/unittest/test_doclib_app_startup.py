import asyncio
import logging
import tomllib
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from loguru import logger as loguru_logger

from mineru.config import LogConfig, PatchedConfig, config as mineru_config
from mineru.doclib import app as doclib_app
from mineru.doclib.app import _assert_required_schema
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.server import _tail_log, _write_temp_asset
from mineru.version import __version__


def _clear_test_loggers() -> None:
    for name in ("mineru", "uvicorn", "uvicorn.error", "uvicorn.access", "py.warnings"):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        logger.propagate = True


def test_required_schema_check_fails_before_migration_and_passes_after(tmp_path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))

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
    assert "watchfiles" in dependency_names


def test_background_task_crash_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    async def _boom() -> None:
        raise RuntimeError("boom")

    async def _run() -> None:
        task = asyncio.create_task(_boom())
        await asyncio.sleep(0)
        doclib_app._log_background_task_result("test-task", task)

    with caplog.at_level(logging.ERROR):
        asyncio.run(_run())

    assert "Background task test-task crashed: boom" in caplog.text


def test_background_tasks_are_cancelled_and_awaited_on_shutdown() -> None:
    async def _run() -> None:
        state = doclib_app.AppState()
        state.db_ready = asyncio.Event()

        async def _blocked() -> None:
            await asyncio.Event().wait()

        task = doclib_app._create_background_task(state, "blocked", _blocked)
        await asyncio.sleep(0)

        await doclib_app._cancel_background_tasks(state)

        assert task.cancelled()
        assert task.done()
        assert state.background_tasks == []

    asyncio.run(_run())


def test_pending_loop_tasks_are_cancelled_before_loop_close() -> None:
    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(asyncio.Event().wait())

        doclib_app._cancel_pending_loop_tasks(loop)

        assert task.cancelled()
        assert task.done()
    finally:
        loop.close()


def test_setup_logging_routes_application_logs_to_rotating_file_without_stderr_duplication(tmp_path: Path) -> None:
    _clear_test_loggers()
    log_path = tmp_path / "doclib.log"
    access_log_path = tmp_path / "doclib.access.log"
    try:
        doclib_app._setup_logging(LogConfig(app_path=str(log_path), access_path=str(access_log_path)))

        mineru_logger = logging.getLogger("mineru")
        uvicorn_error_logger = logging.getLogger("uvicorn.error")
        uvicorn_access_logger = logging.getLogger("uvicorn.access")
        warnings_logger = logging.getLogger("py.warnings")

        mineru_rotating_handlers = [handler for handler in mineru_logger.handlers if isinstance(handler, RotatingFileHandler)]
        mineru_plain_stream_handlers = [
            handler
            for handler in mineru_logger.handlers
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler)
        ]

        assert len(mineru_rotating_handlers) == 1
        assert mineru_rotating_handlers[0].baseFilename == str(log_path)
        assert mineru_plain_stream_handlers == []
        assert mineru_logger.propagate is False
        assert [handler.baseFilename for handler in uvicorn_error_logger.handlers if isinstance(handler, RotatingFileHandler)] == [
            str(log_path)
        ]
        assert [handler.baseFilename for handler in warnings_logger.handlers if isinstance(handler, RotatingFileHandler)] == [
            str(log_path)
        ]
        assert [handler.baseFilename for handler in uvicorn_access_logger.handlers if isinstance(handler, RotatingFileHandler)] == [
            str(access_log_path)
        ]
    finally:
        _clear_test_loggers()


def test_setup_logging_routes_loguru_records_to_application_log(tmp_path: Path) -> None:
    _clear_test_loggers()
    log_path = tmp_path / "doclib.log"
    access_log_path = tmp_path / "doclib.access.log"
    try:
        doclib_app._setup_logging(LogConfig(app_path=str(log_path), access_path=str(access_log_path)))

        patched_logger = loguru_logger.patch(lambda record: record.update(name="mineru.test_loguru"))
        patched_logger.error("loguru bridged message")

        for handler in logging.getLogger("mineru").handlers:
            handler.flush()

        content = log_path.read_text(encoding="utf-8")
        assert "mineru.test_loguru" in content
        assert "loguru bridged message" in content
    finally:
        _clear_test_loggers()


def test_server_status_reports_configured_socket_path(monkeypatch, tmp_path) -> None:
    app_log_path = tmp_path / "doclib.log"
    access_log_path = tmp_path / "doclib.access.log"
    stdout_log_path = tmp_path / "doclib.stdout.log"
    stderr_log_path = tmp_path / "doclib.stderr.log"
    for prefix, path, count in (
        ("app", app_log_path, 30),
        ("access", access_log_path, 12),
        ("stdout", stdout_log_path, 12),
        ("stderr", stderr_log_path, 12),
    ):
        path.write_text("".join(f"{prefix}-{idx}\n" for idx in range(count)), encoding="utf-8")

    cfg = PatchedConfig(
        doclib={
            "data_dir": str(tmp_path),
            "sqlite": {"path": str(tmp_path / "doclib.db")},
            "log": {
                "app_path": str(app_log_path),
                "access_path": str(access_log_path),
                "stdout_path": str(stdout_log_path),
                "stderr_path": str(stderr_log_path),
            },
            "uds": {"path": str(tmp_path / "doclib.sock")},
        }
    )

    def _skip_background_task(*args, **kwargs):
        return None

    monkeypatch.setattr(doclib_app, "_create_background_task", _skip_background_task)

    with TestClient(doclib_app.create_app(cfg)) as client:
        response = client.get("/api/v1/server/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mineru_home"]
    assert payload["version"] == __version__
    assert isinstance(payload["python_version"], str)
    assert payload["socket_path"] == str(tmp_path / "doclib.sock")
    assert payload["sqlite_path"] == str(tmp_path / "doclib.db")
    assert payload["log_path"] == str(tmp_path / "doclib.log")
    assert payload["access_log_path"] == str(access_log_path)
    assert payload["stdout_log_path"] == str(stdout_log_path)
    assert payload["stderr_log_path"] == str(stderr_log_path)
    assert payload["tcp"] == {"enabled": cfg.doclib.resolved_tcp_enabled, "host": cfg.doclib.tcp.host, "port": None}
    assert payload["app_logs"] == [f"app-{idx}\n" for idx in range(5, 30)]
    assert payload["access_logs"] == [f"access-{idx}\n" for idx in range(2, 12)]
    assert payload["stdout_logs"] == [f"stdout-{idx}\n" for idx in range(2, 12)]
    assert payload["stderr_logs"] == [f"stderr-{idx}\n" for idx in range(2, 12)]


def test_doclib_app_uses_lifespan_instead_of_deprecated_event_handlers(monkeypatch, tmp_path) -> None:
    cfg = PatchedConfig(
        doclib={
            "data_dir": str(tmp_path),
            "sqlite": {"path": str(tmp_path / "doclib.db")},
            "log": {
                "app_path": str(tmp_path / "doclib.log"),
                "access_path": str(tmp_path / "doclib.access.log"),
            },
            "uds": {"path": str(tmp_path / "doclib.sock")},
        }
    )

    def _skip_background_task(*args, **kwargs):
        return None

    monkeypatch.setattr(doclib_app, "_create_background_task", _skip_background_task)

    app = doclib_app.create_app(cfg)

    assert app.router.on_startup == []
    assert app.router.on_shutdown == []
    with TestClient(app) as client:
        response = client.get("/api/v1/server/status")

    assert response.status_code == 200


def test_startup_resets_running_scans_to_failed(monkeypatch, tmp_path) -> None:
    db = DatabaseManager(str(tmp_path / "doclib.db"))
    asyncio.run(db.initialize())
    asyncio.run(
        db.execute(
            "INSERT INTO scans (path, kind, source, status, locked_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(tmp_path / "watch-root"), "watch", "watch", "running", 123, 1000, 1000),
        )
    )

    cfg = PatchedConfig(
        doclib={
            "data_dir": str(tmp_path),
            "sqlite": {"path": str(tmp_path / "doclib.db")},
            "log": {
                "app_path": str(tmp_path / "doclib.log"),
                "access_path": str(tmp_path / "doclib.access.log"),
            },
            "uds": {"path": str(tmp_path / "doclib.sock")},
        }
    )

    def _skip_background_task(*args, **kwargs):
        return None

    monkeypatch.setattr(doclib_app, "_create_background_task", _skip_background_task)

    with TestClient(doclib_app.create_app(cfg)) as client:
        response = client.get("/api/v1/server/status")

    assert response.status_code == 200
    row = asyncio.run(db.fetchone("SELECT status, locked_at, error_code FROM scans"))
    assert row == {"status": "failed", "locked_at": None, "error_code": "scan_interrupted"}


def test_telemetry_observation_route_uses_context_headers(monkeypatch, tmp_path) -> None:
    cfg = PatchedConfig(
        doclib={
            "data_dir": str(tmp_path),
            "sqlite": {"path": str(tmp_path / "doclib.db")},
            "log": {
                "app_path": str(tmp_path / "doclib.log"),
                "access_path": str(tmp_path / "doclib.access.log"),
            },
            "uds": {"path": str(tmp_path / "doclib.sock")},
        }
    )

    def _skip_background_task(*args, **kwargs):
        return None

    monkeypatch.setattr(doclib_app, "_create_background_task", _skip_background_task)

    with TestClient(doclib_app.create_app(cfg)) as client:
        response = client.post(
            "/api/v1/observations",
            headers={"X-MinerU-Telemetry-Source": "cli", "X-MinerU-Telemetry-Caller": "agent"},
            json={"observations": [{"metric_name": "parse.request.count"}]},
        )
        preview = client.get("/api/v1/telemetry/preview")

    assert response.status_code == 200
    assert response.json() == {"accepted": 1}
    assert preview.status_code == 200
    metrics = preview.json()["body"]["metrics"]
    assert metrics == [{"name": "parse.request.count", "value": 1, "dimensions": {"caller": "agent", "source": "cli"}}]


def test_write_temp_asset_uses_temp_read_assets_directory(tmp_path: Path) -> None:
    asset = _write_temp_asset(
        str(tmp_path),
        "abc1234",
        "png",
        b"image-bytes",
        mime_type="image/png",
        width=10,
        height=10,
    )

    asset_path = Path(asset.path)
    assert asset_path.parent == tmp_path / "temp" / "read-assets"


def test_tail_log_uses_current_config_default_when_data_dir_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log_path = tmp_path / "doclib.log"
    log_path.write_text("line-1\nline-2\n", encoding="utf-8")
    monkeypatch.setattr(mineru_config.doclib, "log", LogConfig(app_path=str(log_path)))

    assert _tail_log("") == ["line-1\n", "line-2\n"]
