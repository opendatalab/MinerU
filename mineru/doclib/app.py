"""New doclib app factory and process entrypoint using the interface server."""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import socket
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING, Any, AsyncIterator, cast

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger as loguru_logger

from ..config import Config, LogConfig, _mineru_home, config
from ..errors import MineruError, error_response, http_status_for
from .endpoint import EndpointTransport, remove_endpoint_file, uds_available, write_endpoint_file
from .server import DoclibServer
from .types import PARSE_STATUS_FAILED, PARSE_STATUS_PARSING, SCAN_STATUS_FAILED, SCAN_STATUS_RUNNING

if TYPE_CHECKING:
    from loguru import Message as LoguruMessage
else:
    LoguruMessage = Any


def create_app(cfg: Config | None = None) -> FastAPI:
    """Create the new interface-backed FastAPI app."""
    if cfg is None:
        cfg = config
    _setup_logging(cfg.doclib.log)
    state = AppState()
    app = DoclibServer(state).app
    app.title = "MinerU DocLib"
    app.version = "1.0.0"
    app.state.doclib_state = state

    async def startup() -> None:
        from .background.compaction import Compaction
        from .background.device_monitor import DeviceMonitor
        from .background.ingest import IngestWorkerPool
        from .background.parse_server_health import (
            ParseServerHealthCheck,
            get_health,
            start_managed_parse_server,
        )
        from .background.parse_worker import ParseWorkerPool
        from .background.scan_worker import ScanWorkerPool
        from .background.telemetry_flush import TelemetryFlushLoop
        from .background.watch import WatchLoop
        from .core.db import DatabaseManager
        from .core.fts import FTSManager
        from .services.cleanup_svc import CleanupService
        from .services.config_svc import ConfigService
        from .services.parse_svc import ParseService
        from .services.scan_svc import ScanService
        from .services.search_svc import SearchService
        from .telemetry import TelemetryService, TelemetryStore

        data_dir = os.path.expanduser(cfg.doclib.data_dir)
        os.makedirs(data_dir, exist_ok=True)

        db_path = os.path.expanduser(cfg.doclib.sqlite.path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        state.db = DatabaseManager(db_path, cfg.doclib.sqlite)
        await state.db.initialize()
        await _assert_required_schema(state.db)
        state.db_ready = asyncio.Event()
        state.db_ready.set()

        state.fts = FTSManager(state.db)
        state.config_svc = ConfigService(state.db)
        state.telemetry_svc = TelemetryService(TelemetryStore(state.db))
        await state.telemetry_svc.initialize()
        state.parse_svc = ParseService(
            state.db,
            state.fts,
            state.config_svc,
            data_dir,
            parse_lock_timeout_sec=cfg.doclib.parse_lock_timeout_sec,
            telemetry_svc=state.telemetry_svc,
        )
        state.scan_svc = ScanService(
            state.db,
            state.config_svc,
            state.parse_svc,
            scan_lock_timeout_sec=cfg.doclib.scan_lock_timeout_sec,
            telemetry_svc=state.telemetry_svc,
        )
        state.search_svc = SearchService(state.db, state.fts)
        state.cleanup_svc = CleanupService(state.db, data_dir)

        await state.db.execute("UPDATE files SET locked_at = NULL WHERE locked_at IS NOT NULL")
        await state.db.execute(
            "UPDATE parses SET locked_at = NULL, status = ? WHERE status = ?",
            (PARSE_STATUS_FAILED, PARSE_STATUS_PARSING),
        )
        now_ms = int(time.time() * 1000)
        await state.db.execute(
            "UPDATE scans SET locked_at = NULL, status = ?, error_code = COALESCE(error_code, ?), "
            "error_msg = COALESCE(error_msg, ?), finished_at = COALESCE(finished_at, ?), updated_at = ? "
            "WHERE status = ?",
            (
                SCAN_STATUS_FAILED,
                "scan_interrupted",
                "Scan was interrupted by server shutdown or crash.",
                now_ms,
                now_ms,
                SCAN_STATUS_RUNNING,
            ),
        )

        local_mode = (await state.config_svc.get("parse_server.local.mode")) or "disabled"
        health = get_health()
        health.local_mode = local_mode
        if local_mode == "managed":
            managed_tier = (await state.config_svc.get("parse_server.local.managed_tier")) or "standard"
            health.managed_tier = managed_tier
            try:
                proc, managed_url = start_managed_parse_server(
                    tier=managed_tier,
                    managed_cfg=cfg.doclib.managed_parse_server,
                    log_cfg=cfg.doclib.log,
                    marker="start",
                )
                health.managed_url = managed_url
                logging.info("Managed parse-server started (PID %d, tier=%s)", proc.pid, managed_tier)
                health.managed_proc = proc
                health.local_starting = True
                health.local_started_at = asyncio.get_event_loop().time()
            except Exception as exc:
                logging.error("Failed to start managed parse-server: %s", exc)

        state.watch = WatchLoop(
            state.db,
            state.config_svc,
            state.parse_svc,
            scan_interval_sec=cfg.doclib.scan_interval_sec,
            scan_svc=state.scan_svc,
        )
        state.scan_workers = ScanWorkerPool(state.scan_svc, num_workers=1)
        state.ingest_workers = IngestWorkerPool(
            state.parse_svc,
            num_workers=cfg.doclib.ingest_workers,
            lock_timeout_sec=cfg.doclib.ingest_lock_timeout_sec,
        )
        state.parse_workers = ParseWorkerPool(state.parse_svc, num_workers=cfg.doclib.parse_workers)
        state.device_monitor = DeviceMonitor(
            state.db,
            state.config_svc,
            interval_sec=cfg.doclib.device_check_interval_sec,
            scan_svc=state.scan_svc,
        )
        state.compaction = Compaction(state.db, interval_sec=cfg.doclib.compaction_interval_sec, data_dir=data_dir)
        state.health_check = ParseServerHealthCheck(
            state.config_svc,
            interval_sec=cfg.doclib.parse_server_health_check_interval_sec,
            probe_timeout_sec=cfg.doclib.parse_server_probe_timeout_sec,
            startup_grace_sec=cfg.doclib.parse_server_startup_grace_sec,
            stop_timeout_sec=cfg.doclib.parse_server_stop_timeout_sec,
            managed_parse_server=cfg.doclib.managed_parse_server,
            log_cfg=cfg.doclib.log,
        )
        state.telemetry_flush = TelemetryFlushLoop(state.telemetry_svc)

        _create_background_task(state, "watch", state.watch.run)
        _create_background_task(state, "scan_workers", state.scan_workers.run)
        _create_background_task(state, "ingest_workers", state.ingest_workers.run)
        _create_background_task(state, "parse_workers", state.parse_workers.run)
        _create_background_task(state, "device_monitor", state.device_monitor.run)
        _create_background_task(state, "compaction", state.compaction.run)
        _create_background_task(state, "health_check", state.health_check.run)
        _create_background_task(state, "telemetry_flush", state.telemetry_flush.run)

        state.start_time = time.time()
        state.pid = os.getpid()
        state.mineru_home = _mineru_home()
        state.data_dir = data_dir
        state.sqlite_path = db_path
        state.log_path = os.path.expanduser(cfg.doclib.log.resolved_app_path)
        state.access_log_path = os.path.expanduser(cfg.doclib.log.resolved_access_path)
        state.stdout_log_path = os.path.expanduser(cfg.doclib.log.resolved_stdout_path)
        state.stderr_log_path = os.path.expanduser(cfg.doclib.log.resolved_stderr_path)
        state.parse_server_stdout_log_path = os.path.expanduser(cfg.doclib.log.resolved_parse_server_stdout_path)
        state.parse_server_stderr_log_path = os.path.expanduser(cfg.doclib.log.resolved_parse_server_stderr_path)
        state.socket_path = os.path.expanduser(cfg.doclib.uds.path)
        state.tcp_enabled = cfg.doclib.resolved_tcp_enabled
        state.tcp_host = cfg.doclib.tcp.host
        state.config = cfg

    async def shutdown() -> None:
        from .background.parse_server_health import get_health, stop_managed_parse_server

        health = get_health()
        if health.managed_proc:
            stop_managed_parse_server(
                health.managed_proc,
                timeout_sec=cfg.doclib.parse_server_stop_timeout_sec,
                reason="doclib shutdown",
            )
            health.managed_proc = None

        for comp in [
            "watch",
            "scan_workers",
            "ingest_workers",
            "parse_workers",
            "device_monitor",
            "compaction",
            "health_check",
            "telemetry_flush",
        ]:
            c = getattr(state, comp, None)
            if c and hasattr(c, "stop"):
                await c.stop()

        await _cancel_background_tasks(state)

        if state.db:
            await state.db.close()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await startup()
        try:
            yield
        finally:
            await shutdown()

    app.router.lifespan_context = lifespan

    @app.middleware("http")
    async def attach_state(request: Request, call_next: Callable) -> Any:
        request.state.app = state
        from .telemetry.constants import TelemetryCaller, TelemetrySource
        from .telemetry import TelemetryContext, reset_telemetry_context, set_telemetry_context

        source = request.headers.get("X-MinerU-Telemetry-Source") or "http_api"
        caller = request.headers.get("X-MinerU-Telemetry-Caller") or "http_client"
        if source not in {"cli", "sdk", "http_api", "watch", "background", "unknown"}:
            source = "unknown"
        if caller not in {"agent", "user", "sdk", "http_client", "system", "unknown"}:
            caller = "unknown"
        token = set_telemetry_context(
            TelemetryContext(source=cast(TelemetrySource, source), caller=cast(TelemetryCaller, caller))
        )
        try:
            return await call_next(request)
        finally:
            reset_telemetry_context(token)

    @app.exception_handler(MineruError)
    async def mineru_error_handler(_request: Request, exc: MineruError) -> JSONResponse:
        return JSONResponse(
            status_code=http_status_for(exc.code),
            content=error_response(exc),
        )

    @app.exception_handler(Exception)
    async def catch_all_handler(_request: Request, exc: Exception) -> JSONResponse:
        import traceback

        logging.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content=error_response(MineruError("internal_error", str(exc))),
        )

    return app


class AppState:
    def __init__(self) -> None:
        self.db: Any = None
        self.fts: Any = None
        self.config_svc: Any = None
        self.parse_svc: Any = None
        self.scan_svc: Any = None
        self.search_svc: Any = None
        self.cleanup_svc: Any = None
        self.telemetry_svc: Any = None
        self.db_ready: asyncio.Event | None = None
        self.watch: Any = None
        self.scan_workers: Any = None
        self.ingest_workers: Any = None
        self.parse_workers: Any = None
        self.device_monitor: Any = None
        self.compaction: Any = None
        self.health_check: Any = None
        self.telemetry_flush: Any = None
        self.background_tasks: list[asyncio.Task[Any]] = []
        self.start_time: float = 0.0
        self.pid: int = 0
        self.mineru_home: str = ""
        self.socket_path: str = ""
        self.data_dir: str = ""
        self.sqlite_path: str = ""
        self.log_path: str = ""
        self.access_log_path: str = ""
        self.stdout_log_path: str = ""
        self.stderr_log_path: str = ""
        self.parse_server_stdout_log_path: str = ""
        self.parse_server_stderr_log_path: str = ""
        self.tcp_enabled: bool = False
        self.tcp_host: str = ""
        self.tcp_port: int | None = None
        self.config: Config | None = None


REQUIRED_SCHEMA_TABLES = {
    "_migrations",
    "config",
    "docs",
    "exclude_rules",
    "files",
    "fts_contents",
    "fts_filenames",
    "parses",
    "parsing_rules",
    "scans",
    "telemetry_aggregates",
    "telemetry_state",
    "watches",
}


async def _assert_required_schema(db: Any) -> None:
    rows = await db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
    existing = {row["name"] for row in rows}
    missing = sorted(REQUIRED_SCHEMA_TABLES - existing)
    if missing:
        raise RuntimeError(f"Doclib database migration incomplete; missing tables: {', '.join(missing)}")


def _create_background_task(state: AppState, name: str, runner: Callable[[], Any]) -> asyncio.Task:
    task = asyncio.create_task(_run_after_db_ready(state, name, runner), name=f"doclib:{name}")
    state.background_tasks.append(task)
    task.add_done_callback(lambda completed: _finish_background_task(state, name, completed))
    return task


def _finish_background_task(state: AppState, name: str, task: asyncio.Task[Any]) -> None:
    _log_background_task_result(name, task)
    if task in state.background_tasks:
        state.background_tasks.remove(task)


def _log_background_task_result(name: str, task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    if exc is not None:
        logging.error(
            "Background task %s crashed: %s",
            name,
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )


async def _cancel_background_tasks(state: AppState) -> None:
    tasks = list(state.background_tasks)
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    state.background_tasks.clear()


def _cancel_pending_loop_tasks(loop: asyncio.AbstractEventLoop) -> None:
    if loop.is_closed():
        return
    tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))


async def _run_after_db_ready(state: AppState, name: str, runner: Callable[[], Any]) -> None:
    if state.db_ready is None:
        raise RuntimeError(f"Cannot start background task {name}: database is not initialized")
    await state.db_ready.wait()
    await runner()


def _format_transport(transport: EndpointTransport) -> str:
    if transport.type == "uds":
        return f"UDS {transport.path}"
    return f"TCP {transport.base_url}"


def _bind_tcp_socket(host: str, port: int, *, strict_port: bool, port_probe_count: int) -> tuple[socket.socket, int]:
    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    if strict_port:
        tcp_sock.bind((host, port))
        return tcp_sock, tcp_sock.getsockname()[1]

    last_exc: OSError | None = None
    for candidate_port in range(port, port + port_probe_count):
        try:
            tcp_sock.bind((host, candidate_port))
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            last_exc = exc
            continue
        return tcp_sock, tcp_sock.getsockname()[1]

    raise RuntimeError(f"No available TCP port in range {port}-{port + port_probe_count - 1}.") from last_exc


def main() -> None:
    """Entry point: python -m mineru.doclib.app"""
    cfg = config
    uds_path = os.path.expanduser(cfg.doclib.uds.path)
    endpoint_path = os.path.expanduser(cfg.doclib.endpoint_path)
    uds_enabled = cfg.doclib.resolved_uds_enabled
    tcp_enabled = cfg.doclib.resolved_tcp_enabled

    if not uds_enabled and not tcp_enabled:
        raise RuntimeError("At least one doclib local transport must be enabled.")
    if uds_enabled and not uds_available():
        raise RuntimeError(
            "Unix domain socket is enabled but is not available in this Python runtime. Enable doclib.tcp or disable doclib.uds."
        )

    app = create_app(cfg)
    uv_config = uvicorn.Config(
        app,
        log_config=None,
        lifespan="on",
        timeout_keep_alive=cfg.doclib.tcp.timeout,
    )
    server = uvicorn.Server(uv_config)

    sockets: list[socket.socket] = []
    transports: list[EndpointTransport] = []

    if uds_enabled:
        try:
            os.unlink(uds_path)
        except OSError:
            pass
        uds_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        uds_sock.bind(uds_path)
        os.chmod(uds_path, cfg.doclib.uds.permission)
        uds_sock.listen(cfg.doclib.tcp.backlog)
        sockets.append(uds_sock)
        transports.append(EndpointTransport(type="uds", path=uds_path))

    if tcp_enabled:
        tcp_sock, port = _bind_tcp_socket(
            cfg.doclib.tcp.host,
            cfg.doclib.tcp.port,
            strict_port=cfg.doclib.tcp.strict_port,
            port_probe_count=cfg.doclib.tcp.port_probe_count,
        )
        if cfg.doclib.tcp.port != 0 and port != cfg.doclib.tcp.port:
            print(f"Port {cfg.doclib.tcp.port} in use, using {port}")
        tcp_sock.listen(cfg.doclib.tcp.backlog)
        sockets.append(tcp_sock)
        app.state.doclib_state.tcp_port = port
        transports.append(EndpointTransport(type="tcp", base_url=f"http://{cfg.doclib.tcp.host}:{port}"))
    else:
        app.state.doclib_state.tcp_port = None

    write_endpoint_file(endpoint_path, pid=os.getpid(), transports=transports)
    print("MinerU server listening on " + " and ".join(_format_transport(transport) for transport in transports))

    loop = asyncio.new_event_loop()

    async def serve() -> None:
        await server.serve(sockets=sockets)

    try:
        loop.run_until_complete(serve())
    except KeyboardInterrupt:
        pass
    finally:
        if uds_enabled:
            try:
                os.unlink(uds_path)
            except OSError:
                pass
        remove_endpoint_file(endpoint_path)
        _cancel_pending_loop_tasks(loop)
        loop.close()


def _setup_logging(log_cfg: LogConfig) -> None:
    level = getattr(logging, log_cfg.level.upper(), logging.INFO)
    log_path = os.path.expanduser(log_cfg.resolved_app_path)
    access_log_path = os.path.expanduser(log_cfg.resolved_access_path)
    _ensure_log_dir(log_path)
    _ensure_log_dir(access_log_path)

    app_logger_names = ("mineru", "uvicorn", "uvicorn.error", "py.warnings")
    access_logger_names = ("uvicorn.access",)
    logger_names = (*app_logger_names, *access_logger_names)
    old_handlers: list[logging.Handler] = []
    for name in logger_names:
        logger = logging.getLogger(name)
        old_handlers.extend(logger.handlers)
        logger.handlers.clear()

    for handler in {id(handler): handler for handler in old_handlers}.values():
        handler.close()

    app_handler = _rotating_log_handler(log_path, level)
    access_handler = _rotating_log_handler(access_log_path, level)

    for name in app_logger_names:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        logger.addHandler(app_handler)

    for name in access_logger_names:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        logger.addHandler(access_handler)

    logging.captureWarnings(True)
    _setup_loguru_bridge(log_cfg.level.upper())


def _ensure_log_dir(path: str) -> None:
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)


def _setup_loguru_bridge(level_name: str) -> None:
    loguru_logger.remove()
    loguru_logger.add(
        _forward_loguru_to_logging,
        level=level_name,
        backtrace=False,
        diagnose=False,
        format="{message}",
    )


def _forward_loguru_to_logging(message: LoguruMessage) -> None:
    record = message.record
    logger_name = record["name"] or "mineru.loguru"
    exc = record["exception"]
    exc_info = None
    if exc is not None and exc.type is not None and exc.value is not None:
        exc_info = (exc.type, exc.value, exc.traceback)
    logging.getLogger(logger_name).log(
        record["level"].no,
        record["message"],
        exc_info=exc_info,
    )


def _rotating_log_handler(path: str, level: int) -> RotatingFileHandler:
    handler = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=3)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    return handler


if __name__ == "__main__":
    main()
