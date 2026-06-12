"""New doclib app factory and process entrypoint using the interface server."""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..errors import MineruError, error_response, http_status_for
from ..config import Config, LogConfig, config
from .server import DoclibServer
from .types import PARSE_STATUS_FAILED, PARSE_STATUS_PARSING


def create_app(cfg: Config | None = None) -> FastAPI:
    """Create the new interface-backed FastAPI app."""
    if cfg is None:
        cfg = config
    state = AppState()
    app = DoclibServer(state).app
    app.title = "MinerU DocLib"
    app.version = "1.0.0"

    @app.on_event("startup")
    async def startup() -> None:
        from .background.compaction import Compaction
        from .background.device_monitor import DeviceMonitor
        from .background.ingest import IngestWorkerPool
        from .background.parse_server_health import ParseServerHealthCheck, api_server_args_for_tier, get_health
        from .background.parse_worker import ParseWorkerPool
        from .background.scan_worker import ScanWorkerPool
        from .background.watch import WatchLoop
        from .core.db import DatabaseManager
        from .core.fts import FTSManager
        from .services.cleanup_svc import CleanupService
        from .services.config_svc import ConfigService
        from .services.parse_svc import ParseService
        from .services.scan_svc import ScanService
        from .services.search_svc import SearchService

        data_dir = os.path.expanduser(cfg.doclib.data_dir)
        os.makedirs(data_dir, exist_ok=True)

        _setup_logging(cfg.doclib.log)

        db_path = os.path.expanduser(cfg.doclib.sqlite.path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        state.db = DatabaseManager(db_path, cfg.doclib.sqlite)
        await state.db.initialize()

        state.fts = FTSManager(state.db)
        state.config_svc = ConfigService(state.db)
        state.parse_svc = ParseService(state.db, state.fts, state.config_svc, data_dir)
        state.scan_svc = ScanService(state.db, state.config_svc, state.parse_svc)
        state.search_svc = SearchService(state.db, state.fts)
        state.cleanup_svc = CleanupService(state.db, data_dir)

        await state.db.execute("UPDATE files SET locked_at = NULL WHERE locked_at IS NOT NULL")
        await state.db.execute(
            "UPDATE parses SET locked_at = NULL, status = ? WHERE status = ?",
            (PARSE_STATUS_FAILED, PARSE_STATUS_PARSING),
        )

        state.parse_server_proc = None
        local_mode = (await state.config_svc.get("parse_server.local.mode")) or "disabled"
        health = get_health()
        health.local_mode = local_mode
        if local_mode == "managed":
            managed_tier = (await state.config_svc.get("parse_server.local.managed_tier")) or "standard"
            try:
                cmd = [sys.executable, "-m", "mineru.parser.api_server", *api_server_args_for_tier(managed_tier)]
                logging.info("Starting managed parse-server: %s", " ".join(cmd))
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logging.info("Managed parse-server started (PID %d, tier=%s)", proc.pid, managed_tier)
                state.parse_server_proc = proc
                health.managed_proc = proc
                health.local_starting = True
                health.local_started_at = time.time()
            except Exception as exc:
                logging.error("Failed to start managed parse-server: %s", exc)

        state.watch = WatchLoop(state.db, state.config_svc, state.parse_svc, state.scan_svc)
        state.scan_workers = ScanWorkerPool(state.scan_svc, num_workers=1)
        state.ingest_workers = IngestWorkerPool(state.parse_svc, num_workers=cfg.doclib.ingest_workers)
        state.parse_workers = ParseWorkerPool(state.parse_svc, num_workers=cfg.doclib.parse_workers)
        state.device_monitor = DeviceMonitor(state.db, state.config_svc)
        state.compaction = Compaction(state.db, interval_sec=cfg.doclib.compaction_interval_sec, data_dir=data_dir)
        state.health_check = ParseServerHealthCheck(state.config_svc)

        asyncio.create_task(state.watch.run())
        asyncio.create_task(state.scan_workers.run())
        asyncio.create_task(state.ingest_workers.run())
        asyncio.create_task(state.parse_workers.run())
        asyncio.create_task(state.device_monitor.run())
        asyncio.create_task(state.compaction.run())
        asyncio.create_task(state.health_check.run())

        state.start_time = time.time()
        state.pid = os.getpid()
        state.data_dir = data_dir
        state.config = cfg

    @app.on_event("shutdown")
    async def shutdown() -> None:
        if state.parse_server_proc:
            try:
                pid = state.parse_server_proc.pid
                logging.info("Stopping managed parse-server (PID %d)", pid)
                state.parse_server_proc.terminate()
                state.parse_server_proc.wait(timeout=10)
                logging.info("Managed parse-server stopped (PID %d)", pid)
            except subprocess.TimeoutExpired:
                logging.warning("Managed parse-server did not stop within 10s, killing")
                state.parse_server_proc.kill()
            except Exception as exc:
                logging.error("Error stopping managed parse-server: %s", exc)

        for comp in [
            "watch",
            "scan_workers",
            "ingest_workers",
            "parse_workers",
            "device_monitor",
            "compaction",
            "health_check",
        ]:
            c = getattr(state, comp, None)
            if c and hasattr(c, "stop"):
                await c.stop()

        if state.db:
            await state.db.close()

    @app.middleware("http")
    async def attach_state(request: Request, call_next: Callable) -> Any:
        request.state.app = state
        return await call_next(request)

    @app.exception_handler(MineruError)
    async def mineru_error_handler(_request: Request, exc: MineruError) -> JSONResponse:
        return JSONResponse(
            status_code=http_status_for(exc.code, exc.type),
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
    pass


def main() -> None:
    """Entry point: python -m mineru.doclib.app"""
    cfg = config
    uds_path = cfg.doclib.uds.path

    try:
        os.unlink(uds_path)
    except OSError:
        pass

    app = create_app(cfg)
    uv_config = uvicorn.Config(
        app,
        lifespan="on",
        timeout_keep_alive=cfg.doclib.http.timeout,
    )
    server = uvicorn.Server(uv_config)

    uds_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    uds_sock.bind(uds_path)
    os.chmod(uds_path, cfg.doclib.uds.permission)
    uds_sock.listen(cfg.doclib.http.backlog)
    sockets = [uds_sock]

    if cfg.doclib.http.enabled:
        tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            tcp_sock.bind((cfg.doclib.http.host, cfg.doclib.http.port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE and not cfg.doclib.http.strict_port:
                tcp_sock.bind((cfg.doclib.http.host, 0))
                port = tcp_sock.getsockname()[1]
                print(f"Port {cfg.doclib.http.port} in use, using {port}")
            else:
                raise
        else:
            port = cfg.doclib.http.port
        tcp_sock.listen(cfg.doclib.http.backlog)
        sockets.append(tcp_sock)
        print(f"MinerU server listening on UDS {uds_path} and TCP {cfg.doclib.http.host}:{port}")
    else:
        print(f"MinerU server listening on UDS {uds_path}")

    loop = asyncio.new_event_loop()

    async def serve() -> None:
        await server.serve(sockets=sockets)

    try:
        loop.run_until_complete(serve())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            os.unlink(uds_path)
        except OSError:
            pass
        loop.close()


def _setup_logging(log_cfg: LogConfig) -> None:
    logger = logging.getLogger("mineru")
    level = getattr(logging, log_cfg.level.upper(), logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(ch)

        log_path = os.path.expanduser(log_cfg.path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(fh)


if __name__ == "__main__":
    main()
