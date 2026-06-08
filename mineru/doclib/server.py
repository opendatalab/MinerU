"""mineru doclib — FastAPI app factory, lifecycle, UDS + optional TCP."""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import socket
import time
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mineru.errors import MineruError, error_response
from .config import Config, LogConfig


def create_app(cfg: Config | None = None) -> FastAPI:
    """Create the FastAPI app.  Accepts an optional Config for testing."""
    if cfg is None:
        cfg = Config()
    app = FastAPI(title="MinerU DocLib", version="1.0.0")
    state = AppState()

    # ── startup ──────────────────────────────────────────────────

    @app.on_event("startup")
    async def startup() -> None:
        from .core.db import DatabaseManager
        from .core.fts import FTSManager
        from .services.parse_svc import ParseService
        from .services.search_svc import SearchService
        from .services.config_svc import ConfigService
        from .services.cleanup_svc import CleanupService
        from .background.watch import WatchLoop
        from .background.ingest import IngestWorkerPool
        from .background.parse_worker import ParseWorkerPool
        from .background.device_monitor import DeviceMonitor
        from .background.compaction import Compaction

        data_dir = os.path.expanduser(cfg.server.data_dir)
        os.makedirs(data_dir, exist_ok=True)

        _setup_logging(cfg.server.log)

        db_path = os.path.expanduser(cfg.sqlite.path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        state.db = DatabaseManager(db_path, cfg.sqlite)
        await state.db.initialize()

        state.fts = FTSManager(state.db)
        state.config_svc = ConfigService(state.db)
        state.parse_svc = ParseService(state.db, state.fts, state.config_svc, data_dir)
        state.search_svc = SearchService(state.db, state.fts)
        state.cleanup_svc = CleanupService(state.db, data_dir)

        # crash recovery
        await state.db.execute(
            "UPDATE files SET locked_at = NULL WHERE locked_at IS NOT NULL"
        )
        await state.db.execute(
            "UPDATE parses SET locked_at = NULL, status = 'failed' WHERE status = 'parsing'"
        )

        # background tasks
        state.watch = WatchLoop(state.db, state.config_svc, state.parse_svc)
        state.ingest_workers = IngestWorkerPool(
            state.parse_svc, num_workers=cfg.server.ingest_workers
        )
        state.parse_workers = ParseWorkerPool(
            state.parse_svc, num_workers=cfg.server.parse_workers
        )
        state.device_monitor = DeviceMonitor(state.db, state.config_svc)
        state.compaction = Compaction(
            state.db, interval_sec=cfg.server.compaction_interval_sec
        )

        asyncio.create_task(state.watch.run())
        asyncio.create_task(state.ingest_workers.run())
        asyncio.create_task(state.parse_workers.run())
        asyncio.create_task(state.device_monitor.run())
        asyncio.create_task(state.compaction.run())

        state.start_time = time.time()
        state.pid = os.getpid()
        state.data_dir = data_dir
        state.config = cfg

    # ── shutdown ─────────────────────────────────────────────────

    @app.on_event("shutdown")
    async def shutdown() -> None:
        for comp in [
            "watch",
            "ingest_workers",
            "parse_workers",
            "device_monitor",
            "compaction",
        ]:
            c = getattr(state, comp, None)
            if c and hasattr(c, "stop"):
                await c.stop()

        if state.db:
            await state.db.close()

    # ── middleware ────────────────────────────────────────────────

    @app.middleware("http")
    async def attach_state(request: Request, call_next: Callable) -> Any:
        request.state.app = state
        return await call_next(request)

    # ── error handler ────────────────────────────────────────────

    @app.exception_handler(MineruError)
    async def mineru_error_handler(_request: Request, exc: MineruError) -> JSONResponse:
        status_map = {
            "invalid_request_error": 400,
            "authentication_error": 401,
            "permission_error": 403,
            "rate_limit_error": 429,
            "engine_error": 500,
            "api_error": 500,
        }
        return JSONResponse(
            status_code=status_map.get(exc.type, 500),
            content=error_response(exc),
        )

    # ── routes ───────────────────────────────────────────────────

    from .routes.parse import router as parse_router
    from .routes.search import router as search_router
    from .routes.info import router as info_router
    from .routes.config import router as config_router
    from .routes.cleanup import router as cleanup_router
    from .routes.server import router as server_router

    app.include_router(parse_router)
    app.include_router(search_router)
    app.include_router(info_router)
    app.include_router(config_router)
    app.include_router(cleanup_router)
    app.include_router(server_router)

    return app


# ── AppState ────────────────────────────────────────────────────────


class AppState:
    pass


# ── server entry point ─────────────────────────────────────────────


def main() -> None:
    """Entry point: python -m mineru.doclib.server"""
    cfg = Config()
    uds_path = cfg.server.uds.path

    # remove stale socket
    try:
        os.unlink(uds_path)
    except OSError:
        pass

    app = create_app(cfg)
    uv_config = uvicorn.Config(
        app,
        lifespan="on",
        timeout_keep_alive=cfg.server.http.timeout,
    )
    server = uvicorn.Server(uv_config)

    # UDS
    uds_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    uds_sock.bind(uds_path)
    os.chmod(uds_path, cfg.server.uds.permission)
    uds_sock.listen(cfg.server.http.backlog)
    sockets = [uds_sock]

    if cfg.server.http.enabled:
        tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            tcp_sock.bind((cfg.server.http.host, cfg.server.http.port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE and not cfg.server.http.strict_port:
                tcp_sock.bind((cfg.server.http.host, 0))
                port = tcp_sock.getsockname()[1]
                print(f"Port {cfg.server.http.port} in use, using {port}")
            else:
                raise
        else:
            port = cfg.server.http.port
        tcp_sock.listen(cfg.server.http.backlog)
        sockets.append(tcp_sock)
        print(
            f"MinerU server listening on UDS {uds_path} and TCP {cfg.server.http.host}:{port}"
        )
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
        # console
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(ch)

        # file
        log_path = os.path.expanduser(log_cfg.path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(fh)


if __name__ == "__main__":
    main()
