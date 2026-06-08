"""Server routes: status and shutdown."""

from __future__ import annotations

import os
import signal
import time

from fastapi import APIRouter, Request

router = APIRouter(tags=["server"])


@router.get("/server/status")
async def server_status(request: Request) -> dict:
    state = request.state.app

    files_total = await state.db.fetchone(
        "SELECT COUNT(*) as cnt FROM files WHERE scan_status='active'"
    )
    docs_total = await state.db.fetchone("SELECT COUNT(*) as cnt FROM docs")
    parse_q = await state.parse_svc.get_queue_length()
    ingest_q = await state.db.fetchone(
        "SELECT COUNT(*) as cnt FROM files WHERE sha256 IS NULL AND scan_status='active'"
    )
    watches = await state.config_svc.list_watches()
    uptime = time.time() - state.start_time if hasattr(state, "start_time") else 0

    return {
        "running": True,
        "pid": state.pid,
        "uptime_seconds": uptime,
        "socket_path": getattr(state, "socket_path", ""),
        "data_dir": getattr(state, "data_dir", ""),
        "files_total": files_total["cnt"] if files_total else 0,
        "docs_total": docs_total["cnt"] if docs_total else 0,
        "parse_queue_length": parse_q,
        "ingest_queue_length": ingest_q["cnt"] if ingest_q else 0,
        "parse_server": _build_parse_server_status(),
        "watch_count": len(watches),
        "watches": [
            {
                "id": w["id"],
                "path": w["path"],
                "removable": bool(w["removable"]),
                "watch_status": w["watch_status"],
            }
            for w in watches
        ],
    }


@router.post("/shutdown")
async def shutdown(request: Request) -> dict:
    """Gracefully shut down the server."""
    # Signal shutdown in a background task so we can return a response first
    import asyncio

    async def _shutdown() -> None:
        await asyncio.sleep(0.1)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(_shutdown())
    return {"message": "Server shutting down..."}


def _build_parse_server_status() -> dict:
    from mineru.doclib.background.parse_server_health import get_health

    health = get_health()
    return {
        "local": {
            "mode": health.local_mode,
            "healthy": health.local_healthy,
            "supported_tiers": health.local_supported_tiers,
        },
        "remote": {
            "healthy": health.remote_healthy,
            "supported_tiers": health.remote_supported_tiers,
        },
    }
