"""Server routes: status and shutdown."""

from __future__ import annotations

import asyncio
import os
import signal
import time

from fastapi import APIRouter, Request

router = APIRouter(tags=["server"])


@router.get("/server/status")
async def server_status(request: Request):
    state = request.state.app

    files_total = await state.db.fetchone(
        "SELECT COUNT(*) as cnt FROM files WHERE scan_status='active'"
    )
    docs_total = await state.db.fetchone("SELECT COUNT(*) as cnt FROM docs")
    parse_q = await state.parse_svc.get_queue_length()
    reg_q = await state.db.fetchone(
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
        "reg_queue_length": reg_q["cnt"] if reg_q else 0,
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
async def shutdown(request: Request):
    """Gracefully shut down the server."""
    # Signal shutdown in a background task so we can return a response first
    import asyncio

    async def _shutdown():
        await asyncio.sleep(0.1)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(_shutdown())
    return {"message": "Server shutting down..."}
