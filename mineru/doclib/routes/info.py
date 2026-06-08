"""Info route."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

router = APIRouter(tags=["info"])


@router.get("/info")
async def info(request: Request, path: str = Query(...)) -> dict:
    state = request.state.app

    file_row = await state.db.fetchone(
        "SELECT * FROM files WHERE path=? AND scan_status='active'", (path,)
    )
    if file_row is None:
        return {"path": path, "found": False}

    sha = file_row["sha256"]
    doc = (
        await state.db.fetchone("SELECT * FROM docs WHERE sha256=?", (sha,))
        if sha
        else None
    )

    # aggregate parse batches
    parse_rows = (
        await state.db.fetchall(
            "SELECT * FROM parses WHERE sha256=? ORDER BY tier, created_at DESC", (sha,)
        )
        if sha
        else []
    )

    tiers: list[dict] = []
    for pr in parse_rows:
        tiers.append(
            {
                "tier": pr["tier"],
                "status": pr["status"],
                "pages": pr["pages"],
                "done_at": pr["done_at"],
            }
        )

    return {
        "path": file_row["path"],
        "filename": file_row["filename"],
        "ext": file_row["ext"],
        "size_bytes": file_row["size_bytes"],
        "mtime_ms": file_row["mtime_ms"],
        "sha256": sha,
        "scan_status": file_row["scan_status"],
        "title": doc["title"] if doc else None,
        "author": doc["author"] if doc else None,
        "page_count": doc["page_count"] if doc else None,
        "lang": doc["lang"] if doc else None,
        "is_encrypted": doc["is_encrypted"] if doc else 0,
        "tiers": tiers,
        "found": True,
    }
