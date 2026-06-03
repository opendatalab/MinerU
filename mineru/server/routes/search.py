"""Search routes."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

router = APIRouter(tags=["search"])


@router.get("/search")
async def search(
    request: Request,
    q: str = Query(..., description="Search query"),
    type: str | None = Query(None),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    state = request.state.app
    results, total = await state.search_svc.search(
        query=q, file_type=type, limit=limit, offset=offset,
    )
    return {"results": results, "total": total, "query": q}


@router.get("/find")
async def find(
    request: Request,
    q: str = Query(..., description="Filename search query"),
    limit: int = Query(50, ge=1, le=500),
):
    state = request.state.app
    results, total = await state.search_svc.search_filenames(query=q, limit=limit)
    return {"results": results, "total": total, "query": q}
