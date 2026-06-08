"""Cleanup routes."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

router = APIRouter(tags=["cleanup"])


@router.post("/cleanup/deleted")
async def cleanup_deleted(
    request: Request,
    older_than_days: int = Query(30, ge=1),
    dry_run: bool = Query(True),
) -> dict:
    state = request.state.app
    count = await state.cleanup_svc.cleanup_deleted(
        older_than_days=older_than_days,
        dry_run=dry_run,
    )
    return {"deleted_files": count, "dry_run": dry_run}


@router.post("/cleanup/orphans")
async def cleanup_orphans(
    request: Request,
    dry_run: bool = Query(True),
) -> dict:
    state = request.state.app
    count = await state.cleanup_svc.cleanup_orphans(dry_run=dry_run)
    return {"orphan_docs": count, "dry_run": dry_run}


@router.post("/cleanup/temp")
async def cleanup_temp(
    request: Request,
    older_than_days: int = Query(7, ge=1),
) -> dict:
    state = request.state.app
    count = await state.cleanup_svc.cleanup_temp_files(older_than_days=older_than_days)
    return {"temp_files_removed": count}
