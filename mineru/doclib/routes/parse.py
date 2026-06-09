"""Parse routes."""

from __future__ import annotations

import os

from fastapi import APIRouter, Query, Request

from ..types import ParseRequest

router = APIRouter(tags=["parse"])


@router.post("/parse")
async def parse(req: ParseRequest, request: Request) -> dict:
    state = request.state.app
    result = await state.parse_svc.request_parse(
        req.path,
        tier=req.tier,
        pages=req.pages,
        force=req.force,
        remote=req.remote,
        remote_url=req.remote_url,
    )
    return result


@router.get("/parse/status")
async def parse_status(
    request: Request,
    sha256: str = Query(...),
    tier: str = Query(...),
) -> dict:
    state = request.state.app
    result = await state.parse_svc.get_parse_status(sha256, tier)
    if result is None:
        return {"sha256": sha256, "tier": tier, "status": "not_found"}
    return result


@router.get("/parse/invalidate")
async def parse_invalidate(
    request: Request,
    path: str = Query(..., description="File path"),
    tier: str | None = Query(None, description="Parse tier to invalidate (omit = all tiers)"),
) -> dict:
    """Mark done parse results as superseded."""
    state = request.state.app
    sha256 = await _resolve_sha256(state, path)
    if sha256 is None:
        return {"sha256": "", "tier": tier or "", "invalidated": 0, "error": "File not found or not ingested"}
    count = await state.parse_svc.invalidate(sha256, tier)
    return {"sha256": sha256, "tier": tier, "invalidated": count}


async def _resolve_sha256(state, path: str) -> str | None:
    row = await state.db.fetchone(
        "SELECT sha256 FROM files WHERE path=? AND scan_status='active'", (path,)
    )
    if row and row["sha256"]:
        return row["sha256"]
    return None


@router.get("/parse/content")
async def parse_content(
    request: Request,
    sha256: str = Query(...),
    tier: str = Query(...),
    output: str | None = Query(None),
    no_marker: bool = Query(False, description="Omit page markers from markdown output"),
) -> dict:
    """Generate markdown on the fly from saved JSON results."""
    state = request.state.app
    data_dir = getattr(state, "data_dir", os.path.expanduser("~/MinerU"))
    tier_dir = os.path.join(data_dir, "parsed", sha256[:2], sha256, tier)

    if not os.path.isdir(tier_dir):
        return {"sha256": sha256, "tier": tier, "content": None, "error": "Content not found"}

    # read JSON files from done batches (DB-driven, no directory listing)
    import json as _json

    from mineru.render import render_markdown
    from mineru.types import PageInfo
    from mineru.doclib.services.parse_svc import _safe_filename

    done_rows = await state.db.fetchall(
        "SELECT pages, done_at FROM parses WHERE sha256=? AND tier=? AND status='done' ORDER BY done_at DESC",
        (sha256, tier),
    )
    all_pages: list[PageInfo] = []
    for row in done_rows:
        key = _safe_filename(row["pages"], row["done_at"])
        fpath = os.path.join(tier_dir, f"{key}.json")
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                data = _json.load(f)
            for raw in data.get("pages", []):
                all_pages.append(PageInfo.from_dict(raw))
        except Exception:
            pass

    if not all_pages:
        return {"sha256": sha256, "tier": tier, "content": None, "error": "Content not found"}

    content = render_markdown(all_pages, add_markers=not no_marker)

    if output and output != "-":
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        return {"sha256": sha256, "tier": tier, "output": os.path.abspath(output)}

    return {"sha256": sha256, "tier": tier, "content": content}
