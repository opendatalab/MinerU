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


@router.get("/parse/content")
async def parse_content(
    request: Request,
    sha256: str = Query(...),
    tier: str = Query(...),
    output: str | None = Query(None),
) -> dict:
    """Read parsed markdown content from saved .md files."""
    state = request.state.app
    data_dir = getattr(state, "data_dir", os.path.expanduser("~/MinerU"))
    tier_dir = os.path.join(data_dir, "parsed", sha256[:2], sha256, tier)

    if not os.path.isdir(tier_dir):
        return {
            "sha256": sha256,
            "tier": tier,
            "content": None,
            "error": "Content not found",
        }

    # merge all .md files, sorted by filename
    parts: list[str] = []
    for fname in sorted(os.listdir(tier_dir)):
        if not fname.endswith(".md"):
            continue
        try:
            with open(os.path.join(tier_dir, fname), encoding="utf-8") as f:
                parts.append(f.read())
        except Exception:
            pass

    if not parts:
        return {
            "sha256": sha256,
            "tier": tier,
            "content": None,
            "error": "Content not found",
        }

    content = "\n".join(parts)

    if output and output != "-":
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        return {"sha256": sha256, "tier": tier, "output": os.path.abspath(output)}

    return {"sha256": sha256, "tier": tier, "content": content}
