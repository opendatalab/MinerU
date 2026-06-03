"""Parse routes."""

from __future__ import annotations

import os

from fastapi import APIRouter, Query, Request

from ..types import ParseRequest

router = APIRouter(tags=["parse"])


@router.post("/parse")
async def parse(req: ParseRequest, request: Request):
    state = request.state.app
    result = await state.parse_svc.request_parse(
        req.path, tier=req.tier, pages=req.pages, force=req.force,
    )
    return result


@router.get("/parse/status")
async def parse_status(
    request: Request,
    sha256: str = Query(...),
    tier: str = Query(...),
):
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
):
    """Read parsed markdown content from per-batch JSON files."""
    state = request.state.app
    data_dir = getattr(state, "data_dir", os.path.expanduser("~/MinerU"))
    tier_dir = os.path.join(data_dir, "parsed", sha256[:2], sha256, tier)

    if not os.path.isdir(tier_dir):
        return {"sha256": sha256, "tier": tier, "content": None, "error": "Content not found"}

    import json as _json

    # merge all JSON files by page_idx
    pages_by_idx: dict[int, dict] = {}
    for fname in sorted(os.listdir(tier_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(tier_dir, fname), encoding="utf-8") as f:
                data = _json.load(f)
            for p in data.get("pdf_info", []):
                pages_by_idx[p["page_idx"]] = p
        except Exception:
            pass

    if not pages_by_idx:
        return {"sha256": sha256, "tier": tier, "content": None, "error": "Content not found"}

    sorted_pages = [pages_by_idx[i] for i in sorted(pages_by_idx)]
    content = _markdown_from_pages(sorted_pages)

    if output and output != "-":
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        return {"sha256": sha256, "tier": tier, "output": os.path.abspath(output)}

    return {"sha256": sha256, "tier": tier, "content": content}


def _markdown_from_pages(pages: list[dict]) -> str:
    """Generate markdown from page dicts — iterate blocks/lines/spans."""
    parts: list[str] = []
    for page in pages:
        for block in page.get("para_blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("content"):
                        parts.append(span["content"])
    return "\n\n".join(parts)
