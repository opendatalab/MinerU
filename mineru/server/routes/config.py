"""Config routes: watch, exclude, parsing-rules, global config."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel


class WatchRequest(BaseModel):
    path: str
    removable: bool = False
    label: str | None = None


class RuleRequest(BaseModel):
    pattern: str
    name: str | None = None
    tier: str | None = None
    pages: str | None = None
    remote: bool = False
    priority: int = 0


router = APIRouter(tags=["config"])


# ── global config ────────────────────────────────────────────────

@router.get("/config")
async def get_config(request: Request):
    state = request.state.app
    cfg = await state.config_svc.get_all()
    watches = await state.config_svc.list_watches()
    rules = await state.config_svc.list_rules()
    return {"config": cfg, "watches": watches, "rules": rules}


# ── watch ────────────────────────────────────────────────────────

@router.get("/config/watch")
async def list_watches(request: Request):
    state = request.state.app
    watches = await state.config_svc.list_watches()
    return {"watches": watches}


@router.post("/config/watch")
async def add_watch(req: WatchRequest, request: Request):
    state = request.state.app
    w = await state.config_svc.add_watch(
        req.path, removable=req.removable, label=req.label,
    )
    return w


@router.delete("/config/watch")
async def remove_watch(request: Request, path: str = Query(...)):
    state = request.state.app
    await state.config_svc.remove_watch(path)
    return {"message": f"Watch removed: {path}"}


# ── exclude ──────────────────────────────────────────────────────

@router.get("/config/exclude")
async def list_excludes(request: Request):
    state = request.state.app
    rules = await state.config_svc.list_rules("exclude")
    return {"rules": rules}


@router.post("/config/exclude")
async def add_exclude(req: RuleRequest, request: Request):
    state = request.state.app
    rid = await state.config_svc.add_rule(
        req.name or "用户规则", "exclude", req.pattern, priority=req.priority,
    )
    return {"id": rid}


@router.delete("/config/exclude/{rule_id}")
async def remove_exclude(rule_id: int, request: Request):
    state = request.state.app
    await state.config_svc.remove_rule(rule_id)
    return {"message": f"Exclude rule {rule_id} removed"}


# ── parsing-rules ────────────────────────────────────────────────

@router.get("/config/parsing-rules")
async def list_parsing_rules(request: Request):
    state = request.state.app
    rules = await state.config_svc.list_rules("parsing_rule")
    return {"rules": rules}


@router.post("/config/parsing-rules")
async def add_parsing_rule(req: RuleRequest, request: Request):
    state = request.state.app
    rid = await state.config_svc.add_rule(
        req.name or "规则", "parsing_rule", req.pattern,
        tier=req.tier, pages=req.pages, remote=req.remote, priority=req.priority,
    )
    return {"id": rid}


@router.delete("/config/parsing-rules/{rule_id}")
async def remove_parsing_rule(rule_id: int, request: Request):
    state = request.state.app
    await state.config_svc.remove_rule(rule_id)
    return {"message": f"Parsing rule {rule_id} removed"}
