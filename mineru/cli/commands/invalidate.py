"""mineru invalidate — mark parsed results as superseded."""

from __future__ import annotations

from pathlib import Path

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import InvalidateRequest, InvalidateResponse
from ...errors import MineruError
from ...types import Tier
from ..path_utils import normalize_cli_path
from ..contracts import CliContext
from ..runtime import run_cli


def invalidate_cmd(
    path: str = typer.Argument(..., help="Path to the document file"),
    tier: Tier | None = typer.Option(None, "--tier", help="Parse tier to invalidate (omit = all tiers)"),
) -> None:
    """Mark done parse results as superseded."""
    ctx = CliContext(json_mode=False)
    run_cli(ctx, lambda: _invalidate(path, tier=tier), render=_render_invalidate)


def _invalidate(path: str, *, tier: Tier | None) -> InvalidateResponse:
    file_path = normalize_cli_path(path)

    if not Path(file_path).exists():
        raise MineruError("file_not_found", f"File not found: {file_path}", "path")
    return DoclibClient(timeout=10).invalidate(InvalidateRequest(path=file_path, tier=tier))


def _render_invalidate(data: InvalidateResponse) -> str:
    count = data.invalidated_count
    doc_id = data.short_id
    tier_label = f" (tier={data.tier})" if data.tier else ""
    if count:
        return f"Invalidated {count} batch(es) for {doc_id}{tier_label}. Use 'mineru parse' to re-parse."
    return f"No done batches found for {doc_id}{tier_label}."
