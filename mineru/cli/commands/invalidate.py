"""mineru invalidate — mark parsed results as superseded."""

from __future__ import annotations

from pathlib import Path

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import InvalidateRequest
from ...types import Tier
from ..output import print_error, print_success
from ..path_utils import normalize_cli_path


def invalidate_cmd(
    path: str = typer.Argument(..., help="Path to the document file"),
    tier: Tier | None = typer.Option(None, "--tier", help="Parse tier to invalidate (omit = all tiers)"),
) -> None:
    """Mark done parse results as superseded."""
    file_path = normalize_cli_path(path)

    if not Path(file_path).exists():
        print_error(f"File not found: {file_path}")
        raise typer.Exit(1)

    try:
        client = DoclibClient(timeout=10)
    except Exception:
        print_error("Cannot connect to mineru server. Run 'mineru server start' first.")
        raise typer.Exit(1) from None

    try:
        result = client.invalidate(InvalidateRequest(path=file_path, tier=tier))
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    count = result.invalidated_count
    sha256 = result.sha256[:12]
    tier_label = f" (tier={tier})" if tier else ""
    if count:
        print_success(f"Invalidated {count} batch(es) for {sha256}...{tier_label}. Use 'mineru parse' to re-parse.")
    else:
        print(f"No done batches found for {sha256}...{tier_label}.")
