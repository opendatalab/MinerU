"""mineru invalidate — mark parsed results as superseded."""

from __future__ import annotations

from pathlib import Path

import typer

from mineru.doclib.client import MineruClient
from mineru.cli_next.output import print_error, print_success


def invalidate_cmd(
    path: str = typer.Argument(..., help="Path to the document file"),
    tier: str = typer.Option(None, "--tier", help="Parse tier to invalidate (omit = all tiers)"),
) -> None:
    """Mark done parse results as superseded."""
    file_path = str(Path(path).resolve())

    if not Path(file_path).exists():
        print_error(f"File not found: {file_path}")
        raise typer.Exit(1)

    try:
        client = MineruClient(timeout=10)
    except Exception:
        print_error("Cannot connect to mineru server. Run 'mineru server start' first.")
        raise typer.Exit(1) from None

    try:
        result = client.invalidate(file_path, tier=tier)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    count = result.get("invalidated", 0)
    sha256 = result.get("sha256", "")[:12]
    tier_label = f" (tier={tier})" if tier else ""
    if count:
        print_success(f"Invalidated {count} batch(es) for {sha256}...{tier_label}. Use 'mineru parse' to re-parse.")
    else:
        print(f"No done batches found for {sha256}...{tier_label}.")
