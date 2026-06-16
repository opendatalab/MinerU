"""mineru search / find commands."""

from __future__ import annotations

import typer

from ...doclib.client import DoclibClient
from ...types import Tier
from ..output import format_search_results, print_error


def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    file_type: str | None = typer.Option(None, "--type", help="File type filter (pdf, docx, ...)"),
    tier: Tier | None = typer.Option(None, "--tier", help="Exact search index tier: flash, standard, pro"),
    min_tier: Tier | None = typer.Option(None, "--min-tier", help="Minimum search index tier: flash, standard, pro"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    offset: int = typer.Option(0, "--offset", help="Result offset"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Search parsed document content."""
    try:
        client = DoclibClient(timeout=10)
        result = client.search(query, file_type=file_type, tier=tier, min_tier=min_tier, limit=limit, offset=offset)
        format_search_results(result, json_mode=json_mode)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


def find_cmd(
    query: str = typer.Argument(..., help="Filename search query"),
    ext: str | None = typer.Option(None, "--ext", help="File extension filter, e.g. pdf"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Search filenames only (not document content)."""
    try:
        client = DoclibClient(timeout=10)
        result = client.find(query, ext=ext, limit=limit)
        format_search_results(result, json_mode=json_mode)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None
