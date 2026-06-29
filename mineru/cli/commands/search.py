"""mineru search / find commands."""

from __future__ import annotations

import typer

from ...doclib.client import DoclibClient
from ...types import Tier
from ..json_errors import exit_with_error
from ..output import format_find_results, format_search_results


def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    file_type: str | None = typer.Option(
        None,
        "--type",
        help="File type filter: pdf, image, docx, pptx, xlsx, html, markdown, csv, rst, tex, txt",
    ),
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
        exit_with_error(exc, json_mode=json_mode)


def find_cmd(
    query: str = typer.Argument(..., help="Filename search query"),
    ext: str | None = typer.Option(
        None,
        "--ext",
        help="File extension filter: pdf, png, jpg, jpeg, jp2, webp, gif, bmp, tiff, docx, pptx, xlsx, html, htm, md, markdown, csv, rst, tex, txt",
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Search filenames only (not document content)."""
    try:
        client = DoclibClient(timeout=10)
        result = client.find(query, ext=ext, limit=limit)
        format_find_results(result, json_mode=json_mode)
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)
