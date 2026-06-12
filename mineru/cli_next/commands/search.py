"""mineru search / find / info commands."""

from __future__ import annotations

import typer

from ...doclib.client import DoclibClient
from ..output import format_info, format_search_results, print_error


def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    file_type: str = typer.Option(None, "--type", help="File type filter (pdf, docx, ...)"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    offset: int = typer.Option(0, "--offset", help="Result offset"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Search parsed document content."""
    try:
        client = DoclibClient(timeout=10)
        result = client.search(query, file_type=file_type, limit=limit, offset=offset)
        format_search_results(result, json_mode=json_mode)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


def find_cmd(
    query: str = typer.Argument(..., help="Filename search query"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Search filenames only (not document content)."""
    try:
        client = DoclibClient(timeout=10)
        result = client.find(query, limit=limit)
        format_search_results(result, json_mode=json_mode)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


def info_cmd(
    path: str = typer.Argument(..., help="File path"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show file/document details."""
    from pathlib import Path

    file_path = str(Path(path).resolve())
    try:
        client = DoclibClient(timeout=10)
        result = client.get_file_info(file_path)
        format_info(result, json_mode=json_mode)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None
