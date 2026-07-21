"""mineru search / find commands."""

from __future__ import annotations

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import FindResponse, SearchResponse
from ...filetypes import FILE_TYPE_BY_EXTENSION
from ...types import Tier
from ..contracts import CliContext
from ..runtime import run_cli

FILE_TYPES = ", ".join(dict.fromkeys(FILE_TYPE_BY_EXTENSION.values()))
FILE_EXTS = ", ".join(FILE_TYPE_BY_EXTENSION)


def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    file_type: str | None = typer.Option(None, "--type", help=f"File type filter: {FILE_TYPES}"),
    tier: Tier | None = typer.Option(None, "--tier", help="Exact search index tier: flash, basic, standard, advanced"),
    min_tier: Tier | None = typer.Option(
        None,
        "--min-tier",
        help="Minimum search index tier: flash, basic, standard, advanced",
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    offset: int = typer.Option(0, "--offset", help="Result offset"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Search parsed document content."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: DoclibClient(timeout=10).search(
            query,
            file_type=file_type,
            tier=tier,
            min_tier=min_tier,
            limit=limit,
            offset=offset,
        ),
        render=_render_search_results,
    )


def find_cmd(
    query: str = typer.Argument(..., help="Filename search query"),
    ext: str | None = typer.Option(None, "--ext", help=f"File extension filter: {FILE_EXTS}"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Search filenames only (not document content)."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: DoclibClient(timeout=10).find(query, ext=ext, limit=limit),
        render=_render_find_results,
    )


def _render_search_results(data: SearchResponse) -> str:
    if not data.results:
        return "No results found."

    lines = [f"Search results ({data.total} total)"]
    for index, result in enumerate(data.results, start=1):
        label = result.title or f"Document {result.short_id}"
        item_line = f"{index}. {label}"
        if result.tier:
            item_line += f" Tier: {result.tier}"
        lines.append(item_line)
        active_files = [file for file in result.files if file.status == "active"]
        display_files = active_files or result.files
        if display_files:
            lines.append("   Files:")
            for file in display_files:
                suffix = f" ({file.status})" if file.status != "active" else ""
                lines.append(f"   {file.path}{suffix}")
        else:
            lines.append("   File no longer exists.")
        snippet = _format_snippet(result.snippet)
        if snippet:
            lines.append(f"   {snippet}")
        if index < len(data.results):
            lines.append("")
    return "\n".join(lines)


def _render_find_results(data: FindResponse) -> str:
    if not data.results:
        return "No results found."

    lines = [f"Search results ({data.total} total)"]
    for index, result in enumerate(data.results, start=1):
        path = _format_result_path(result.paths)
        lines.append(f"{index}. {result.filename}{path}")
        if index < len(data.results):
            lines.append("")
    return "\n".join(lines)


def _format_result_path(paths: list[str]) -> str:
    if not paths:
        return ""
    return f" ({paths[0]})"


def _format_snippet(snippet: str | None) -> str:
    return str(snippet or "").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "  ")
