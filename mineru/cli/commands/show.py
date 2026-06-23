"""mineru show — show doclib resource details."""

from __future__ import annotations

import typer

from ...doclib.client import DoclibClient
from ..json_errors import exit_with_error
from ..output import format_info, print_json
from ..path_utils import normalize_cli_path
from .scan import _print_scan

app = typer.Typer(help="Show doclib resource details", no_args_is_help=True)


@app.command("parse")
def show_parse(
    parse_id: int = typer.Argument(..., help="Parse task id"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show one parse task."""
    try:
        result = _client().get_parse(parse_id)
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)

    if json_mode:
        print_json(result)
        return
    print(f"Parse {result.id}: {result.status}")
    print(f"  sha256: {result.sha256}")
    print(f"  tier: {result.tier}")
    print(f"  pages: {result.page_range}")
    print(f"  privacy: {result.privacy}")
    if result.error_code or result.error_msg:
        print(f"  error: {result.error_code or ''} {result.error_msg or ''}".rstrip())


@app.command("scan")
def show_scan(
    scan_id: int = typer.Argument(..., help="Scan task id"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show one scan task."""
    try:
        result = _client().get_scan(scan_id)
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)

    _print_scan(result, json_mode=json_mode)


@app.command("file")
def show_file(
    path: str = typer.Argument(..., help="File path"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show file, doc, and parse state for a local path."""
    file_path = normalize_cli_path(path)
    try:
        result = _client().get_file_by_path(file_path)
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)

    format_info(result, json_mode=json_mode)


@app.command("doc")
def show_doc(
    sha256: str = typer.Argument(..., help="Document sha256"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show one doc by content hash."""
    try:
        result = _client().get_doc(sha256, expand_files=True)
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)

    if json_mode:
        print_json(result)
        return
    print(f"Doc {result.sha256}")
    print(f"  type: {result.file_type or '-'}")
    print(f"  title: {result.title or '-'}")
    print(f"  pages: {result.page_count if result.page_count is not None else '-'}")
    print(f"  image_based: {result.is_image_based}")
    files = result.files or []
    if files:
        print("  files:")
        for file_info in files:
            print(f"    [{file_info.status}] {file_info.path}")


def _client() -> DoclibClient:
    return DoclibClient(timeout=30)
