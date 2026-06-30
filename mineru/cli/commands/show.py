"""mineru show — show doclib resource details."""

from __future__ import annotations

import typer
from rich.table import Table

from ...doclib.client import DoclibClient
from ...doclib.types import DocInfo, FileInfoResponse, ParseInfo, ScanInfo
from ..contracts import CliContext
from ..path_utils import normalize_cli_path
from ..runtime import run_cli

app = typer.Typer(help="Show doclib resource details", no_args_is_help=True)


@app.command("parse")
def show_parse(
    parse_id: int = typer.Argument(..., help="Parse task id"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show one parse task."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: _client().get_parse(parse_id),
        render=_render_parse_info,
    )


@app.command("scan")
def show_scan(
    scan_id: int = typer.Argument(..., help="Scan task id"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show one scan task."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: _client().get_scan(scan_id),
        render=_render_scan,
    )


@app.command("file")
def show_file(
    path: str = typer.Argument(..., help="File path"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show file, doc, and parse state for a local path."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: _client().get_file_by_path(normalize_cli_path(path)),
        render=_render_file_info,
    )


@app.command("doc")
def show_doc(
    doc_ref: str = typer.Argument(..., help="Document Doc ID or SHA-256"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show one doc by Doc ID or content hash."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: _client().get_doc(doc_ref, expand_files=True),
        render=_render_doc_info,
    )


def _client() -> DoclibClient:
    return DoclibClient(timeout=30)


def _render_parse_info(data: ParseInfo) -> Table:
    table = Table(title=f"Parse {data.id}: {data.status}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("SHA-256", data.sha256)
    table.add_row("Tier", data.tier)
    table.add_row("Pages", data.page_range)
    table.add_row("Privacy", data.privacy)
    if data.error_code or data.error_msg:
        table.add_row("Error", f"{data.error_code or ''} {data.error_msg or ''}".rstrip())
    return table


def _render_scan(data: ScanInfo) -> Table | str:
    if data.status == "failed":
        return f"Scan failed: {data.error_code or ''} {data.error_msg or ''}"
    table = Table(title=f"Scan {data.id}: {data.status}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Seen", str(data.files_seen))
    table.add_row("Refreshed", str(data.files_refreshed))
    table.add_row("New", str(data.files_new))
    table.add_row("Changed", str(data.files_changed))
    table.add_row("Deleted", str(data.files_deleted))
    table.add_row("Unreachable", str(data.files_unreachable))
    table.add_row("Excluded", str(data.files_excluded))
    table.add_row("Unsupported", str(data.files_unsupported))
    return table


def _render_file_info(data: FileInfoResponse) -> Table | str:
    if not data.file:
        return "File not found in database."

    table = Table(title=f"File Info: {data.file.filename or '?'}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Path", data.file.path or "?")
    table.add_row("Type", data.file.ext or "?")
    table.add_row("Size", _format_info_bytes(data.file.size_bytes))
    table.add_row("Doc ID", data.file.short_id or "-")
    table.add_row("Page count", str(data.doc.page_count if data.doc else "?"))
    table.add_row("Title", (data.doc.title if data.doc else None) or "—")
    table.add_row("Author", (data.doc.author if data.doc else None) or "—")

    if data.parsed_tiers:
        tier_str = ", ".join(f"{tier.tier}={tier.status}" for tier in data.parsed_tiers)
        table.add_row("Tiers", tier_str)

    return table


def _format_info_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n} {unit}"
        n //= 1024
    return f"{n} TB"


def _render_doc_info(data: DocInfo) -> Table:
    table = Table(title=f"Doc {data.short_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("SHA-256", data.sha256)
    table.add_row("Type", data.file_type or "-")
    table.add_row("Title", data.title or "-")
    table.add_row("Pages", str(data.page_count if data.page_count is not None else "-"))
    table.add_row("Image based", str(data.is_image_based))
    if data.files:
        files = "\n".join(f"[{file_info.status}] {file_info.path}" for file_info in data.files)
        table.add_row("Files", files)
    return table
