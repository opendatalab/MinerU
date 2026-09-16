"""mineru show — show doclib resource details."""

from __future__ import annotations

import typer
from rich.table import Table

from ...doclib.client import DoclibClient
from ...doclib.types import DocInfo, FileInfoResponse, ParseInfo, ScanInfo
from ...utils.i18n import t
from ..contracts import CliContext
from ..path_utils import normalize_cli_path
from ..runtime import run_cli

app = typer.Typer(help=t("Show doclib resource details"), no_args_is_help=True)


@app.command("parse", help=t("Show one parse task."))
def show_parse(
    parse_id: int = typer.Argument(..., help=t("Parse task id")),
    json_mode: bool = typer.Option(False, "--json", help=t("JSON output")),
) -> None:
    """Show one parse task."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: _client().get_parse(parse_id),
        render=_render_parse_info,
    )


@app.command("scan", help=t("Show one scan task."))
def show_scan(
    scan_id: int = typer.Argument(..., help=t("Scan task id")),
    json_mode: bool = typer.Option(False, "--json", help=t("JSON output")),
) -> None:
    """Show one scan task."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: _client().get_scan(scan_id),
        render=_render_scan,
    )


@app.command("file", help=t("Show file, doc, and parse state for a local path."))
def show_file(
    path: str = typer.Argument(..., help=t("File path")),
    json_mode: bool = typer.Option(False, "--json", help=t("JSON output")),
) -> None:
    """Show file, doc, and parse state for a local path."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: _client().get_file_by_path(normalize_cli_path(path)),
        render=_render_file_info,
    )


@app.command("doc", help=t("Show one doc by Doc ID or content hash."))
def show_doc(
    doc_ref: str = typer.Argument(..., help=t("Document Doc ID or SHA-256")),
    json_mode: bool = typer.Option(False, "--json", help=t("JSON output")),
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
    table = Table(title=t("Parse {id}: {status}", id=data.id, status=data.status))
    table.add_column(t("Field"), style="cyan")
    table.add_column(t("Value"), style="green")
    table.add_row(t("SHA-256"), data.sha256)
    table.add_row(t("Tier"), data.tier)
    table.add_row(t("Pages"), data.page_range)
    table.add_row(t("Privacy"), data.privacy)
    if data.error_code or data.error_msg:
        table.add_row(t("Error"), f"{data.error_code or ''} {data.error_msg or ''}".rstrip())
    return table


def _render_scan(data: ScanInfo) -> Table | str:
    if data.status == "failed":
        return t("Scan failed: {code} {msg}", code=data.error_code or "", msg=data.error_msg or "")
    table = Table(title=t("Scan {id}: {status}", id=data.id, status=data.status))
    table.add_column(t("Metric"), style="cyan")
    table.add_column(t("Value"), style="green", justify="right")
    table.add_row(t("Seen"), str(data.files_seen))
    table.add_row(t("Refreshed"), str(data.files_refreshed))
    table.add_row(t("New"), str(data.files_new))
    table.add_row(t("Changed"), str(data.files_changed))
    table.add_row(t("Deleted"), str(data.files_deleted))
    table.add_row(t("Unreachable"), str(data.files_unreachable))
    table.add_row(t("Excluded"), str(data.files_excluded))
    table.add_row(t("Unsupported"), str(data.files_unsupported))
    return table


def _render_file_info(data: FileInfoResponse) -> Table | str:
    if not data.file:
        return t("File not found in database.")

    table = Table(title=t("File Info: {name}", name=data.file.filename or "?"))
    table.add_column(t("Field"), style="cyan")
    table.add_column(t("Value"), style="green")
    table.add_row(t("Path"), data.file.path or "?")
    table.add_row(t("Type"), data.file.ext or "?")
    table.add_row(t("Size"), _format_info_bytes(data.file.size_bytes))
    table.add_row(t("Doc ID"), data.file.short_id or "-")
    table.add_row(t("Page count"), str(data.doc.page_count if data.doc else "?"))
    table.add_row(t("Title"), (data.doc.title if data.doc else None) or "—")
    table.add_row(t("Author"), (data.doc.author if data.doc else None) or "—")

    if data.parsed_tiers:
        tier_str = ", ".join(f"{tier.tier}={tier.status}" for tier in data.parsed_tiers)
        table.add_row(t("Tiers"), tier_str)

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
    table = Table(title=t("Doc {short_id}", short_id=data.short_id))
    table.add_column(t("Field"), style="cyan")
    table.add_column(t("Value"), style="green")
    table.add_row(t("SHA-256"), data.sha256)
    table.add_row(t("Type"), data.file_type or "-")
    table.add_row(t("Title"), data.title or "-")
    table.add_row(t("Pages"), str(data.page_count if data.page_count is not None else "-"))
    table.add_row(t("Image based"), str(data.is_image_based))
    if data.files:
        files = "\n".join(f"[{file_info.status}] {file_info.path}" for file_info in data.files)
        table.add_row(t("Files"), files)
    return table
