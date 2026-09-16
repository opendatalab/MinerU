"""mineru list — list doclib resource collections."""

from __future__ import annotations

import typer
from rich.table import Table

from ...doclib.client import DoclibClient
from ...doclib.types import (
    FileStatus,
    ListDocsResponse,
    ListFilesResponse,
    ListParsesResponse,
    ParseStatus,
    ScanKind,
    ScanListResponse,
    ScanStatus,
)
from ...types import Tier
from ...utils.i18n import t
from ..contracts import CliContext
from ..runtime import run_cli

app = typer.Typer(help=t("List doclib resources"), no_args_is_help=True)


@app.command("parses", help=t("List parse tasks."))
def list_parses(
    status: ParseStatus | None = typer.Option(None, "--status", help=t("Parse status filter")),
    tier: Tier | None = typer.Option(None, "--tier", help=t("Parse tier filter")),
    limit: int = typer.Option(50, "--limit", "-n", help=t("Max rows")),
    offset: int = typer.Option(0, "--offset", help=t("Result offset")),
    json_mode: bool = typer.Option(False, "--json", help=t("JSON output")),
) -> None:
    """List parse tasks."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: _client().list_parses(status=status, tier=tier, limit=limit, offset=offset),
        render=_render_list_parses,
    )


@app.command("scans", help=t("List scan tasks."))
def list_scans(
    status: ScanStatus | None = typer.Option(None, "--status", help=t("Scan status filter")),
    kind: ScanKind | None = typer.Option(None, "--kind", help=t("Scan kind filter")),
    watch_id: int | None = typer.Option(None, "--watch-id", help=t("Watch id filter")),
    limit: int = typer.Option(50, "--limit", "-n", help=t("Max rows")),
    offset: int = typer.Option(0, "--offset", help=t("Result offset")),
    json_mode: bool = typer.Option(False, "--json", help=t("JSON output")),
) -> None:
    """List scan tasks."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: _client().list_scans(status=status, kind=kind, watch_id=watch_id, limit=limit, offset=offset),
        render=_render_list_scans,
    )


@app.command("files", help=t("List file path records."))
def list_files(
    status: FileStatus | None = typer.Option(None, "--status", help=t("File status filter")),
    ext: str | None = typer.Option(None, "--ext", help=t("File extension filter, e.g. pdf")),
    watch_id: int | None = typer.Option(None, "--watch-id", help=t("Watch id filter")),
    limit: int = typer.Option(200, "--limit", "-n", help=t("Max rows")),
    offset: int = typer.Option(0, "--offset", help=t("Result offset")),
    json_mode: bool = typer.Option(False, "--json", help=t("JSON output")),
) -> None:
    """List file path records."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: _client().list_files(status=status, ext=ext, watch_id=watch_id, limit=limit, offset=offset),
        render=_render_list_files,
    )


@app.command("docs", help=t("List active docs."))
def list_docs(
    file_type: str | None = typer.Option(None, "--file-type", help=t("Document file type filter, e.g. pdf")),
    limit: int = typer.Option(200, "--limit", "-n", help=t("Max rows")),
    offset: int = typer.Option(0, "--offset", help=t("Result offset")),
    json_mode: bool = typer.Option(False, "--json", help=t("JSON output")),
) -> None:
    """List active docs."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: _client().list_docs(file_type=file_type, limit=limit, offset=offset),
        render=_render_list_docs,
    )


def _client() -> DoclibClient:
    return DoclibClient(timeout=30)


def _render_list_parses(data: ListParsesResponse) -> Table | str:
    if not data.parses:
        return t("No parses found.")
    table = Table(title=t("Parses ({total} total)", total=data.total))
    table.add_column(t("ID"), justify="right")
    table.add_column(t("Status"), style="green")
    table.add_column(t("Tier"), style="cyan")
    table.add_column(t("Pages"))
    table.add_column(t("Doc ID"))
    for item in data.parses:
        table.add_row(str(item.id), item.status, item.tier, item.page_range, item.short_id)
    return table


def _render_list_scans(data: ScanListResponse) -> Table | str:
    if not data.scans:
        return t("No scans found.")
    table = Table(title=t("Scans ({total} total)", total=data.total))
    table.add_column(t("ID"), justify="right")
    table.add_column(t("Status"), style="green")
    table.add_column(t("Kind"), style="cyan")
    table.add_column(t("Path"))
    table.add_column(t("Seen"), justify="right")
    table.add_column(t("Refreshed"), justify="right")
    table.add_column(t("Errors"), justify="right")
    for item in data.scans:
        table.add_row(
            str(item.id),
            item.status,
            item.kind,
            item.path,
            str(item.files_seen),
            str(item.files_refreshed),
            str(item.files_error),
        )
    return table


def _render_list_files(data: ListFilesResponse) -> Table | str:
    if not data.files:
        return t("No files found.")
    table = Table(title=t("Files ({total} total)", total=data.total))
    table.add_column(t("Status"), style="green")
    table.add_column(t("Path"))
    table.add_column(t("Ext"), style="cyan")
    table.add_column(t("Doc ID"))
    for item in data.files:
        table.add_row(item.status, item.path, item.ext, item.short_id or "-")
    return table


def _render_list_docs(data: ListDocsResponse) -> Table | str:
    if not data.docs:
        return t("No docs found.")
    table = Table(title=t("Docs ({total} total)", total=data.total))
    table.add_column(t("Doc ID"))
    table.add_column(t("Type"), style="cyan")
    table.add_column(t("Pages"), justify="right")
    table.add_column(t("Title"))
    for item in data.docs:
        title = item.title or "-"
        table.add_row(item.short_id, item.file_type or "-", str(item.page_count or "-"), title)
    return table
