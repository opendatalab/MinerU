"""mineru list — list doclib resource collections."""

from __future__ import annotations

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import FileStatus, ParseStatus, ScanKind, ScanStatus
from ...types import Tier
from ..json_errors import exit_with_error
from ..output import print_error, print_info, print_json

app = typer.Typer(help="List doclib resources", no_args_is_help=True)


@app.command("parses")
def list_parses(
    status: ParseStatus | None = typer.Option(None, "--status", help="Parse status filter"),
    tier: Tier | None = typer.Option(None, "--tier", help="Parse tier filter"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max rows"),
    offset: int = typer.Option(0, "--offset", help="Result offset"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List parse tasks."""
    try:
        result = _client().list_parses(status=status, tier=tier, limit=limit, offset=offset)
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)

    if json_mode:
        print_json(result)
        return
    if not result.parses:
        print_info("No parses found.")
        return
    print(f"Parses ({result.total} total)")
    for item in result.parses:
        print(f"  [{item.id}] {item.status} {item.tier} pages={item.page_range} sha256={item.sha256[:12]}")


@app.command("scans")
def list_scans(
    status: ScanStatus | None = typer.Option(None, "--status", help="Scan status filter"),
    kind: ScanKind | None = typer.Option(None, "--kind", help="Scan kind filter"),
    watch_id: int | None = typer.Option(None, "--watch-id", help="Watch id filter"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max rows"),
    offset: int = typer.Option(0, "--offset", help="Result offset"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List scan tasks."""
    try:
        result = _client().list_scans(status=status, kind=kind, watch_id=watch_id, limit=limit, offset=offset)
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)

    if json_mode:
        print_json(result)
        return
    if not result.scans:
        print_info("No scans found.")
        return
    print(f"Scans ({result.total} total)")
    for item in result.scans:
        print(
            f"  [{item.id}] {item.status} {item.kind} {item.path} "
            f"seen={item.files_seen} refreshed={item.files_refreshed} errors={item.files_error}"
        )


@app.command("files")
def list_files(
    status: FileStatus | None = typer.Option(None, "--status", help="File status filter"),
    ext: str | None = typer.Option(None, "--ext", help="File extension filter, e.g. pdf"),
    watch_id: int | None = typer.Option(None, "--watch-id", help="Watch id filter"),
    limit: int = typer.Option(200, "--limit", "-n", help="Max rows"),
    offset: int = typer.Option(0, "--offset", help="Result offset"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List file path records."""
    try:
        result = _client().list_files(status=status, ext=ext, watch_id=watch_id, limit=limit, offset=offset)
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)

    if json_mode:
        print_json(result)
        return
    if not result.files:
        print_info("No files found.")
        return
    print(f"Files ({result.total} total)")
    for item in result.files:
        sha = item.sha256[:12] if item.sha256 else "-"
        print(f"  [{item.status}] {item.path} ext={item.ext} sha256={sha}")


@app.command("docs")
def list_docs(
    file_type: str | None = typer.Option(None, "--file-type", help="Document file type filter, e.g. pdf"),
    limit: int = typer.Option(200, "--limit", "-n", help="Max rows"),
    offset: int = typer.Option(0, "--offset", help="Result offset"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List active docs."""
    try:
        result = _client().list_docs(file_type=file_type, limit=limit, offset=offset)
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)

    if json_mode:
        print_json(result)
        return
    if not result.docs:
        print_info("No docs found.")
        return
    print(f"Docs ({result.total} total)")
    for item in result.docs:
        title = item.title or "-"
        print(f"  {item.sha256[:12]} type={item.file_type or '-'} pages={item.page_count or '-'} title={title}")


def _client() -> DoclibClient:
    return DoclibClient(timeout=30)
