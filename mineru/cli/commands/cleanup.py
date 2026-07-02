"""mineru cleanup — clean up deleted files, orphan docs, and temp files."""

from __future__ import annotations

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import (
    CleanupDeletedRequest,
    CleanupDeletedResponse,
    CleanupOrphansRequest,
    CleanupOrphansResponse,
    CleanupTempRequest,
    CleanupTempResponse,
)
from ..contracts import CliContext
from ..runtime import run_cli

app = typer.Typer(
    name="cleanup",
    help="Clean up local doclib records and temp files.",
    no_args_is_help=True,
)


@app.command("deleted-files")
def cleanup_deleted_files(
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Preview only"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Remove all file rows already marked as deleted."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: _client().cleanup_deleted_files(CleanupDeletedRequest(dry_run=dry_run)),
        render=_render_cleanup_deleted,
    )


@app.command("orphan-docs")
def cleanup_orphan_docs(
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Preview only"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Remove docs that are no longer referenced by any file row."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: _client().cleanup_orphan_docs(CleanupOrphansRequest(dry_run=dry_run)),
        render=_render_cleanup_orphans,
    )


@app.command("temp")
def cleanup_temp_files(
    older_than: int = typer.Option(7, "--older-than", help="Days threshold for temp cleanup"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Remove old process temp files."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: _client().cleanup_temp_files(CleanupTempRequest(older_than_days=older_than)),
        render=_render_cleanup_temp,
    )


def _client() -> DoclibClient:
    return DoclibClient(timeout=30)


def _render_cleanup_deleted(data: CleanupDeletedResponse) -> str:
    if data.dry_run:
        return f"Would remove {data.deleted_files} deleted file record(s). Use --no-dry-run to proceed."
    return f"Removed {data.deleted_files} deleted file record(s)."


def _render_cleanup_orphans(data: CleanupOrphansResponse) -> str:
    if data.dry_run:
        return f"Would remove {data.orphan_docs} orphan doc(s). Use --no-dry-run to proceed."
    return f"Removed {data.orphan_docs} orphan doc(s)."


def _render_cleanup_temp(data: CleanupTempResponse) -> str:
    return f"Removed {data.temp_files_removed} temp file(s)."
