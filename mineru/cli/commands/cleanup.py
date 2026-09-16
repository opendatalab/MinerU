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
from ...utils.i18n import t
from ..contracts import CliContext
from ..runtime import run_cli

app = typer.Typer(
    name="cleanup",
    help=t("Clean up local doclib records and temp files."),
    no_args_is_help=True,
)


@app.command("deleted-files", help=t("Remove all file rows already marked as deleted."))
def cleanup_deleted_files(
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help=t("Preview only")),
    json_mode: bool = typer.Option(False, "--json", help=t("JSON output")),
) -> None:
    """Remove all file rows already marked as deleted."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: _client().cleanup_deleted_files(CleanupDeletedRequest(dry_run=dry_run)),
        render=_render_cleanup_deleted,
    )


@app.command("orphan-docs", help=t("Remove docs that are no longer referenced by any file row."))
def cleanup_orphan_docs(
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help=t("Preview only")),
    json_mode: bool = typer.Option(False, "--json", help=t("JSON output")),
) -> None:
    """Remove docs that are no longer referenced by any file row."""
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: _client().cleanup_orphan_docs(CleanupOrphansRequest(dry_run=dry_run)),
        render=_render_cleanup_orphans,
    )


@app.command("temp", help=t("Remove old process temp files."))
def cleanup_temp_files(
    older_than: int = typer.Option(7, "--older-than", help=t("Days threshold for temp cleanup")),
    json_mode: bool = typer.Option(False, "--json", help=t("JSON output")),
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
        return t("Would remove {count} deleted file record(s). Use --no-dry-run to proceed.", count=data.deleted_files)
    return t("Removed {count} deleted file record(s).", count=data.deleted_files)


def _render_cleanup_orphans(data: CleanupOrphansResponse) -> str:
    if data.dry_run:
        return t("Would remove {count} orphan doc(s). Use --no-dry-run to proceed.", count=data.orphan_docs)
    return t("Removed {count} orphan doc(s).", count=data.orphan_docs)


def _render_cleanup_temp(data: CleanupTempResponse) -> str:
    return t("Removed {count} temp file(s).", count=data.temp_files_removed)
