"""mineru cleanup — clean up deleted files, orphan docs, and temp files."""

from __future__ import annotations

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import CleanupDeletedRequest, CleanupOrphansRequest, CleanupTempRequest
from ..json_errors import exit_with_error
from ..output import print_info, print_json, print_success

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
    try:
        client = _client()
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode, fallback_message="Cannot connect to mineru server. Run 'mineru server start' first.")
    try:
        data = client.cleanup_deleted_files(CleanupDeletedRequest(dry_run=dry_run))
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)

    if json_mode:
        print_json(data)
        return

    count = data.deleted_files
    if dry_run:
        print_info(f"Would remove {count} deleted file record(s). Use --no-dry-run to proceed.")
    else:
        print_success(f"Removed {count} deleted file record(s).")


@app.command("orphan-docs")
def cleanup_orphan_docs(
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Preview only"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Remove docs that are no longer referenced by any file row."""
    try:
        client = _client()
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode, fallback_message="Cannot connect to mineru server. Run 'mineru server start' first.")
    try:
        data = client.cleanup_orphan_docs(CleanupOrphansRequest(dry_run=dry_run))
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)

    if json_mode:
        print_json(data)
        return

    count = data.orphan_docs
    if dry_run:
        print_info(f"Would remove {count} orphan doc(s). Use --no-dry-run to proceed.")
    else:
        print_success(f"Removed {count} orphan doc(s).")


@app.command("temp")
def cleanup_temp_files(
    older_than: int = typer.Option(7, "--older-than", help="Days threshold for temp cleanup"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Remove old process temp files."""
    try:
        client = _client()
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode, fallback_message="Cannot connect to mineru server. Run 'mineru server start' first.")
    try:
        data = client.cleanup_temp_files(CleanupTempRequest(older_than_days=older_than))
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)

    if json_mode:
        print_json(data)
        return

    print_success(f"Removed {data.temp_files_removed} temp file(s).")


def _client() -> DoclibClient:
    return DoclibClient(timeout=30)
