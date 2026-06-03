"""mineru cleanup — clean up orphan docs, deleted records, temp files."""

from __future__ import annotations

import typer

from mineru.server.client import MineruClient
from mineru.cli_next.output import print_error, print_success, print_info


def cleanup_cmd(
    orphans: bool = typer.Option(False, "--orphans", help="Clean orphan docs"),
    deleted: bool = typer.Option(False, "--deleted", help="Clean deleted file records"),
    temp: bool = typer.Option(False, "--temp", help="Clean temporary files"),
    older_than: int = typer.Option(30, "--older-than", help="Days threshold for deleted cleanup"),
    dry_run: bool = typer.Option(True, "--dry-run", help="Preview only — don't actually delete"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Clean up database records and temp files.

    Examples:
        mineru cleanup --orphans --dry-run
        mineru cleanup --deleted --older-than 60
        mineru cleanup --temp
    """
    try:
        client = MineruClient(timeout=30)
    except Exception:
        print_error("Cannot connect to mineru server. Run 'mineru server start' first.")
        raise typer.Exit(1) from None

    if orphans:
        try:
            data = client._post("/cleanup/orphans", json_data={"dry_run": dry_run})
        except Exception as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None
        count = data.get("orphan_docs", 0)
        if dry_run:
            print_info(f"Would remove {count} orphan doc(s). Use --no-dry-run to proceed.")
        else:
            print_success(f"Removed {count} orphan doc(s).")
    elif deleted:
        try:
            data = client._post("/cleanup/deleted", json_data={
                "dry_run": dry_run, "older_than_days": older_than,
            })
        except Exception as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None
        count = data.get("deleted_files", 0)
        if dry_run:
            print_info(f"Would remove {count} deleted file record(s). Use --no-dry-run to proceed.")
        else:
            print_success(f"Removed {count} deleted file record(s).")
    elif temp:
        try:
            data = client._post("/cleanup/temp", json_data={"older_than_days": older_than})
        except Exception as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None
        count = data.get("temp_files_removed", 0)
        print_success(f"Removed {count} temp file(s).")
    else:
        print_info("Specify --orphans, --deleted, or --temp.")
