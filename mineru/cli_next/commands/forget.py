"""mineru forget — remove doclib records for a path without touching source files."""

from __future__ import annotations

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import ForgetPathRequest
from ..output import print_error, print_info, print_json, print_success


def forget_cmd(
    path: str,
    *,
    dry_run: bool = True,
    json_mode: bool = False,
) -> None:
    try:
        client = DoclibClient(timeout=30)
    except Exception:
        print_error("Cannot connect to mineru server. Run 'mineru server start' first.")
        raise typer.Exit(1) from None

    try:
        result = client.forget_path(ForgetPathRequest(path=path, dry_run=dry_run))
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    if json_mode:
        print_json(result)
        return

    for warning in result.warnings:
        print_info(warning)

    if dry_run:
        print_info(
            f"Would forget {result.forgotten_files} file record(s) "
            f"(matched_as={result.matched_as}). Use --no-dry-run to proceed."
        )
        return

    print_success(f"Forgot {result.forgotten_files} file record(s) (matched_as={result.matched_as}).")
