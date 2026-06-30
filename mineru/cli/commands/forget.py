"""mineru forget — remove doclib records for a path without touching source files."""

from __future__ import annotations

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import ForgetPathRequest, ForgetPathResponse
from ..contracts import CliContext
from ..path_utils import normalize_cli_path
from ..runtime import run_cli


def forget_cmd(
    path: str = typer.Argument(..., help="File or directory path to forget from doclib"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Preview only"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    run_cli(
        CliContext(json_mode=json_mode),
        lambda: DoclibClient(timeout=30).forget_path(
            ForgetPathRequest(path=normalize_cli_path(path), dry_run=dry_run),
        ),
        render=_render_forget_result,
        warnings=lambda result: result.warnings,
    )


def _render_forget_result(data: ForgetPathResponse) -> str:
    if data.dry_run:
        return (
            f"Would forget {data.forgotten_files} file record(s) (matched_as={data.matched_as}). Use --no-dry-run to proceed."
        )
    return f"Forgot {data.forgotten_files} file record(s) (matched_as={data.matched_as})."
