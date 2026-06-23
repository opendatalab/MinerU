"""mineru forget — remove doclib records for a path without touching source files."""

from __future__ import annotations

from ...doclib.client import DoclibClient
from ...doclib.types import ForgetPathRequest
from ..json_errors import exit_with_error
from ..output import print_info, print_json, print_success
from ..path_utils import normalize_cli_path


def forget_cmd(
    path: str,
    *,
    dry_run: bool = True,
    json_mode: bool = False,
) -> None:
    try:
        client = DoclibClient(timeout=30)
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode, fallback_message="Cannot connect to mineru server. Run 'mineru server start' first.")

    file_path = normalize_cli_path(path)
    try:
        result = client.forget_path(ForgetPathRequest(path=file_path, dry_run=dry_run))
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)

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
