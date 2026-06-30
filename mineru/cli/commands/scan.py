"""mineru scan — create doclib scan tasks."""

from __future__ import annotations

import time
from pathlib import Path

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import ScanInfo, ScanRequest
from ...errors import MineruError
from ..contracts import CliContext, CliTaskResult
from ..path_utils import normalize_cli_path
from ..runtime import cli_task, run_cli


def scan_cmd(
    path: str = typer.Argument(..., help="File or directory path to scan"),
    wait: int = typer.Option(30, "--wait", help="Max seconds to wait for scan completion"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Return immediately after creating the scan"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _scan(path, wait=wait, no_wait=no_wait), render=_render_scan)


def _scan(path: str, *, wait: int, no_wait: bool) -> CliTaskResult[ScanInfo]:
    scan_path = normalize_cli_path(path)
    if not Path(scan_path).exists():
        raise MineruError("file_not_found", f"File or directory not found: {scan_path}", "path")
    client = _client()
    scan_info = client.create_scan(ScanRequest(path=scan_path, kind="manual", source="cli"))
    if not no_wait and wait > 0:
        scan_info = _wait_for_scan(client, scan_info.id, wait)
    return cli_task(scan_info, status=scan_info.status, render=_render_scan, fail_if_final_failed=not no_wait)


def _client() -> DoclibClient:
    return DoclibClient(timeout=30)


def _wait_for_scan(client: DoclibClient, scan_id: int, wait_seconds: int) -> ScanInfo:
    deadline = time.time() + wait_seconds
    scan_info = client.get_scan(scan_id)
    while scan_info.status in ("pending", "running") and time.time() < deadline:
        time.sleep(0.5)
        scan_info = client.get_scan(scan_id)
    return scan_info


def _render_scan(scan_info: ScanInfo) -> str:
    status = scan_info.status
    if status == "failed":
        return f"Scan failed: {scan_info.error_code or ''} {scan_info.error_msg or ''}"
    return (
        f"Scan {scan_info.id} {status}: "
        f"seen={scan_info.files_seen}, refreshed={scan_info.files_refreshed}, "
        f"new={scan_info.files_new}, changed={scan_info.files_changed}, "
        f"deleted={scan_info.files_deleted}, unreachable={scan_info.files_unreachable}, "
        f"excluded={scan_info.files_excluded}, unsupported={scan_info.files_unsupported}"
    )
