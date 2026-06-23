"""mineru scan — create doclib scan tasks."""

from __future__ import annotations

from pathlib import Path
import time

from ...errors import MineruError
from ...doclib.client import DoclibClient
from ...doclib.types import ScanRequest
from ..json_errors import exit_with_error
from ..output import print_error, print_info, print_json, print_success
from ..path_utils import normalize_cli_path


def scan_cmd(
    path: str,
    *,
    wait: int = 30,
    no_wait: bool = False,
    json_mode: bool = False,
) -> None:
    try:
        _scan_path(path, wait=wait, no_wait=no_wait, json_mode=json_mode)
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode, fallback_message="Cannot connect to mineru server. Run 'mineru server start' first.")


def _scan_path(path: str, *, wait: int, no_wait: bool, json_mode: bool) -> None:
    scan_path = normalize_cli_path(path)
    if not Path(scan_path).exists():
        raise MineruError("file_not_found", f"File or directory not found: {scan_path}", "path")
    client = _client()
    scan_info = client.create_scan(ScanRequest(path=scan_path, kind="manual", source="cli"))
    if not no_wait and wait > 0:
        scan_info = _wait_for_scan(client, scan_info.id, wait)
    _print_scan(scan_info, json_mode=json_mode)


def _client() -> DoclibClient:
    try:
        return DoclibClient(timeout=30)
    except Exception:
        raise RuntimeError("Cannot connect to mineru server. Run 'mineru server start' first.") from None


def _wait_for_scan(client: DoclibClient, scan_id: int, wait_seconds: int) -> object:
    deadline = time.time() + wait_seconds
    scan_info = client.get_scan(scan_id)
    while scan_info.status in ("pending", "running") and time.time() < deadline:
        time.sleep(0.5)
        scan_info = client.get_scan(scan_id)
    return scan_info


def _print_scan(scan_info: object, *, json_mode: bool) -> None:
    if json_mode:
        print_json(scan_info)
        return
    status = getattr(scan_info, "status", "?")
    if status == "failed":
        print_error(f"Scan failed: {getattr(scan_info, 'error_code', '')} {getattr(scan_info, 'error_msg', '')}")
        return
    message = (
        f"Scan {getattr(scan_info, 'id', '?')} {status}: "
        f"seen={getattr(scan_info, 'files_seen', 0)}, refreshed={getattr(scan_info, 'files_refreshed', 0)}, "
        f"new={getattr(scan_info, 'files_new', 0)}, changed={getattr(scan_info, 'files_changed', 0)}, "
        f"deleted={getattr(scan_info, 'files_deleted', 0)}, unreachable={getattr(scan_info, 'files_unreachable', 0)}, "
        f"excluded={getattr(scan_info, 'files_excluded', 0)}, unsupported={getattr(scan_info, 'files_unsupported', 0)}"
    )
    if status in ("pending", "running"):
        print_info(message)
    else:
        print_success(message)
