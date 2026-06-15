"""mineru scan — create and inspect doclib scan tasks."""

from __future__ import annotations

import time

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import ScanKind, ScanRequest, ScanStatus
from ..output import print_error, print_info, print_json, print_success


def scan_cmd(
    args: list[str],
    *,
    wait: int = 30,
    no_wait: bool = False,
    json_mode: bool = False,
) -> None:
    if not args:
        print_error("Usage: mineru scan <path> | mineru scan status <scan_id> | mineru scan list")
        raise typer.Exit(1)

    command = args[0]
    try:
        if command == "status":
            _show_scan_task(args[1:], json_mode=json_mode)
            return
        if command == "list":
            _scan_list(args[1:], json_mode=json_mode)
            return
        _scan_path(command, wait=wait, no_wait=no_wait, json_mode=json_mode)
    except Exception as exc:
        print_error(str(exc) or "Cannot connect to mineru server. Run 'mineru server start' first.")
        raise typer.Exit(1) from None


def _scan_path(path: str, *, wait: int, no_wait: bool, json_mode: bool) -> None:
    client = _client()
    scan_info = client.create_scan(ScanRequest(path=path, kind="manual", source="cli"))
    if not no_wait and wait > 0:
        scan_info = _wait_for_scan(client, scan_info.id, wait)
    _print_scan(scan_info, json_mode=json_mode)


def _show_scan_task(args: list[str], *, json_mode: bool) -> None:
    if len(args) != 1:
        raise ValueError("Usage: mineru scan status <scan_id>")
    scan_info = _client().get_scan(int(args[0]))
    _print_scan(scan_info, json_mode=json_mode)


def _scan_list(args: list[str], *, json_mode: bool) -> None:
    limit = 50
    status: ScanStatus | None = None
    kind: ScanKind | None = None
    watch_id: int | None = None

    index = 0
    while index < len(args):
        option = args[index]
        if option in {"--limit", "-n"}:
            limit = int(_option_value(args, index, option))
            index += 2
            continue
        if option == "--status":
            status = _option_value(args, index, option)  # type: ignore[assignment]
            index += 2
            continue
        if option == "--kind":
            kind = _option_value(args, index, option)  # type: ignore[assignment]
            index += 2
            continue
        if option == "--watch-id":
            watch_id = int(_option_value(args, index, option))
            index += 2
            continue
        raise ValueError(f"Unknown scan list option: {option}")

    result = _client().list_scans(limit=limit, status=status, kind=kind, watch_id=watch_id)
    if json_mode:
        print_json(result)
        return
    if not result.scans:
        print_info("No scans found.")
        return
    for item in result.scans:
        print(
            f"[{item.id}] {item.status} {item.kind} {item.path} "
            f"seen={item.files_seen} refreshed={item.files_refreshed} "
            f"new={item.files_new} changed={item.files_changed} deleted={item.files_deleted} "
            f"unreachable={item.files_unreachable} errors={item.files_error}"
        )


def _option_value(args: list[str], index: int, option: str) -> str:
    try:
        return args[index + 1]
    except IndexError:
        raise ValueError(f"Missing value for {option}") from None


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
