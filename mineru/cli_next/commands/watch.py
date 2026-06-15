"""mineru watch — watch target management."""

from __future__ import annotations

import os

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import ScanRequest, WatchInfo, WatchRequest
from ..output import print_error, print_info, print_json, print_success
from .scan import _print_scan, _wait_for_scan

app = typer.Typer(help="Watch target management", no_args_is_help=True)


@app.command("add")
def watch_add(
    path: str = typer.Argument(..., help="Directory path to watch"),
    removable: bool = typer.Option(False, "--removable", help="Removable device"),
    label: str = typer.Option(None, "--label", help="Label for this watch"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Add a directory to watch."""
    try:
        data = _client().add_watch(WatchRequest(path=path, removable=removable, label=label))
    except Exception as exc:
        print_error(str(exc) or "Cannot add watch target.")
        raise typer.Exit(1) from None

    if json_mode:
        print_json(data)
        return
    print_success(f"Watch added: {data.path} (id={data.id})")


@app.command("list")
def watch_list(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """List watched directories."""
    try:
        data = _client().list_watches()
    except Exception as exc:
        print_error(str(exc) or "Cannot list watch targets.")
        raise typer.Exit(1) from None

    if json_mode:
        print_json(data)
        return

    if not data.watches:
        print_info("No watches configured.")
        return
    for watch in data.watches:
        status = watch.status
        extra = " [removable]" if watch.removable else ""
        print(f"  [{watch.id}] {watch.path}{extra}  [{status}]")


@app.command("remove")
def watch_remove(
    target: str = typer.Argument(..., help="Watch id or exact watch root path to remove"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Remove a watched directory."""
    try:
        client = _client()
        watch = _resolve_watch(client, target)
        result = client.remove_watch(watch.id)
    except Exception as exc:
        print_error(str(exc) or "Cannot remove watch target.")
        raise typer.Exit(1) from None

    if json_mode:
        print_json(result)
        return
    print_success(f"Watch removed: {watch.path}")


@app.command("rescan")
def watch_rescan(
    target: str = typer.Argument(..., help="Watch id or exact watch root path"),
    wait: int = typer.Option(30, "--wait", help="Max seconds to wait for scan completion"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Return immediately after creating the scan"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Create a watch scan task for an existing watch target."""
    try:
        client = _client()
        watch = _resolve_watch(client, target)
        scan_info = client.create_scan(ScanRequest(path=watch.path, kind="watch", source="cli", watch_id=watch.id))
        if not no_wait and wait > 0:
            scan_info = _wait_for_scan(client, scan_info.id, wait)
    except Exception as exc:
        print_error(str(exc) or "Cannot rescan watch target.")
        raise typer.Exit(1) from None

    _print_scan(scan_info, json_mode=json_mode)


def _client() -> DoclibClient:
    return DoclibClient(timeout=30)


def _resolve_watch(client: DoclibClient, target: str) -> WatchInfo:
    watches = client.list_watches().watches
    if target.isdigit():
        watch_id = int(target)
        for watch in watches:
            if watch.id == watch_id:
                return watch
        raise ValueError(f"watch_not_found: watch id {watch_id} not found")

    normalized = os.path.abspath(os.path.expanduser(target))
    for watch in watches:
        if os.path.abspath(os.path.expanduser(watch.path)) == normalized:
            return watch
    raise ValueError(f"watch_not_found: watch path {normalized} not found")
