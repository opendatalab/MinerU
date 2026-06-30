"""mineru watch — watch target management."""

from __future__ import annotations

import typer
from rich.table import Table

from ...doclib.client import DoclibClient
from ...doclib.types import RemoveWatchResponse, ScanInfo, ScanRequest, WatchInfo, WatchListResponse, WatchRequest
from ...errors import MineruError
from ..contracts import CliContext, CliTaskResult
from ..path_utils import normalize_cli_path
from ..runtime import cli_task, run_cli
from .scan import _render_scan, _wait_for_scan

app = typer.Typer(help="Watch target management", no_args_is_help=True)


@app.command("add")
def watch_add(
    path: str = typer.Argument(..., help="Directory path to watch"),
    removable: bool = typer.Option(False, "--removable", help="Removable device"),
    label: str = typer.Option(None, "--label", help="Label for this watch"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Add a directory to watch."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(
        ctx,
        lambda: _client().add_watch(
            WatchRequest(path=normalize_cli_path(path), removable=removable, label=label)
        ),
        render=_render_watch_added,
    )


@app.command("list")
def watch_list(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """List watched directories."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _client().list_watches(), render=_render_watch_list)


@app.command("remove")
def watch_remove(
    target: str = typer.Argument(..., help="Watch id or exact watch root path to remove"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Remove a watched directory."""
    ctx = CliContext(json_mode=json_mode)

    def remove_watch() -> RemoveWatchResponse:
        client = _client()
        watch = _resolve_watch(client, target)
        return client.remove_watch(watch.id)

    run_cli(ctx, remove_watch, render=_render_watch_removed)


@app.command("rescan")
def watch_rescan(
    target: str = typer.Argument(..., help="Watch id or exact watch root path"),
    wait: int = typer.Option(30, "--wait", help="Max seconds to wait for scan completion"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Return immediately after creating the scan"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Create a watch scan task for an existing watch target."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _rescan_watch(target, wait=wait, no_wait=no_wait), render=_render_scan)


def _rescan_watch(target: str, *, wait: int, no_wait: bool) -> CliTaskResult[ScanInfo]:
    client = _client()
    watch = _resolve_watch(client, target)
    scan_info = client.create_scan(ScanRequest(path=watch.path, kind="watch", source="cli", watch_id=watch.id))
    if not no_wait and wait > 0:
        scan_info = _wait_for_scan(client, scan_info.id, wait)
    return cli_task(scan_info, status=scan_info.status, render=_render_scan, fail_if_final_failed=not no_wait)


def _client() -> DoclibClient:
    return DoclibClient(timeout=30)


def _render_watch_added(data: WatchInfo) -> str:
    return f"Watch added: {data.path} (id={data.id})"


def _render_watch_list(data: WatchListResponse) -> Table | str:
    if not data.watches:
        return "No watches configured."
    table = Table(title="Watches")
    table.add_column("ID", justify="right")
    table.add_column("Path", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Removable")
    table.add_column("Label")
    for watch in data.watches:
        table.add_row(
            str(watch.id),
            watch.path,
            watch.status,
            "yes" if watch.removable else "no",
            watch.label or "-",
        )
    return table


def _render_watch_removed(data: RemoveWatchResponse) -> str:
    action = "removed" if data.removed else "unchanged"
    return f"Watch {data.watch_id} {action}."


def _resolve_watch(client: DoclibClient, target: str) -> WatchInfo:
    watches = client.list_watches().watches
    if target.isdigit():
        watch_id = int(target)
        for watch in watches:
            if watch.id == watch_id:
                return watch
        raise MineruError("watch_not_found", f"Watch id {watch_id} not found.", "watch_id")

    normalized = normalize_cli_path(target)
    for watch in watches:
        if normalize_cli_path(watch.path) == normalized:
            return watch
    raise MineruError("watch_not_found", f"Watch path {normalized} not found.", "path")
