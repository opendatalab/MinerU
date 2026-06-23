"""mineru telemetry — manage doclib telemetry."""

from __future__ import annotations

import json

import typer

from ...doclib.client import DoclibClient
from ..json_errors import exit_with_error
from ..output import print_json, print_success

app = typer.Typer(help="Telemetry management", no_args_is_help=True)


@app.command("status")
def telemetry_status(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """Show telemetry status."""
    try:
        data = _client().get_telemetry_status()
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)
    if json_mode:
        print_json(data)
        return
    last_flush = data.last_flush_at if data.last_flush_at is not None else "never"
    print(f"state: {data.state}")
    print(f"installation_id: {data.installation_id}")
    print(f"pending_periods: {data.pending_periods}")
    print(f"pending_metrics: {data.pending_metrics}")
    print(f"last_flush_at: {last_flush}")


@app.command("preview")
def telemetry_preview() -> None:
    """Print the next telemetry request body without sending it."""
    try:
        data = _client().get_telemetry_preview()
    except Exception as exc:
        exit_with_error(exc, json_mode=False)
    print(json.dumps(data.body.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True))


@app.command("enable")
def telemetry_enable(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """Enable telemetry."""
    _run_action("enable", json_mode=json_mode)


@app.command("disable")
def telemetry_disable(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """Disable telemetry and clear pending local aggregates."""
    _run_action("disable", json_mode=json_mode)


@app.command("flush")
def telemetry_flush(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """Flush pending telemetry now when telemetry is enabled."""
    _run_action("flush", json_mode=json_mode)


def _run_action(action: str, *, json_mode: bool) -> None:
    try:
        data = _client().telemetry_action(action)
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)
    if json_mode:
        print_json(data)
        return
    if action == "flush":
        print_success(f"telemetry flush: {data.reason or 'success'}")
        return
    print_success(f"telemetry {data.state}")


def _client() -> DoclibClient:
    return DoclibClient(timeout=30)


__all__ = ["app"]
