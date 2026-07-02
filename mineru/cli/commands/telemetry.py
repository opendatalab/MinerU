"""mineru telemetry — manage doclib telemetry."""

from __future__ import annotations

import json
from typing import Literal

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import TelemetryActionResponse, TelemetryPayload, TelemetryStatusResponse
from ..contracts import CliContext
from ..runtime import run_cli

app = typer.Typer(help="Telemetry management", no_args_is_help=True)


@app.command("status")
def telemetry_status(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """Show telemetry status."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _client().get_telemetry_status(), render=_render_telemetry_status)


@app.command("preview")
def telemetry_preview(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """Print the next telemetry request body without sending it."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _client().get_telemetry_preview().body, render=_render_telemetry_preview)


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


def _run_action(action: Literal["enable", "disable", "flush"], *, json_mode: bool) -> None:
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _client().telemetry_action(action), render=_render_telemetry_action)


def _client() -> DoclibClient:
    return DoclibClient(timeout=30)


def _render_telemetry_status(data: TelemetryStatusResponse) -> str:
    last_flush = data.last_flush_at if data.last_flush_at is not None else "never"
    return "\n".join(
        [
            f"state: {data.state}",
            f"installation_id: {data.installation_id}",
            f"pending_periods: {data.pending_periods}",
            f"pending_metrics: {data.pending_metrics}",
            f"last_flush_at: {last_flush}",
        ]
    )


def _render_telemetry_action(data: TelemetryActionResponse) -> str:
    if data.action == "flush":
        return f"telemetry flush: {data.reason or 'success'}"
    return f"telemetry {data.state}"


def _render_telemetry_preview(data: TelemetryPayload) -> str:
    return json.dumps(data.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)


__all__ = ["app"]
