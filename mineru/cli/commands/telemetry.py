"""mineru telemetry — manage doclib telemetry."""

from __future__ import annotations

import json
from typing import Literal

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import TelemetryActionResponse, TelemetryPayload, TelemetryStatusResponse
from ...utils.i18n import t
from ..contracts import CliContext
from ..runtime import run_cli

app = typer.Typer(help=t("Telemetry management"), no_args_is_help=True)


@app.command("status", help=t("Show telemetry status."))
def telemetry_status(json_mode: bool = typer.Option(False, "--json", help=t("JSON output"))) -> None:
    """Show telemetry status."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _client().get_telemetry_status(), render=_render_telemetry_status)


@app.command("preview", help=t("Print the next telemetry request body without sending it."))
def telemetry_preview(json_mode: bool = typer.Option(False, "--json", help=t("JSON output"))) -> None:
    """Print the next telemetry request body without sending it."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _client().get_telemetry_preview().body, render=_render_telemetry_preview)


@app.command("enable", help=t("Enable telemetry."))
def telemetry_enable(json_mode: bool = typer.Option(False, "--json", help=t("JSON output"))) -> None:
    """Enable telemetry."""
    _run_action("enable", json_mode=json_mode)


@app.command("disable", help=t("Disable telemetry and clear pending local aggregates."))
def telemetry_disable(json_mode: bool = typer.Option(False, "--json", help=t("JSON output"))) -> None:
    """Disable telemetry and clear pending local aggregates."""
    _run_action("disable", json_mode=json_mode)


@app.command("flush", help=t("Flush pending telemetry now when telemetry is enabled (or unset during prerelease)."))
def telemetry_flush(json_mode: bool = typer.Option(False, "--json", help=t("JSON output"))) -> None:
    """Flush pending telemetry now when telemetry is enabled (or unset during prerelease)."""
    _run_action("flush", json_mode=json_mode)


def _run_action(action: Literal["enable", "disable", "flush"], *, json_mode: bool) -> None:
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _client().telemetry_action(action), render=_render_telemetry_action)


def _client() -> DoclibClient:
    return DoclibClient(timeout=30)


def _render_telemetry_status(data: TelemetryStatusResponse) -> str:
    last_flush = data.last_flush_at if data.last_flush_at is not None else t("never")
    return "\n".join(
        [
            t("state: {state}", state=data.state),
            t("installation_id: {id}", id=data.installation_id),
            t("pending_periods: {n}", n=data.pending_periods),
            t("pending_metrics: {n}", n=data.pending_metrics),
            t("last_flush_at: {value}", value=last_flush),
        ]
    )


def _render_telemetry_action(data: TelemetryActionResponse) -> str:
    if data.action == "flush":
        return t("telemetry flush: {reason}", reason=data.reason or "success")
    return t("telemetry {state}", state=data.state)


def _render_telemetry_preview(data: TelemetryPayload) -> str:
    return json.dumps(data.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)


__all__ = ["app"]
