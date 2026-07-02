"""CLI telemetry context and first-run prompt."""

from __future__ import annotations

import os
import sys

import typer

from ..doclib.client import DoclibClient
from ..doclib.telemetry import TelemetryContext, infer_default_client_context, reset_telemetry_context, set_telemetry_context

PROMPT_COMMANDS = {
    "parse",
    "read",
    "scan",
    "watch",
    "search",
    "find",
    "list",
    "show",
    "invalidate",
    "forget",
    "cleanup",
}

TELEMETRY_CONSENT_MESSAGE = """
Help improve MinerU by sending anonymous, locally aggregated usage and diagnostic data.

Collected:
    command names, MinerU version, OS, architecture, Python version, install channel,
    coarse CPU/GPU categories, success/failure status, error categories, tiers,
    and performance timing buckets.

NOT collected:
    document contents, extracted text/images, file names, file paths, raw URLs,
    search queries, prompts, snippets, tracebacks, exception messages, hostnames,
    usernames, account IDs, API keys, or exact CPU/GPU models.

Press Enter or type Y to enable, or type N to disable.
You can change this later with `mineru telemetry enable` or `mineru telemetry disable`.
Preview what would be sent with `mineru telemetry preview`.
""".strip()


def prepare_cli_telemetry(ctx: typer.Context) -> None:
    telemetry_context = infer_default_client_context(source="cli")
    token = set_telemetry_context(telemetry_context)
    ctx.call_on_close(lambda: reset_telemetry_context(token))
    command = _top_level_command(ctx)
    if not _should_prompt_telemetry_consent(ctx, command, telemetry_context):
        return
    maybe_prompt_telemetry_consent()


def maybe_prompt_telemetry_consent() -> None:
    if not _is_interactive():
        return
    try:
        client = DoclibClient(timeout=5)
        status = client.get_telemetry_status()
    except Exception:
        return
    if status.state != "unset":
        return

    typer.echo(TELEMETRY_CONSENT_MESSAGE)
    enabled = typer.confirm("Enable telemetry?", default=True)
    try:
        client.telemetry_action("enable" if enabled else "disable")
    except Exception:
        return


def _top_level_command(ctx: typer.Context) -> str | None:
    command = ctx.invoked_subcommand
    if command:
        return command
    args = list(getattr(ctx, "args", []) or [])
    return args[0] if args else None


def _should_prompt_telemetry_consent(ctx: typer.Context, command: str | None, telemetry_context: TelemetryContext) -> bool:
    if command not in PROMPT_COMMANDS:
        return False
    if telemetry_context.caller == "agent":
        return False
    args = _raw_args(ctx)
    if _has_help_option(args) or _has_json_option(args):
        return False
    if _is_ci_environment():
        return False
    return True


def _raw_args(ctx: typer.Context) -> list[str]:
    raw_args = ctx.meta.get("mineru_raw_args")
    if isinstance(raw_args, list):
        return [str(arg) for arg in raw_args]
    args = list(getattr(ctx, "args", []) or [])
    if ctx.invoked_subcommand and (not args or args[0] != ctx.invoked_subcommand):
        return [ctx.invoked_subcommand, *args]
    return args


def _has_help_option(args: list[str]) -> bool:
    return any(arg in {"--help", "-h"} for arg in args)


def _has_json_option(args: list[str]) -> bool:
    return any(arg == "--json" or arg.startswith("--json=") for arg in args)


def _is_ci_environment() -> bool:
    for name in ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE", "JENKINS_URL"):
        value = os.environ.get(name)
        if value and value.lower() not in {"0", "false", "no"}:
            return True
    return False


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


__all__ = ["maybe_prompt_telemetry_consent", "prepare_cli_telemetry"]
