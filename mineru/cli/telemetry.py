"""CLI 遥测上下文与一次性默认开启说明。"""

from __future__ import annotations

import os
import sys
from contextlib import suppress

import typer

from ..doclib.client import DoclibClient
from ..doclib.telemetry import TelemetryContext, infer_default_client_context, reset_telemetry_context, set_telemetry_context
from ..utils.i18n import t

NOTICE_COMMANDS = {
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

TELEMETRY_NOTICE_MESSAGE = """
Doclib telemetry is enabled by default: locally aggregated usage, coarse environment categories, and performance metrics.
Document contents, file names and paths, search queries, and API keys are not included.
Inspect data with `mineru telemetry preview`; turn it off with `mineru telemetry disable`.
Set MINERU_TELEMETRY_DISABLED=1 before starting Doclib to disable it from the first startup.
""".strip()


def prepare_cli_telemetry(ctx: typer.Context) -> None:
    """设置调用上下文，仅在适合人阅读的业务命令中展示一次说明。"""
    telemetry_context = infer_default_client_context(source="cli")
    token = set_telemetry_context(telemetry_context)
    ctx.call_on_close(lambda: reset_telemetry_context(token))
    command = _top_level_command(ctx)
    if not _should_show_telemetry_notice(ctx, command, telemetry_context):
        return
    maybe_show_telemetry_notice()


def maybe_show_telemetry_notice() -> None:
    """非阻塞地告知默认策略，单独确认已展示，不修改用户开关。"""
    if not _is_interactive():
        return
    client: DoclibClient | None = None
    try:
        client = DoclibClient(timeout=5)
        status = client.get_telemetry_status()
        if status.state != "enabled" or status.notice_shown:
            return
        typer.echo(t(TELEMETRY_NOTICE_MESSAGE), err=True)
        client.telemetry_action("notice-shown")
    except Exception:
        return
    finally:
        if client is not None:
            with suppress(Exception):
                client.close()


def _top_level_command(ctx: typer.Context) -> str | None:
    command = ctx.invoked_subcommand
    if command:
        return command
    args = list(getattr(ctx, "args", []) or [])
    return args[0] if args else None


def _should_show_telemetry_notice(ctx: typer.Context, command: str | None, telemetry_context: TelemetryContext) -> bool:
    """帮助、JSON、CI 与 Agent 不插入提示，保持原命令输出协议。"""
    if command not in NOTICE_COMMANDS:
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


__all__ = ["maybe_show_telemetry_notice", "prepare_cli_telemetry"]
