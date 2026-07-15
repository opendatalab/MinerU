"""mineru usage - Remote API usage and limits."""

from __future__ import annotations

from datetime import datetime, timezone

import typer
from pydantic import BaseModel, Field

from ...doclib.client import DoclibClient
from ...doclib.types import RemoteUsageResponse
from ..contracts import CliContext, RenderableOutput
from ..guidance import (
    REMOTE_API_URL_CONFIG,
    api_key_guidance_for_anonymous_usage,
    api_key_guidance_for_error,
)
from ..runtime import run_cli


class RemoteUsageOutput(BaseModel):
    remote_url: str
    usage: RemoteUsageResponse
    guidance: dict[str, object] | None = None
    guidance_text: str | None = Field(default=None, exclude=True, repr=False)


def usage_cmd(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """Show Remote API usage and limits."""
    run_cli(
        CliContext(json_mode=json_mode),
        _get_remote_usage,
        render=_render_remote_usage,
        error_guidance=api_key_guidance_for_error,
    )


def _get_remote_usage() -> RemoteUsageOutput:
    client = DoclibClient(timeout=30)
    remote_url = client.get_config_key(REMOTE_API_URL_CONFIG).value
    usage = client.get_remote_usage()
    guidance = api_key_guidance_for_anonymous_usage(remote_url) if usage.access_level == "anonymous" else None
    return RemoteUsageOutput(
        remote_url=remote_url,
        usage=usage,
        guidance=guidance.data if guidance is not None else None,
        guidance_text=guidance.text if guidance is not None else None,
    )


def _render_remote_usage(view: RemoteUsageOutput) -> RenderableOutput:
    usage = view.usage
    lines = [
        "Remote API Usage",
        "",
        f"Remote URL: {view.remote_url}",
        f"Access level: {usage.access_level}",
        f"Billing period: {_format_billing_period(usage.billing_period.start, usage.billing_period.end)}",
        "",
        "Current",
        f"  Pages processed: {usage.current.pages_processed}",
        f"  Files processed: {usage.current.files_processed}",
        f"  Jobs created: {usage.current.jobs_created}",
        "",
        "Limits",
        f"  Max pages per file: {usage.limits.max_pages_per_file}",
        f"  Max file size: {_format_bytes(usage.limits.max_file_size_bytes)}",
        f"  Max files per job: {usage.limits.max_files_per_job}",
        f"  Max concurrent jobs: {usage.limits.max_concurrent_jobs}",
        f"  File retention: {_format_retention(usage.limits.max_file_retention_days)}",
    ]
    output: list[str] = ["\n".join(lines)]
    if view.guidance_text is not None:
        output.append(view.guidance_text)
    return output


def _format_billing_period(start: str, end: str | None) -> str:
    formatted_start = _format_utc_timestamp(start)
    if end is None:
        return f"{formatted_start} - ongoing"
    return f"{formatted_start} - {_format_utc_timestamp(end)}"


def _format_utc_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return value
    return parsed.strftime("%Y-%m-%d %H:%M UTC")


def _format_bytes(value: int) -> str:
    mib = value / (1024 * 1024)
    return f"{mib:g} MiB"


def _format_retention(days: int | None) -> str:
    if days is None:
        return "not specified"
    return f"{days} day" if days == 1 else f"{days} days"


__all__ = ["usage_cmd"]
