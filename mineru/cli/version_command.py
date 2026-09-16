"""Shared version command for MinerU CLI entry points."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import typer

from ..utils.i18n import t
from .contracts import CliContext
from .runtime import run_cli

__all__ = ["VersionInfo", "render_version", "show_version", "version_cmd", "version_info"]


@dataclass(frozen=True)
class VersionInfo:
    mineru_version: str
    python_version: str


def version_info() -> VersionInfo:
    from ..version import __version__

    return VersionInfo(mineru_version=__version__, python_version=sys.version.split()[0])


def render_version(data: VersionInfo) -> str:
    return "\n".join(
        [
            t("MinerU version: {version}", version=data.mineru_version),
            t("Python version: {version}", version=data.python_version),
        ]
    )


def show_version(ctx: typer.Context, value: bool) -> None:
    if not value:
        return
    typer.echo(render_version(version_info()))
    ctx.exit()


def version_cmd(json_mode: bool = typer.Option(False, "--json", help=t("JSON output"))) -> None:
    """Print MinerU and Python versions."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, version_info, render=render_version)
