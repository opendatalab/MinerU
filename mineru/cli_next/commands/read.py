"""mineru read — read doclib content by locator."""

from __future__ import annotations

from typing import Literal

import typer

from ...doclib.client import DoclibClient
from ..json_errors import exit_with_error
from .parse import output_doc_content_response


def read_cmd(
    locator: str = typer.Argument(..., help="Doclib locator, e.g. doc:ab12cd3/tier:standard/page:4"),
    context: int = typer.Option(0, "--context", help="Read N pages/blocks before and after the locator"),
    limit: int = typer.Option(30000, "--limit", help="Soft character limit for STDOUT content"),
    format: Literal["markdown", "image"] = typer.Option("markdown", "-f", "--format", help="Output format: markdown, image"),
    output: str = typer.Option(None, "-o", "--output", help="Output path; creates parent directories"),
    no_marker: bool = typer.Option(False, "--no-marker", help="Omit continuation marker from output"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Read parsed doclib content by locator."""
    try:
        client = DoclibClient(timeout=60)
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode, fallback_message="Cannot connect to mineru server. Run 'mineru server start' first.")

    try:
        content = client.read_content(
            locator,
            context=context,
            limit=limit,
            format=format,
            no_marker=no_marker,
        )
        output_doc_content_response(
            content,
            json_mode=json_mode,
            output=output,
            source_path=None,
            read_mode=True,
            no_marker=no_marker,
        )
    except typer.Exit:
        raise
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)


if __name__ != "__main__":

    def _register(app: typer.Typer) -> None:
        app.command()(read_cmd)
