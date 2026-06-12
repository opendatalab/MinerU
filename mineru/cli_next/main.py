"""mineru CLI — local document center."""

from __future__ import annotations

import typer

from ..types import Tier
from .commands import config, server

app = typer.Typer(
    name="mineru",
    help="MinerU — 本地文档中心",
    no_args_is_help=True,
)

app.add_typer(server.app, name="server")
app.add_typer(config.app, name="config")


# top-level commands
@app.command()
def parse(
    path: str = typer.Argument(..., help="Path to the document file"),
    tier: Tier | None = typer.Option(None, "--tier", help="Parse tier: flash, standard, pro (default: server decides)"),
    pages: str = typer.Option(
        None, "-p", "--pages", help="Page range, e.g. '1~5' or 'all'"
    ),
    format: str = typer.Option(
        "markdown", "-f", "--format", help="Output format: markdown, text, json, html"
    ),
    force: bool = typer.Option(False, "--force", help="Force re-parse, ignore cache"),
    remote: bool = typer.Option(False, "--remote", help="Use remote parse-server (https://mineru.net/api)"),
    wait: int = typer.Option(60, "--wait", help="Max seconds to wait for parse to complete"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Don't wait — return immediately"),
    output: str = typer.Option(None, "-o", "--output", help="Output file path (default: STDOUT)"),
    no_marker: bool = typer.Option(
        False, "--no-marker", help="Omit document structure markers from output"
    ),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
) -> None:
    """Parse a document file."""
    from .commands.parse import parse_cmd

    parse_cmd(
        path=path, tier=tier, pages=pages, format=format, force=force,
        remote=remote,
        wait=wait, no_wait=no_wait, output=output,
        no_marker=no_marker, json_mode=json_mode, verbose=verbose,
    )


@app.command()
def invalidate(
    path: str = typer.Argument(..., help="Path to the document file"),
    tier: Tier | None = typer.Option(None, "--tier", help="Parse tier to invalidate (omit = all tiers)"),
) -> None:
    """Mark done parse results as superseded so the next parse re-runs."""
    from .commands.invalidate import invalidate_cmd

    invalidate_cmd(path=path, tier=tier)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    file_type: str = typer.Option(None, "--type", help="File type filter"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    offset: int = typer.Option(0, "--offset", help="Result offset"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Search parsed document content."""
    from .commands.search import search_cmd

    search_cmd(query=query, file_type=file_type, limit=limit, offset=offset, json_mode=json_mode)


@app.command()
def find(
    query: str = typer.Argument(..., help="Filename search query"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Search filenames only (not document content)."""
    from .commands.search import find_cmd

    find_cmd(query=query, limit=limit, json_mode=json_mode)


@app.command()
def info(
    path: str = typer.Argument(..., help="File path"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show file/document details and parse status."""
    from .commands.search import info_cmd

    info_cmd(path=path, json_mode=json_mode)


@app.command()
def cleanup(
    orphans: bool = typer.Option(False, "--orphans", help="Clean orphan docs"),
    deleted: bool = typer.Option(False, "--deleted", help="Clean deleted file records"),
    temp: bool = typer.Option(False, "--temp", help="Clean temporary files"),
    older_than: int = typer.Option(30, "--older-than", help="Days threshold for deleted cleanup"),
    dry_run: bool = typer.Option(True, "--dry-run", help="Preview only"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Clean up database records and temp files."""
    from .commands.cleanup import cleanup_cmd

    cleanup_cmd(
        orphans=orphans, deleted=deleted, temp=temp,
        older_than=older_than, dry_run=dry_run, json_mode=json_mode,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
