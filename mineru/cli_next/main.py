"""mineru CLI — local document center."""

from __future__ import annotations

import typer

from ..types import Tier
from .commands import cleanup, config, server, watch

app = typer.Typer(
    name="mineru",
    help="MinerU — 本地文档中心",
    no_args_is_help=True,
)

app.add_typer(server.app, name="server")
app.add_typer(watch.app, name="watch")
app.add_typer(config.app, name="config")
app.add_typer(cleanup.app, name="cleanup")


# top-level commands
@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def scan(
    ctx: typer.Context,
    args: list[str] = typer.Argument(..., help="Path, or subcommand: status/list"),
    wait: int = typer.Option(30, "--wait", help="Max seconds to wait for scan completion"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Return immediately after creating the scan"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Create a scan task, or inspect scan tasks."""
    from .commands.scan import scan_cmd

    scan_cmd(args=args + list(ctx.args), wait=wait, no_wait=no_wait, json_mode=json_mode)


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
    file_type: str | None = typer.Option(None, "--type", help="File type filter"),
    tier: Tier | None = typer.Option(None, "--tier", help="Exact search index tier: flash, standard, pro"),
    min_tier: Tier | None = typer.Option(None, "--min-tier", help="Minimum search index tier: flash, standard, pro"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    offset: int = typer.Option(0, "--offset", help="Result offset"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Search parsed document content."""
    from .commands.search import search_cmd

    search_cmd(query=query, file_type=file_type, tier=tier, min_tier=min_tier, limit=limit, offset=offset, json_mode=json_mode)


@app.command()
def find(
    query: str = typer.Argument(..., help="Filename search query"),
    ext: str | None = typer.Option(None, "--ext", help="File extension filter, e.g. pdf"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Search filenames only (not document content)."""
    from .commands.search import find_cmd

    find_cmd(query=query, ext=ext, limit=limit, json_mode=json_mode)


@app.command()
def info(
    path: str = typer.Argument(..., help="File path"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show file/document details and parse status."""
    from .commands.search import info_cmd

    info_cmd(path=path, json_mode=json_mode)


@app.command()
def forget(
    path: str = typer.Argument(..., help="File or directory path to forget from doclib"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Preview only"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Forget doclib records for a path without deleting source files."""
    from .commands.forget import forget_cmd

    forget_cmd(path=path, dry_run=dry_run, json_mode=json_mode)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
