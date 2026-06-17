"""mineru CLI — personal document center, built for agents."""

from __future__ import annotations

from typing import Literal

import typer
from click.core import Context
from typer.core import TyperGroup

from ..types import Tier
from .commands import cleanup, config, list_resources, server, show, watch

TOP_LEVEL_COMMAND_ORDER = [
    "parse",
    "read",
    "scan",
    "watch",
    "search",
    "find",
    "list",
    "show",
    "server",
    "config",
    "invalidate",
    "forget",
    "cleanup",
]


class OrderedRootGroup(TyperGroup):
    def list_commands(self, ctx: Context) -> list[str]:
        ordered = [name for name in TOP_LEVEL_COMMAND_ORDER if name in self.commands]
        return ordered + [name for name in self.commands if name not in TOP_LEVEL_COMMAND_ORDER]


app = typer.Typer(
    name="mineru",
    cls=OrderedRootGroup,
    help="MinerU — your personal document center, built for agents",
    no_args_is_help=True,
)


# top-level commands
@app.command()
def parse(
    path: str = typer.Argument(..., help="Path to the document file"),
    tier: Tier | None = typer.Option(None, "--tier", help="Parse tier: flash, standard, pro (default: server decides)"),
    pages: str = typer.Option(None, "-p", "--pages", help="Page range, e.g. '1~5' or 'all'"),
    after: str = typer.Option(None, "--after", help="Continue reading after a content cursor"),
    limit: int = typer.Option(30000, "--limit", help="Soft character limit for STDOUT content"),
    format: Literal["markdown"] = typer.Option("markdown", "-f", "--format", help="Output format: markdown"),
    force: bool = typer.Option(False, "--force", help="Force re-parse, ignore cache"),
    remote: bool = typer.Option(False, "--remote", help="Use remote parse-server"),
    wait: int = typer.Option(60, "--wait", help="Max seconds to wait for parse to complete"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Don't wait — return immediately"),
    output: str = typer.Option(None, "-o", "--output", help="Output path; creates parent directories"),
    no_marker: bool = typer.Option(False, "--no-marker", help="Omit document structure markers from output"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
) -> None:
    """Parse a document file."""
    from .commands.parse import parse_cmd

    parse_cmd(
        path=path,
        tier=tier,
        pages=pages,
        after=after,
        limit=limit,
        format=format,
        force=force,
        remote=remote,
        wait=wait,
        no_wait=no_wait,
        output=output,
        no_marker=no_marker,
        json_mode=json_mode,
        verbose=verbose,
    )


@app.command()
def read(
    locator: str = typer.Argument(..., help="Doclib locator, e.g. doc:ab12cd3/tier:standard/page:4"),
    context: int = typer.Option(0, "--context", help="Read N pages/blocks before and after the locator"),
    limit: int = typer.Option(30000, "--limit", help="Soft character limit for STDOUT content"),
    format: Literal["markdown", "image"] = typer.Option("markdown", "-f", "--format", help="Output format: markdown, image"),
    output: str = typer.Option(None, "-o", "--output", help="Output path; creates parent directories"),
    no_marker: bool = typer.Option(False, "--no-marker", help="Omit continuation marker from output"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Read parsed doclib content by locator."""
    from .commands.read import read_cmd

    read_cmd(
        locator=locator,
        context=context,
        limit=limit,
        format=format,
        output=output,
        no_marker=no_marker,
        json_mode=json_mode,
    )


@app.command()
def scan(
    path: str = typer.Argument(..., help="File or directory path to scan"),
    wait: int = typer.Option(30, "--wait", help="Max seconds to wait for scan completion"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Return immediately after creating the scan"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Create a one-off scan task."""
    from .commands.scan import scan_cmd

    scan_cmd(path=path, wait=wait, no_wait=no_wait, json_mode=json_mode)


app.add_typer(watch.app, name="watch")


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


app.add_typer(list_resources.app, name="list")
app.add_typer(show.app, name="show")
app.add_typer(server.app, name="server")
app.add_typer(config.app, name="config")


@app.command()
def invalidate(
    path: str = typer.Argument(..., help="Path to the document file"),
    tier: Tier | None = typer.Option(None, "--tier", help="Parse tier to invalidate (omit = all tiers)"),
) -> None:
    """Mark done parse results as superseded so the next parse re-runs."""
    from .commands.invalidate import invalidate_cmd

    invalidate_cmd(path=path, tier=tier)


@app.command()
def forget(
    path: str = typer.Argument(..., help="File or directory path to forget from doclib"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Preview only"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Forget doclib records for a path without deleting source files."""
    from .commands.forget import forget_cmd

    forget_cmd(path=path, dry_run=dry_run, json_mode=json_mode)


app.add_typer(cleanup.app, name="cleanup")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
