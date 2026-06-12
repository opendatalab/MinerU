"""mineru parse — document parsing command."""

from __future__ import annotations

import time
from pathlib import Path

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import ParseRequest
from ...types import Tier
from ..output import format_parse_result, print_error, print_success, print_info


def parse_cmd(
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
    wait: int = typer.Option(
        60, "--wait", help="Max seconds to wait for parse to complete"
    ),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="Don't wait — return immediately"
    ),
    output: str = typer.Option(
        None, "-o", "--output", help="Output file path (default: STDOUT)"
    ),
    no_marker: bool = typer.Option(
        False, "--no-marker", help="Omit document structure markers from output"
    ),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
) -> None:
    """Parse a document file."""
    file_path = str(Path(path).resolve())

    if not Path(file_path).exists():
        print_error(f"File not found: {file_path}")
        raise typer.Exit(1)

    try:
        client = DoclibClient(timeout=wait + 30)
    except Exception:
        print_error("Cannot connect to mineru server. Run 'mineru server start' first.")
        raise typer.Exit(1) from None

    # send parse request
    try:
        result = client.ensure_parse(
            ParseRequest(
                path=file_path,
                tier=tier,
                pages=pages,
                force=force,
                remote=remote,
            )
        )
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    sha256 = result.sha256
    req_tier = result.tier
    result_pages = result.pages or pages
    status = result.status
    wait_parse_ids = result.wait_parse_ids

    # cached
    if status == "done":
        if verbose:
            print_info("Cache hit — returning cached result.")
        _output_content(client, sha256, req_tier, json_mode, output, pages=result_pages, format=format)
        return

    # no-wait
    if no_wait or status not in ("pending", "parsing") or not wait_parse_ids:
        format_parse_result(result, json_mode=json_mode)
        return

    # poll until done or timeout
    if verbose:
        print_info(f"Parse queued (tier={req_tier}). Waiting up to {wait}s...")

    deadline = time.time() + wait
    interval = 0.5
    while time.time() < deadline:
        time.sleep(interval)
        interval = min(interval * 1.2, 3.0)
        try:
            s = client.list_parses(ids=wait_parse_ids)
        except Exception:
            continue

        parse_rows = s.parses
        statuses = {row.status for row in parse_rows}
        st = "done" if parse_rows and statuses == {"done"} else ("failed" if "failed" in statuses else "parsing")
        if verbose or st in ("done", "failed"):
            print_info(f"  Parse status: {st}")

        if st == "done":
            if not json_mode:
                _output_content(client, sha256, req_tier, json_mode, output, pages=result_pages, format=format)
            else:
                format_parse_result(s, json_mode=json_mode)
            return
        if st == "failed":
            failed = next((row for row in parse_rows if row.status == "failed"), None)
            print_error(f"Parse failed: {failed.error_code if failed else '?'} — {failed.error_msg if failed else ''}")
            raise typer.Exit(1)

    print_info(
        f"Parse still in progress (tier={req_tier}). "
        f"Check status with: mineru info {file_path}"
    )
    if json_mode:
        print_info('{"status":"parsing","tip":"Re-run the same command to continue waiting."}')


if __name__ != "__main__":
    def _register(app: typer.Typer) -> None:
        app.command()(parse_cmd)


def _output_content(
    client,
    sha256: str,
    tier: Tier,
    json_mode: bool,
    output: str | None = None,
    pages: str | None = None,
    format: str = "markdown",
) -> None:
    """Fetch and output parsed content.  If --output is specified, server writes to file."""
    try:
        content = client.get_doc_content(sha256, tier=tier, output=output, pages=pages, format=format)
        if output and output != "-":
            if content.output:
                print_success(f"Written to {content.output}")
            else:
                print_error("Failed to write output file.")
            return

        text = content.content or ""
        if text:
            if json_mode:
                import json
                print(json.dumps({"sha256": sha256, "tier": tier, "content": text}, ensure_ascii=False))
            else:
                print(text)
        else:
            print_error("No content returned from parse.")
    except Exception as exc:
        print_error(f"Failed to read content: {exc}")
