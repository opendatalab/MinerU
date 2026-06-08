"""mineru parse — document parsing command."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer

from mineru.doclib.client import MineruClient
from mineru.cli_next.output import format_parse_result, print_error, print_success, print_info


def parse_cmd(
    path: str = typer.Argument(..., help="Path to the document file"),
    tier: str = typer.Option(None, "--tier", help="Parse tier: flash, standard, pro"),
    pages: str = typer.Option(
        None, "-p", "--pages", help="Page range, e.g. '1~5' or 'all'"
    ),
    force: bool = typer.Option(False, "--force", help="Force re-parse, ignore cache"),
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
        client = MineruClient(timeout=wait + 30)
    except Exception:
        print_error("Cannot connect to mineru server. Run 'mineru server start' first.")
        raise typer.Exit(1) from None

    # send parse request
    try:
        result = client.parse(file_path, tier=tier, pages=pages, force=force)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    sha256 = result.get("sha256", "")
    req_tier = result.get("tier", tier or "flash")
    status = result.get("status", "?")

    # cached
    if status == "done":
        if verbose:
            print_info("Cache hit — returning cached result.")
        _output_content(client, sha256, req_tier, json_mode, output)
        return

    # no-wait
    if no_wait or status not in ("pending", "parsing"):
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
            s = client.parse_status(sha256, req_tier)
        except Exception:
            continue

        st = s.get("status", "?")
        if verbose or st in ("done", "failed"):
            print_info(f"  Parse status: {st}")

        if st == "done":
            if not json_mode:
                _output_content(client, sha256, req_tier, json_mode, output)
            else:
                format_parse_result(s, json_mode=json_mode)
            return
        elif st == "failed":
            err = s.get("error", {})
            print_error(f"Parse failed: {err.get('code','?')} — {err.get('message','')}")
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


def _output_content(client, sha256: str, tier: str, json_mode: bool,
                     output: str | None = None) -> None:
    """Fetch and output parsed content.  If --output is specified, server writes to file."""
    try:
        content = client.parse_content(sha256, tier, output=output)
        if output and output != "-":
            if content.get("output"):
                print_success(f"Written to {content['output']}")
            else:
                print_error("Failed to write output file.")
            return

        text = content.get("content", "")
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
