"""mineru parse — document parsing command."""

from __future__ import annotations

import time
from pathlib import Path

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import ContentNextRequest, DocContentExportRequest, ParseRequest
from ...types import Tier
from ..output import format_parse_result, print_error, print_info, print_success


def parse_cmd(
    path: str = typer.Argument(..., help="Path to the document file"),
    tier: Tier | None = typer.Option(None, "--tier", help="Parse tier: flash, standard, pro (default: server decides)"),
    pages: str = typer.Option(None, "-p", "--pages", help="Page range, e.g. '1~5' or 'all'"),
    after: str = typer.Option(None, "--after", help="Continue reading after a content cursor"),
    limit: int = typer.Option(30000, "--limit", help="Soft character limit for STDOUT content"),
    format: str = typer.Option("markdown", "-f", "--format", help="Output format: markdown, text, json, html"),
    force: bool = typer.Option(False, "--force", help="Force re-parse, ignore cache"),
    remote: bool = typer.Option(False, "--remote", help="Use remote parse-server"),
    wait: int = typer.Option(60, "--wait", help="Max seconds to wait for parse to complete"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Don't wait — return immediately"),
    output: str = typer.Option(None, "-o", "--output", help="Output file path (default: STDOUT)"),
    no_marker: bool = typer.Option(False, "--no-marker", help="Omit document structure markers from output"),
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
                page_range=pages,
                force=force,
                remote=remote,
            )
        )
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    sha256 = result.sha256
    req_tier = result.tier
    result_page_range = result.page_range or pages
    status = result.status
    wait_parse_ids = result.wait_parse_ids

    # cached
    if status == "done":
        if verbose:
            print_info("Cache hit — returning cached result.")
        _output_content(
            client,
            sha256,
            req_tier,
            json_mode,
            output,
            page_range=result_page_range,
            after=after,
            limit=limit,
            format=format,
            no_marker=no_marker,
            source_path=file_path,
        )
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

        wait_id_set = set(wait_parse_ids)
        parse_rows = [row for row in s.parses if row.id in wait_id_set]
        filtered_response = s.model_copy(update={"parses": parse_rows, "total": len(parse_rows)})
        statuses = {row.status for row in parse_rows}
        st = "done" if parse_rows and statuses == {"done"} else ("failed" if "failed" in statuses else "parsing")
        if verbose or st in ("done", "failed"):
            print_info(f"  Parse status: {st}")

        if st == "done":
            if not json_mode:
                _output_content(
                    client,
                    sha256,
                    req_tier,
                    json_mode,
                    output,
                    page_range=result_page_range,
                    after=after,
                    limit=limit,
                    format=format,
                    no_marker=no_marker,
                    source_path=file_path,
                )
            else:
                format_parse_result(filtered_response, json_mode=json_mode)
            return
        if st == "failed":
            failed = next((row for row in parse_rows if row.status == "failed"), None)
            print_error(f"Parse failed: {failed.error_code if failed else '?'} — {failed.error_msg if failed else ''}")
            raise typer.Exit(1)

    print_info(f"Parse still in progress (tier={req_tier}). Check status with: mineru show file {file_path}")
    if json_mode:
        print_info('{"status":"parsing","tip":"Re-run the same command to continue waiting."}')


if __name__ != "__main__":

    def _register(app: typer.Typer) -> None:
        app.command()(parse_cmd)


def _output_content(
    client: DoclibClient,
    sha256: str,
    tier: Tier,
    json_mode: bool,
    output: str | None = None,
    page_range: str | None = None,
    after: str | None = None,
    limit: int = 30000,
    format: str = "markdown",
    no_marker: bool = False,
    source_path: str | None = None,
) -> None:
    """Fetch and output parsed content.  If --output is specified, server writes to file."""
    try:
        if output and output != "-":
            exported = client.export_doc_content(
                sha256,
                DocContentExportRequest(
                    tier=tier,
                    page_range=page_range,
                    format=format,
                    output=output,
                    no_marker=no_marker,
                ),
            )
            print_success(f"Written to {exported.output}")
            return

        content = client.get_doc_content(
            sha256,
            tier=tier,
            page_range=page_range,
            after=after,
            limit=limit,
            format=format,
            no_marker=no_marker,
        )
        text = content.content or ""
        if text:
            if json_mode:
                import json

                payload = {
                    "sha256": content.sha256,
                    "short_id": content.short_id,
                    "tier": content.tier,
                    "format": content.format,
                    "content": text,
                    "truncated": content.truncated,
                    "next_request": _next_request_dict(content.next_request),
                }
                print(json.dumps(payload, ensure_ascii=False))
            else:
                print(text)
                if content.next_request and not no_marker and source_path:
                    print(_next_marker(source_path, content.next_request))
        else:
            print_error("No content returned from parse.")
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as exc:
        print_error(f"Failed to read content: {exc}")
        raise typer.Exit(1) from None


def _next_request_dict(next_request: ContentNextRequest | None) -> dict[str, str] | None:
    if next_request is None:
        return None
    result: dict[str, str] = {}
    if next_request.page_range:
        result["page_range"] = next_request.page_range
    if next_request.after:
        result["after"] = next_request.after
    return result


def _next_marker(path: str, next_request: ContentNextRequest) -> str:
    parts = ["mineru", "parse", path]
    if next_request.page_range:
        parts.extend(["--pages", next_request.page_range])
    if next_request.after:
        parts.extend(["--after", next_request.after])
    return f"<!-- Next: {' '.join(parts)} -->"
