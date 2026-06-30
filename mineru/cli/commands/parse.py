"""mineru parse — document parsing command."""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import click
import typer

from ...doclib.client import DoclibClient
from ...doclib.types import (
    ContentNextRequest,
    DocContentExportRequest,
    DocContentResponse,
    ParseRequest,
    ParseResponse,
    TelemetryObservation,
    TelemetryObservationsRequest,
)
from ...errors import MineruError
from ...types import Tier
from ..contracts import CliContext, CliResult
from ..path_utils import normalize_cli_path
from ..runtime import cli_ok, emit_result, run_cli


@dataclass(frozen=True)
class ParseTextOutput:
    text: str


@dataclass(frozen=True)
class ParseSummaryOutput:
    text: str


def _normalize_output_path(output: str | None) -> str | None:
    if output is None or output == "-":
        return output
    return normalize_cli_path(output)


def _ensure_output_parent(output: str | None) -> str | None:
    normalized = _normalize_output_path(output)
    if normalized in (None, "-"):
        return normalized
    Path(normalized).parent.mkdir(parents=True, exist_ok=True)
    return normalized


def _output_info(path: str) -> dict[str, str]:
    return {"status": "written", "path": path}


def _emit_notice(message: str, *, json_mode: bool) -> None:
    emit_result(CliContext(json_mode=json_mode), cli_ok(notices=[message]))


def _validate_page_range_input(page_range: str | None) -> None:
    if page_range is None or page_range.strip() == "" or page_range.strip() == "all":
        return
    try:
        for raw_part in page_range.split(","):
            part = raw_part.strip()
            if not part:
                raise ValueError
            if "~" in part:
                raw_start, raw_end = part.split("~", 1)
                start = int(raw_start.strip())
                end = int(raw_end.strip())
                if start > 0 and end > 0 and start > end:
                    raise ValueError
            else:
                int(part)
    except ValueError:
        raise MineruError("page_range_invalid", f"Invalid page range: {page_range}", "pages") from None


def _option_was_explicit(option_name: str) -> bool:
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return False
    return ctx.get_parameter_source(option_name) == click.core.ParameterSource.COMMANDLINE


def parse_cmd(
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
    ctx = CliContext(json_mode=json_mode, verbose=verbose)
    run_cli(
        ctx,
        lambda: _parse(
            path,
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
        ),
    )


def _parse(
    path: str,
    *,
    tier: Tier | None,
    pages: str | None,
    after: str | None,
    limit: int,
    format: Literal["markdown"],
    force: bool,
    remote: bool,
    wait: int,
    no_wait: bool,
    output: str | None,
    no_marker: bool,
    json_mode: bool,
    verbose: bool,
) -> None:
    file_path = normalize_cli_path(path)

    if not Path(file_path).exists():
        raise MineruError("file_not_found", f"File not found: {file_path}", "path")

    _validate_page_range_input(pages)

    client = DoclibClient(timeout=wait + 30)

    # send parse request
    result = client.ensure_parse(
        ParseRequest(
            path=file_path,
            tier=tier,
            page_range=pages,
            force=force,
            remote=remote,
        )
    )

    req_tier = result.tier
    status = result.status
    wait_parse_ids = result.wait_parse_ids
    next_marker_tier = tier if _option_was_explicit("tier") else None
    next_marker_limit = limit if _option_was_explicit("limit") else None
    next_marker_remote = remote if _option_was_explicit("remote") else False

    # cached
    if status == "done":
        if verbose:
            _emit_notice("Cache hit — returning cached result.", json_mode=json_mode)
        _output_parse_result(
            client,
            result,
            json_mode=json_mode,
            output=output,
            after=after,
            limit=limit,
            format=format,
            no_marker=no_marker,
            source_path=file_path,
            next_marker_tier=next_marker_tier,
            next_marker_limit=next_marker_limit,
            next_marker_remote=next_marker_remote,
        )
        return

    # no-wait
    if no_wait or status not in ("pending", "parsing") or not wait_parse_ids:
        if json_mode:
            _emit_parse_json_response(result, None)
        else:
            emit_result(CliContext(json_mode=False), _prepare_parse_summary_output(result))
        if status == "failed":
            emit_result(CliContext(json_mode=json_mode), cli_ok(exit_code=1))
        return

    # poll until done or timeout
    if verbose:
        _emit_notice(f"Parse queued (tier={req_tier}). Waiting up to {wait}s...", json_mode=json_mode)

    wait_started_at = time.time()
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
        statuses = {row.status for row in parse_rows}
        st = "done" if parse_rows and statuses == {"done"} else ("failed" if "failed" in statuses else "parsing")
        if verbose and not json_mode:
            _emit_notice(f"  Parse status: {st}", json_mode=json_mode)

        if st == "done":
            _record_parse_wait(client, wait_parse_ids, "succeeded", wait_started_at)
            done_result = result.model_copy(update={"status": "done"})
            _output_parse_result(
                client,
                done_result,
                json_mode=json_mode,
                output=output,
                after=after,
                limit=limit,
                format=format,
                no_marker=no_marker,
                source_path=file_path,
                next_marker_tier=next_marker_tier,
                next_marker_limit=next_marker_limit,
                next_marker_remote=next_marker_remote,
            )
            return
        if st == "failed":
            failed = next((row for row in parse_rows if row.status == "failed"), None)
            _record_parse_wait(client, wait_parse_ids, "failed", wait_started_at)
            raise MineruError(failed.error_code if failed else "parse_failed", failed.error_msg if failed else "", None)

    _record_parse_wait(client, wait_parse_ids, "timeout", wait_started_at)
    if json_mode:
        timeout_result = result.model_copy(update={"status": "parsing", "tip": "Re-run the same command to continue waiting."})
        _emit_parse_json_response(timeout_result, None)
    else:
        _emit_notice(
            f"Parse still in progress (tier={req_tier}). Check status with: mineru show file {file_path}", json_mode=json_mode
        )


def _record_parse_wait(client: DoclibClient, parse_ids: list[int], status: str, started_at: float) -> None:
    duration_ms = max(0, int((time.time() - started_at) * 1000))
    try:
        client.record_observations(
            TelemetryObservationsRequest(
                observations=[
                    TelemetryObservation(
                        metric_name="parse.wait",
                        parse_ids=parse_ids,
                        duration_ms=duration_ms,
                        dimensions={"status": status},
                    )
                ]
            )
        )
    except Exception:
        pass


def _output_parse_result(
    client: DoclibClient,
    parse_result: ParseResponse,
    json_mode: bool,
    output: str | None = None,
    after: str | None = None,
    limit: int = 30000,
    format: str = "markdown",
    no_marker: bool = False,
    source_path: str | None = None,
    next_marker_tier: Tier | None = None,
    next_marker_limit: int | None = None,
    next_marker_remote: bool = False,
) -> None:
    """Fetch and output parsed content for a parse command."""
    sha256 = parse_result.sha256
    tier = parse_result.tier
    page_range = parse_result.page_range
    output_path = _ensure_output_parent(output)
    if output_path and output_path != "-":
        exported = client.export_doc_content(
            sha256,
            DocContentExportRequest(
                tier=tier,
                page_range=page_range,
                format=format,
                output=output_path,
                no_marker=no_marker,
            ),
        )
        if json_mode:
            _emit_parse_json_response(parse_result, None, output=exported.output)
            return
        emit_result(
            CliContext(json_mode=False), cli_ok(ParseTextOutput(f"Written to {exported.output}"), render=_render_parse_text)
        )
        return

    content = _fetch_doc_content(
        client,
        sha256,
        tier=tier,
        page_range=page_range,
        after=after,
        limit=limit,
        format=format,
        no_marker=no_marker,
    )
    if not content.content and content.format == "markdown" and not content.content_ranges:
        raise MineruError("parse_empty", "No content returned from parse.")
    if json_mode:
        _emit_parse_json_response(parse_result, content)
        return
    emit_result(
        CliContext(json_mode=False),
        _prepare_parse_content_output(
            content,
            source_path=source_path,
            no_marker=no_marker,
            next_marker_tier=next_marker_tier,
            next_marker_limit=next_marker_limit,
            next_marker_remote=next_marker_remote,
        ),
    )


def _fetch_doc_content(
    client: DoclibClient,
    sha256: str,
    *,
    tier: Tier,
    page_range: str | None,
    after: str | None,
    limit: int,
    format: str,
    no_marker: bool,
) -> DocContentResponse:
    return client.get_doc_content(
        sha256,
        tier=tier,
        page_range=page_range,
        after=after,
        limit=limit,
        format=format,
        no_marker=no_marker,
    )


def _emit_parse_json_response(
    parse_result: ParseResponse,
    content: DocContentResponse | None,
    *,
    output: str | None = None,
) -> None:
    emit_result(CliContext(json_mode=True), cli_ok(_parse_json_payload(parse_result, content, output=output)))


def _parse_json_payload(
    parse_result: ParseResponse,
    content: DocContentResponse | None,
    *,
    output: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "parse": parse_result.model_dump(mode="json"),
        "content": content.model_dump(mode="json") if content is not None else None,
    }
    if output is not None:
        payload["output"] = _output_info(output)
    return payload


def _prepare_parse_summary_output(parse_result: ParseResponse) -> CliResult[ParseSummaryOutput] | CliResult[None]:
    status = parse_result.status
    tip = parse_result.tip or ""
    if status in ("pending", "parsing"):
        return cli_ok(ParseSummaryOutput(f"Parse {status}... {tip}"), render=_render_parse_summary)
    if status == "failed":
        return cli_ok(None, notices=["Parse failed: ? — "])
    return cli_ok(ParseSummaryOutput(f"Parse complete (tier={parse_result.tier}) {tip}"), render=_render_parse_summary)


def _render_parse_summary(data: ParseSummaryOutput) -> str:
    return data.text


def _prepare_parse_content_output(
    content: DocContentResponse,
    *,
    source_path: str | None,
    no_marker: bool,
    next_marker_tier: Tier | None = None,
    next_marker_limit: int | None = None,
    next_marker_remote: bool = False,
) -> CliResult[ParseTextOutput] | CliResult[None]:
    if not content.content and content.content_ranges:
        message = "No renderable content in requested pages."
        if content.next_request and not no_marker:
            marker = _parse_next_marker(
                source_path,
                content.next_request,
                tier=next_marker_tier,
                limit=next_marker_limit,
                remote=next_marker_remote,
            )
            if marker:
                return cli_ok(ParseTextOutput(_append_next_marker(message, marker)), render=_render_parse_text)
        return cli_ok(None, notices=[message])
    if content.next_request and not no_marker:
        marker = _parse_next_marker(
            source_path,
            content.next_request,
            tier=next_marker_tier,
            limit=next_marker_limit,
            remote=next_marker_remote,
        )
        if marker:
            return cli_ok(ParseTextOutput(_append_next_marker(content.content, marker)), render=_render_parse_text)
    return cli_ok(ParseTextOutput(content.content), render=_render_parse_text)


def _render_parse_text(data: ParseTextOutput) -> str:
    return data.text


def _append_next_marker(content: str, marker: str) -> str:
    newline_count = len(content) - len(content.rstrip("\n"))
    separator = "\n" * max(2 - newline_count, 0)
    return f"{content}{separator}{marker}"


def _parse_next_marker(
    path: str | None,
    next_request: ContentNextRequest,
    *,
    tier: Tier | None = None,
    limit: int | None = None,
    remote: bool = False,
) -> str | None:
    if not path:
        return None
    parts = ["mineru", "parse", path]
    if tier:
        parts.extend(["--tier", tier])
    if remote:
        parts.append("--remote")
    if limit is not None:
        parts.extend(["--limit", str(limit)])
    if next_request.page_range:
        parts.extend(["--pages", next_request.page_range])
    if next_request.after:
        parts.extend(["--after", next_request.after])
    return f"<!-- Next: {shlex.join(parts)} -->"
