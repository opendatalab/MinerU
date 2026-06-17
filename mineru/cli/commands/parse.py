"""mineru parse — document parsing command."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any
from typing import Literal

import typer

from ...errors import MineruError
from ...doclib.client import DoclibClient
from ...doclib.types import ContentNextRequest, DocContentExportRequest, DocContentResponse, ParseRequest, ParseResponse
from ...types import Tier
from ..json_errors import exit_with_error
from ..output import format_parse_result, print_error, print_info, print_json, print_success


def _normalize_output_path(output: str | None) -> str | None:
    if output is None or output == "-":
        return output
    return os.path.abspath(os.path.expanduser(output))


def _ensure_output_parent(output: str | None) -> str | None:
    normalized = _normalize_output_path(output)
    if normalized in (None, "-"):
        return normalized
    Path(normalized).parent.mkdir(parents=True, exist_ok=True)
    return normalized


def _output_info(path: str) -> dict[str, str]:
    return {"status": "written", "path": path}


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
    file_path = str(Path(os.path.expanduser(path)).resolve())

    if not Path(file_path).exists():
        exit_with_error(MineruError("file_not_found", f"File not found: {file_path}", "path"), json_mode=json_mode)

    try:
        _validate_page_range_input(pages)
    except MineruError as exc:
        exit_with_error(exc, json_mode=json_mode)

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
        exit_with_error(exc, json_mode=json_mode)

    req_tier = result.tier
    status = result.status
    wait_parse_ids = result.wait_parse_ids

    # cached
    if status == "done":
        if verbose:
            print_info("Cache hit — returning cached result.")
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
        )
        return

    # no-wait
    if no_wait or status not in ("pending", "parsing") or not wait_parse_ids:
        if json_mode:
            _print_parse_json_response(result, None)
        else:
            format_parse_result(result, json_mode=False)
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
        statuses = {row.status for row in parse_rows}
        st = "done" if parse_rows and statuses == {"done"} else ("failed" if "failed" in statuses else "parsing")
        if verbose and not json_mode:
            print_info(f"  Parse status: {st}")

        if st == "done":
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
            )
            return
        if st == "failed":
            failed = next((row for row in parse_rows if row.status == "failed"), None)
            exit_with_error(
                MineruError(failed.error_code if failed else "parse_failed", failed.error_msg if failed else "", None),
                json_mode=json_mode,
            )

    if json_mode:
        timeout_result = result.model_copy(update={"status": "parsing", "tip": "Re-run the same command to continue waiting."})
        _print_parse_json_response(timeout_result, None)
    else:
        print_info(f"Parse still in progress (tier={req_tier}). Check status with: mineru show file {file_path}")


if __name__ != "__main__":

    def _register(app: typer.Typer) -> None:
        app.command()(parse_cmd)


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
) -> None:
    """Fetch and output parsed content for a parse command."""
    sha256 = parse_result.sha256
    tier = parse_result.tier
    page_range = parse_result.page_range
    output_path = _ensure_output_parent(output)
    try:
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
                _print_parse_json_response(parse_result, None, output=exported.output)
                return
            print_success(f"Written to {exported.output}")
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
        if not content.content and content.format == "markdown":
            print_error("No content returned from parse.")
            raise typer.Exit(1)
        if json_mode:
            _print_parse_json_response(parse_result, content)
            return
        output_doc_content_response(content, json_mode=json_mode, output=None, source_path=source_path, read_mode=False, no_marker=no_marker)
    except typer.Exit:
        raise
    except Exception as exc:
        if json_mode:
            exit_with_error(exc, json_mode=True)
        print_error(f"Failed to read content: {exc}")
        raise typer.Exit(1) from None


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


def _print_parse_json_response(
    parse_result: ParseResponse,
    content: DocContentResponse | None,
    *,
    output: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "parse": parse_result.model_dump(mode="json"),
        "content": content.model_dump(mode="json") if content is not None else None,
    }
    if output is not None:
        payload["output"] = _output_info(output)
    print_json(payload)


def output_doc_content_response(
    content: DocContentResponse,
    *,
    json_mode: bool,
    output: str | None,
    source_path: str | None,
    read_mode: bool,
    no_marker: bool,
) -> None:
    output_path = _ensure_output_parent(output)
    if json_mode and output_path and output_path != "-":
        payload = content.model_dump(mode="json")
        if content.format == "image":
            if content.asset is None:
                print_error("No image asset returned.")
                raise typer.Exit(1)
            shutil.copyfile(content.asset.path, output_path)
        else:
            Path(output_path).write_text(content.content, encoding="utf-8")
        payload["content"] = None
        payload["output"] = _output_info(output_path)
        print_json(payload)
        return
    if json_mode:
        print_json(content)
        return
    if content.format == "image":
        if content.asset is None:
            print_error("No image asset returned.")
            raise typer.Exit(1)
        asset_path = content.asset.path
        if output_path and output_path != "-":
            shutil.copyfile(asset_path, output_path)
            print_success(f"Written to {output_path}")
            return
        print(asset_path)
        return
    if output_path and output_path != "-":
        Path(output_path).write_text(content.content, encoding="utf-8")
        print_success(f"Written to {output_path}")
        return
    print(content.content)
    if content.next_request and not no_marker:
        marker = _read_next_marker(content.next_request) if read_mode else _parse_next_marker(source_path, content.next_request)
        if marker:
            print(marker)


def _next_request_dict(next_request: ContentNextRequest | None) -> dict[str, str] | None:
    if next_request is None:
        return None
    result: dict[str, str] = {}
    if next_request.page_range:
        result["page_range"] = next_request.page_range
    if next_request.after:
        result["after"] = next_request.after
    if next_request.locator:
        result["locator"] = next_request.locator
    return result


def _parse_next_marker(path: str | None, next_request: ContentNextRequest) -> str | None:
    if not path:
        return None
    parts = ["mineru", "parse", path]
    if next_request.page_range:
        parts.extend(["--pages", next_request.page_range])
    if next_request.after:
        parts.extend(["--after", next_request.after])
    return f"<!-- Next: {' '.join(parts)} -->"


def _read_next_marker(next_request: ContentNextRequest) -> str | None:
    if not next_request.locator:
        return None
    return f"<!-- Next: mineru read {next_request.locator} -->"
