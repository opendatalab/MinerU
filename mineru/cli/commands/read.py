"""mineru read — read doclib content by locator."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import ContentNextRequest, DocContentResponse, ImageFormat
from ...errors import MineruError
from ..contracts import CliContext, CliResult
from ..runtime import cli_ok, run_cli
from .parse import _append_next_marker, _ensure_output_parent, _output_info


@dataclass(frozen=True)
class ReadTextOutput:
    text: str


def read_cmd(
    locator: str = typer.Argument(..., help="Doclib locator, e.g. doc:ab12cd3/tier:basic/page:4"),
    context: int = typer.Option(0, "--context", help="Read N pages/blocks before and after the locator"),
    limit: int = typer.Option(30000, "--limit", help="Soft character limit for STDOUT content"),
    format: Literal["markdown", "image"] = typer.Option("markdown", "-f", "--format", help="Output format: markdown, image"),
    output: str = typer.Option(None, "-o", "--output", help="Output path; creates parent directories"),
    no_marker: bool = typer.Option(False, "--no-marker", help="Omit continuation marker from output"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Read parsed doclib content by locator."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(
        ctx,
        lambda: _read(
            locator,
            context=context,
            limit=limit,
            format=format,
            output=output,
            no_marker=no_marker,
            json_mode=json_mode,
        ),
    )


def _read(
    locator: str,
    *,
    context: int,
    limit: int,
    format: Literal["markdown", "image"],
    output: str | None,
    no_marker: bool,
    json_mode: bool,
) -> DocContentResponse | dict[str, object] | CliResult[ReadTextOutput] | CliResult[None]:
    image_format = _image_format_for_output(format=format, output=output) if format == "image" else "jpeg"
    client = DoclibClient(timeout=60)
    if format == "image":
        content = client.read_content(
            locator,
            context=context,
            limit=limit,
            format=format,
            image_format=image_format,
            no_marker=no_marker,
        )
    else:
        content = client.read_content(
            locator,
            context=context,
            limit=limit,
            format=format,
            no_marker=no_marker,
        )
    return _prepare_read_output(content, json_mode=json_mode, output=output, no_marker=no_marker)


def _prepare_read_output(
    content: DocContentResponse,
    *,
    json_mode: bool,
    output: str | None,
    no_marker: bool,
) -> DocContentResponse | dict[str, object] | CliResult[ReadTextOutput] | CliResult[None]:
    output_path = _ensure_output_parent(output)
    if json_mode and output_path and output_path != "-":
        payload: dict[str, object] = content.model_dump(mode="json")
        if content.format == "image":
            if content.asset is None:
                raise MineruError("asset_not_available", "No image asset returned.", "format")
            shutil.copyfile(content.asset.path, output_path)
        else:
            Path(output_path).write_text(content.content, encoding="utf-8")
        payload["content"] = None
        payload["output"] = _output_info(output_path)
        return payload

    if json_mode:
        return content

    if content.format == "image":
        if content.asset is None:
            raise MineruError("asset_not_available", "No image asset returned.", "format")
        asset_path = content.asset.path
        if output_path and output_path != "-":
            shutil.copyfile(asset_path, output_path)
            return cli_ok(ReadTextOutput(f"Written to {output_path}"), render=_render_read_text)
        return cli_ok(ReadTextOutput(asset_path), render=_render_read_text)

    if output_path and output_path != "-":
        Path(output_path).write_text(content.content, encoding="utf-8")
        return cli_ok(ReadTextOutput(f"Written to {output_path}"), render=_render_read_text)

    if not content.content and content.content_ranges:
        message = "No renderable content in requested pages."
        if content.next_request and not no_marker:
            marker = _read_next_marker(content.next_request)
            if marker:
                return cli_ok(ReadTextOutput(_append_next_marker(message, marker)), render=_render_read_text)
        return cli_ok(None, notices=[message])

    if content.next_request and not no_marker:
        marker = _read_next_marker(content.next_request)
        if marker:
            return cli_ok(ReadTextOutput(_append_next_marker(content.content, marker)), render=_render_read_text)

    return cli_ok(ReadTextOutput(content.content), render=_render_read_text)


def _render_read_text(data: ReadTextOutput) -> str:
    return data.text


def _image_format_for_output(*, format: Literal["markdown", "image"], output: str | None) -> ImageFormat:
    if format != "image" or output is None:
        return "jpeg"
    if output == "-":
        raise MineruError(
            "image_output_extension_unsupported",
            "Image output requires a file path ending with .png, .jpg, .jpeg, or .webp; stdout is not supported.",
            "output",
        )
    suffix = Path(output).suffix.lower()
    if suffix == ".png":
        return "png"
    if suffix in {".jpg", ".jpeg"}:
        return "jpeg"
    if suffix == ".webp":
        return "webp"
    raise MineruError(
        "image_output_extension_unsupported",
        "Image output path must end with .png, .jpg, .jpeg, or .webp.",
        "output",
    )


def _read_next_marker(next_request: ContentNextRequest) -> str | None:
    if not next_request.locator:
        return None
    return f"<!-- Next: mineru read {next_request.locator} -->"
