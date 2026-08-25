from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Literal, cast

import typer

from ...filetypes import ensure_tier_supported_for_parse_extension, is_flash_only_parse_extension
from ...parser import MinerUApiParser
from ...parser import parse as local_parse
from ...types import Tier
from ...parser.tier import normalize_backend
from ..common import (
    build_remote_api_url,
    effective_local_tier_and_backend,
    ensure_supported_inputs,
    expand_input_paths,
    parse_result_payload,
    resolve_batch_output_paths,
    save_parse_result,
)
from ..errors import exit_with_message
from ..output import print_info, print_success


def parse_cmd(
    inputs: list[str] = typer.Argument(..., help="Input files or directories"),
    output: str = typer.Option(..., "-o", "--output", help="Output path; required"),
    pages: str | None = typer.Option(None, "-p", "--pages", help="Page range, e.g. '1~5' or 'all'"),
    format: str = typer.Option(
        "markdown",
        "-f",
        "--format",
        help="Output format: markdown, middle_json, zip",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
    tier: str | None = typer.Option(None, "--tier", help="Parse tier: flash, basic, standard, advanced"),
    backend: str | None = typer.Option(None, "--backend", help="Expert backend override"),
    remote: bool = typer.Option(False, "--remote", help="Use mineru.net official remote parse service"),
    remote_url: str | None = typer.Option(None, "--remote-url", help="Use a custom remote parse service URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="API key for remote parse service"),
    ocr_mode: str = typer.Option("auto", "--ocr-mode", help="OCR mode: auto, txt, ocr"),
    disable_image_analysis: bool = typer.Option(False, "--disable-image-analysis", help="Disable image analysis"),
) -> None:
    """Parse files or directories into markdown, middle JSON, or zip outputs."""
    output_format = cast(Literal["markdown", "middle_json", "zip"], format)
    parse_tier = cast(Tier | None, tier)
    parse_ocr_mode = cast(Literal["auto", "txt", "ocr"], ocr_mode)
    if not inputs:
        exit_with_message("invalid_request", "At least one input path is required.", "inputs")
    if remote and backend is not None:
        exit_with_message("invalid_request", "--backend is not allowed in remote mode.", "backend")
    api_url = build_remote_api_url(remote, remote_url)
    raw_paths = [Path(raw).expanduser() for raw in inputs]
    if any(path.is_dir() for path in raw_paths) and Path(output).expanduser().suffix:
        exit_with_message(
            "invalid_request",
            "When input is multiple files or directories, --output must be a directory path.",
            "output",
        )
    has_directory_input = any(path.is_dir() for path in raw_paths)
    if len(raw_paths) > 1 and Path(output).expanduser().suffix:
        exit_with_message(
            "invalid_request",
            "When input is multiple files or directories, --output must be a directory path.",
            "output",
        )
    paths = expand_input_paths(inputs)
    try:
        ensure_supported_inputs(paths)
        destinations = resolve_batch_output_paths(paths, Path(output).expanduser(), output_format)
    except Exception as exc:
        exit_with_message("invalid_request", str(exc))

    if api_url:
        parser = MinerUApiParser(api_url=api_url, api_key=api_key, tier=parse_tier, include_images=True)
        parse_one = partial(parser.parse, page_range=pages or "")
    else:
        try:
            normalized_backend = normalize_backend(backend) if backend is not None else None
        except ValueError as exc:
            exit_with_message("invalid_request", str(exc), "backend")
        is_batch = has_directory_input or len(paths) > 1

        def parse_one(path: Path) -> object:
            path_tier: Tier
            if is_flash_only_parse_extension(path):
                if not is_batch and (parse_tier is not None or normalized_backend is not None):
                    resolved_tier, _resolved_backend = effective_local_tier_and_backend(parse_tier, normalized_backend)
                    ensure_tier_supported_for_parse_extension(resolved_tier, path)
                path_tier = "flash"
            else:
                path_tier, _path_backend = effective_local_tier_and_backend(parse_tier, normalized_backend)
            return local_parse(
                path,
                tier=path_tier,
                ocr_mode=parse_ocr_mode,
                image_analysis=not disable_image_analysis,
                page_range=pages or "",
            )

    for path in paths:
        try:
            result = parse_one(path)
            dest = destinations[path]
            save_parse_result(result, dest, output_format)
            if verbose:
                print_info(str(parse_result_payload(path, dest, output_format)))
        except Exception as exc:
            exit_with_message("parse_failed", f"Failed to parse {path}: {exc}")
    print_success(f"Parsed {len(paths)} input(s).")
