from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Literal

from ...parser import MinerUApiParser
from ...parser import parse as local_parse
from ...utils.backend_options import resolve_backend_and_effort
from ...utils.ocr_language import validate_public_ocr_lang
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
    inputs: list[str],
    output: str,
    pages: str | None = None,
    format: Literal["markdown", "middle_json", "zip"] = "markdown",
    verbose: bool = False,
    tier: Literal["flash", "standard", "pro"] | None = None,
    backend: str | None = None,
    remote: bool = False,
    remote_url: str | None = None,
    api_key: str | None = None,
    language: str = "ch",
    ocr_mode: Literal["auto", "txt", "ocr"] = "auto",
    effort: Literal["medium", "high"] = "medium",
    disable_table: bool = False,
    disable_formula: bool = False,
    disable_image_analysis: bool = False,
) -> None:
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
    if len(raw_paths) > 1 and Path(output).expanduser().suffix:
        exit_with_message(
            "invalid_request",
            "When input is multiple files or directories, --output must be a directory path.",
            "output",
        )
    paths = expand_input_paths(inputs)
    try:
        ensure_supported_inputs(paths)
        destinations = resolve_batch_output_paths(paths, Path(output).expanduser(), format)
        normalized_language = validate_public_ocr_lang(language)
    except Exception as exc:
        exit_with_message("invalid_request", str(exc))

    if api_url:
        parser = MinerUApiParser(api_url=api_url, api_key=api_key, tier=tier)
        parse_one = partial(parser.parse, page_range=pages or "")
    else:
        try:
            normalized_backend, normalized_effort = (
                resolve_backend_and_effort(backend, effort) if backend is not None else (None, effort)
            )
        except ValueError as exc:
            exit_with_message("invalid_request", str(exc), "backend")
        resolved_tier, resolved_backend = effective_local_tier_and_backend(tier, normalized_backend)
        parse_one = partial(
            local_parse,
            tier=resolved_tier,
            backend=resolved_backend,
            language=normalized_language,
            ocr_mode=ocr_mode,
            effort=normalized_effort,
            disable_table=disable_table,
            disable_formula=disable_formula,
            disable_image_analysis=disable_image_analysis,
            page_range=pages or "",
        )

    for path in paths:
        try:
            result = parse_one(path)
            dest = destinations[path]
            save_parse_result(result, dest, format)
            if verbose:
                print_info(str(parse_result_payload(path, dest, format)))
        except Exception as exc:
            exit_with_message("parse_failed", f"Failed to parse {path}: {exc}")
    print_success(f"Parsed {len(paths)} input(s).")
