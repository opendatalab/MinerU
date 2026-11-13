from __future__ import annotations

from functools import partial
from pathlib import Path
from tempfile import TemporaryFile
from typing import Literal, cast

import typer

from ...errors import MineruError
from ...filetypes import ensure_tier_supported_for_parse_extension, is_flash_only_parse_extension
from ...parser import MinerUApiParser
from ...parser import parse as local_parse
from ...parser.api_client import _V1APIError
from ...parser.base import ParseResult
from ...parser.page_range import normalize_page_range_input
from ...types import Tier, validate_tier
from ...utils.i18n import t
from ..batch_parse import BatchParseSummary, classify_parse_error, validate_parse_input
from ..common import (
    build_remote_api_url,
    OUTPUT_FILE_SUFFIXES,
    expand_input_paths,
    parse_result_payload,
    resolve_batch_output_paths,
    save_parse_result,
)
from ..errors import exit_with_message
from ..output import print_error, print_info, print_success


def parse_cmd(
    inputs: list[str] = typer.Argument(..., help=t("Input files or directories")),
    output: str = typer.Option(..., "-o", "--output", help=t("Output path; required")),
    pages: str | None = typer.Option(
        None,
        "-p",
        "--pages",
        help=t("PDF pages: '1-5,8', 'r3-r1' (counted from the end), or 'all'; default: all pages"),
    ),
    format: str = typer.Option(
        "markdown",
        "-f",
        "--format",
        help=t("Output format: markdown, middle_json, zip"),
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help=t("Verbose output")),
    tier: str | None = typer.Option(None, "--tier", help=t("Parse tier: flash, basic, standard, advanced")),
    remote: bool = typer.Option(False, "--remote", help=t("Use mineru.net official remote parse service")),
    remote_url: str | None = typer.Option(None, "--remote-url", help=t("Use a custom remote parse service URL")),
    api_key: str | None = typer.Option(None, "--api-key", help=t("API key for remote parse service")),
    ocr_mode: str = typer.Option("auto", "--ocr-mode", help=t("OCR mode: auto, txt, ocr")),
    disable_image_analysis: bool = typer.Option(False, "--disable-image-analysis", help=t("Disable image analysis")),
) -> None:
    """Parse files or directories into markdown, middle JSON, or zip outputs."""
    try:
        pages = normalize_page_range_input(pages)
        if tier is not None:
            validate_tier(tier)
        if ocr_mode not in {"auto", "txt", "ocr"}:
            raise ValueError(t("Unsupported OCR mode: {mode}", mode=ocr_mode))
        if format not in OUTPUT_FILE_SUFFIXES:
            raise ValueError(t("Unsupported format: {format}", format=format))
        api_url = build_remote_api_url(remote, remote_url)
    except (MineruError, ValueError) as exc:
        exit_with_message("invalid_request", str(exc))
    output_format = cast(Literal["markdown", "middle_json", "zip"], format)
    parse_tier = cast(Tier | None, tier)
    parse_ocr_mode = cast(Literal["auto", "txt", "ocr"], ocr_mode)
    if not inputs:
        exit_with_message("invalid_request", t("At least one input path is required."), "inputs")
    raw_paths = [Path(raw).expanduser() for raw in inputs]
    is_batch = any(path.is_dir() for path in raw_paths) or len(raw_paths) > 1
    try:
        paths = expand_input_paths(inputs)
        if not paths:
            raise ValueError(t("No input files found."))
        destinations = resolve_batch_output_paths(paths, Path(output).expanduser(), output_format, batch=is_batch)
        # 在模型加载或远程请求前验证输出根可写；不截断已有结果文件。
        output_parent = next(iter(destinations.values())).parent
        output_parent.mkdir(parents=True, exist_ok=True)
        with TemporaryFile(prefix=".mineru-write-", dir=output_parent):
            pass
    except Exception as exc:
        exit_with_message("invalid_request", str(exc))

    if api_url:
        parser = MinerUApiParser(
            api_url=api_url, api_key=api_key, tier=parse_tier, ocr_mode=parse_ocr_mode, include_images=True
        )
        parse_one = partial(parser.parse, page_range=pages or "")
    else:

        def parse_one(path: Path) -> ParseResult:
            """保持单文件显式 tier 校验，批量原生格式始终使用 Flash。"""
            path_tier: Tier = validate_tier(parse_tier) if parse_tier is not None else "standard"
            if is_flash_only_parse_extension(path):
                if not is_batch and parse_tier is not None:
                    ensure_tier_supported_for_parse_extension(path_tier, path)
                path_tier = "flash"
            return local_parse(
                path,
                tier=path_tier,
                ocr_mode=parse_ocr_mode,
                image_analysis=not disable_image_analysis,
                page_range=pages or "",
            )

    summary = BatchParseSummary()
    for path in paths:
        try:
            if not is_batch and not api_url and parse_tier is not None and is_flash_only_parse_extension(path):
                ensure_tier_supported_for_parse_extension(parse_tier, path)
            validate_parse_input(path)
            result = parse_one(path)
        except Exception as exc:
            disposition, code, message = classify_parse_error(exc)
            if not is_batch:
                # 单文件错误保留本地或远程协议中的参数定位，方便调用方修正请求。
                param = exc.param if isinstance(exc, (MineruError, _V1APIError)) else None
                exit_with_message(code, t("Failed to parse {path}: {error}", path=path, error=message), param)
            if disposition == "skipped":
                summary.skipped += 1
                print_info(t("Skipped {path}: [{code}] {error}", path=path, code=code, error=message))
            else:
                summary.failed += 1
                print_error(t("Failed to parse {path}: {error}", path=path, error=f"[{code}] {message}"))
            if disposition == "abort":
                summary.aborted = True
                break
            continue
        dest = destinations[path]
        try:
            save_parse_result(result, dest, output_format)
        except Exception as exc:
            if not is_batch:
                exit_with_message("result_packaging_failed", str(exc), "output")
            summary.failed += 1
            summary.aborted = True
            print_error(t("Failed to write {path}: {error}", path=dest, error=exc))
            break
        summary.succeeded += 1
        if verbose:
            print_info(str(parse_result_payload(path, dest, output_format)))
    if is_batch:
        print_info(summary.message(len(paths)))
        if summary.exit_code:
            raise typer.Exit(summary.exit_code)
    else:
        print_success(t("Parsed {count} input(s).", count=summary.succeeded))
