"""Command-line interface for the local Knowhere artifact adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from mineru.integrations.knowhere.contract import (
    KnowhereExportError,
    KnowhereExportOptions,
)
from mineru.integrations.knowhere.runner import run_knowhere_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mineru-knowhere-export",
        description="Parse one local PDF or DOCX for a Knowhere artifact consumer.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="pipeline")
    parser.add_argument("--method", default="auto")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--server-url")
    parser.add_argument(
        "--formula-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--table-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--image-analysis-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = KnowhereExportOptions(
        input_path=args.input,
        output_root=args.output,
        backend=args.backend,
        method=args.method,
        language=args.lang,
        formula_enabled=args.formula_enabled,
        table_enabled=args.table_enabled,
        image_analysis_enabled=args.image_analysis_enabled,
        offline=args.offline,
        server_url=args.server_url,
    )
    try:
        manifest_path = run_knowhere_export(options)
    except KnowhereExportError as error:
        print(f"mineru-knowhere-export: {error}", file=sys.stderr)
        return 2
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

