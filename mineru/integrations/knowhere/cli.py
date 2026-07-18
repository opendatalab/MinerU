"""Command-line interface for the local Knowhere artifact adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from mineru.integrations.knowhere.contract import (
    CanonicalManifestOptions,
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
    parser.add_argument(
        "--canonical-manifest",
        action="store_true",
        help="Also emit the source-owned document-extraction-manifest-v1 contract.",
    )
    parser.add_argument("--source-id")
    parser.add_argument("--source-version-id")
    parser.add_argument("--extraction-run-id")
    parser.add_argument("--accelerator-profile", default="unknown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    canonical_manifest = None
    if args.canonical_manifest:
        missing = [
            option
            for option, value in (
                ("--source-id", args.source_id),
                ("--source-version-id", args.source_version_id),
                ("--extraction-run-id", args.extraction_run_id),
            )
            if not value or not value.strip()
        ]
        if missing:
            print(
                "mineru-knowhere-export: --canonical-manifest requires "
                + ", ".join(missing),
                file=sys.stderr,
            )
            return 2
        canonical_manifest = CanonicalManifestOptions(
            source_id=args.source_id,
            source_version_id=args.source_version_id,
            extraction_run_id=args.extraction_run_id,
            accelerator_profile=args.accelerator_profile,
        )
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
        canonical_manifest=canonical_manifest,
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
