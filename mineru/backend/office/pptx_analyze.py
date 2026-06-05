# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import time
from io import BytesIO
from typing import Any

from loguru import logger

from ...data.data_reader_writer import FileBasedDataWriter
from ...model.pptx.main import convert_binary
from ...types import PageInfo
from .model_output_to_middle_json import result_to_middle_json


def office_pptx_analyze(file_bytes: bytes, image_writer: Any = None) -> tuple[list[PageInfo], list[Any]]:
    infer_start = time.time()

    file_stream = BytesIO(file_bytes)
    results = convert_binary(file_stream)

    infer_time = round(time.time() - infer_start, 2)
    safe_time = max(infer_time, 0.01)
    logger.debug(f"infer finished, cost: {infer_time}, speed: {round(len(results) / safe_time, 3)} page/s")

    middle_json = result_to_middle_json(results, image_writer)
    return middle_json, results


if __name__ == "__main__":
    # Resolve a default pptx file relative to this script so the example
    # works no matter what the current working directory is when the
    # module is executed.  Allow the user to override the path via a
    # command-line argument for even greater flexibility.
    import argparse
    import json
    from pathlib import Path

    script_root = Path(__file__).resolve().parent.parent.parent.parent
    default_pptx = script_root / "demo" / "office_docs" / "pptx_01.pptx"

    parser = argparse.ArgumentParser(description="Quick demo runner for office_pptx_analyze")
    parser.add_argument(
        "pptx",
        nargs="?",
        default=str(default_pptx),
        help="path to pptx file (defaults to demo/office_docs/pptx_01.pptx relative to project root)",
    )
    parser.add_argument(
        "--output-images",
        help="directory to write image outputs",
        default="./output_images",
    )
    args = parser.parse_args()

    pptx_path = Path(args.pptx)

    with open(pptx_path, "rb") as f:
        file_bytes = f.read()
    image_writer = FileBasedDataWriter(args.output_images)
    middle_json, results = office_pptx_analyze(
        file_bytes,
        image_writer=image_writer,
    )

    logger.info(json.dumps(middle_json, indent=2, ensure_ascii=False))
