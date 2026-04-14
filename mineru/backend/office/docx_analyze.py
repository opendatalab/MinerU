# Copyright (c) Opendatalab. All rights reserved.
import time
from io import BytesIO

from loguru import logger
from mineru.backend.office.model_output_to_middle_json import result_to_middle_json

from mineru.model.docx.main import convert_binary


def office_docx_analyze(
        file_bytes,
        image_writer=None
):
    infer_start = time.time()

    file_stream = BytesIO(file_bytes)
    results = convert_binary(file_stream)

    infer_time = round(time.time() - infer_start, 2)
    safe_time = max(infer_time, 0.01)
    logger.debug(f"infer finished, cost: {infer_time}, speed: {round(len(results) / safe_time, 3)} page/s")

    middle_json = result_to_middle_json(
        results,
        image_writer,
    )

    return middle_json, results

if __name__ == '__main__':
    # Resolve a default docx file relative to this script so the example
    # works no matter what the current working directory is when the
    # module is executed.  Allow the user to override the path via a
    # command-line argument for even greater flexibility.
    from pathlib import Path
    import argparse

    script_root = Path(__file__).resolve().parent.parent.parent.parent
    default_docx = script_root / "demo" / "office_docs" / "docx_01.docx"

    parser = argparse.ArgumentParser(
        description="Quick demo runner for office_docx_analyze"
    )
    parser.add_argument(
        "docx",
        nargs="?",
        default=str(default_docx),
        help="path to docx file (defaults to demo/office_docs/docx_01.docx relative to project root)"
    )
    parser.add_argument(
        "--output-images",
        help="directory to write image outputs",
        default="./output_images"
    )
    args = parser.parse_args()

    docx_path = Path(args.docx)
    from mineru.data.data_reader_writer import FileBasedDataWriter

    with open(docx_path, 'rb') as f:
        file_bytes = f.read()
    image_writer = FileBasedDataWriter(args.output_images)
    middle_json, results = office_docx_analyze(
        file_bytes,
        image_writer=image_writer,
    )

    import json
    logger.info(json.dumps(middle_json, indent=2, ensure_ascii=False))
