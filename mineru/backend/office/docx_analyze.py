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
    docx_path = "/Users/myhloli/projects/20240809magic_pdf/Magic-PDF/mineru/model/docx/drawingml.docx"
    from mineru.data.data_reader_writer import FileBasedDataWriter
    with open(docx_path, 'rb') as f:
        file_bytes = f.read()
    image_writer = FileBasedDataWriter("./output_images")
    middle_json, results = office_docx_analyze(
        file_bytes,
        image_writer=image_writer,
    )

    import json
    print(json.dumps(middle_json, indent=2, ensure_ascii=False))