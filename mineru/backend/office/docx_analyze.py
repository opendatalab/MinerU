# Copyright (c) Opendatalab. All rights reserved.
import time
from io import BytesIO

from loguru import logger

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

    # middle_json = result_to_middle_json(
    #     results,
    #     image_writer,
    # )
    middle_json= {"pdf_info": results}

    return middle_json, results