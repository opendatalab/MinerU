# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import time
from io import BytesIO
from typing import Any

from loguru import logger

from ...model.flash import DocxModel, PptxModel, XlsxModel
from ...types import PageInfo
from ...utils.image_payload import ImagePayloadCache
from .model_output_to_middle_json import result_to_middle_json


_OFFICE_MODEL_MAP = {
    "docx": DocxModel,
    "pptx": PptxModel,
    "xlsx": XlsxModel,
}


def analyze(
    file_bytes: bytes,
    file_suffix: str,
    image_cache: ImagePayloadCache | None = None,
) -> tuple[list[PageInfo], list[Any]]:
    """根据 Office 文件后缀调用对应模型，并返回中间结果与模型结果。"""

    model_class = _OFFICE_MODEL_MAP.get(file_suffix)
    if model_class is None:
        raise ValueError(f"Unsupported office suffix: {file_suffix!r}")

    infer_start = time.time()

    file_stream = BytesIO(file_bytes)
    results = model_class().predict(file_stream)

    infer_time = round(time.time() - infer_start, 2)
    safe_time = max(infer_time, 0.01)
    logger.debug(f"infer finished, cost: {infer_time}, speed: {round(len(results) / safe_time, 3)} page/s")

    middle_json = result_to_middle_json(results, image_cache=image_cache)
    return middle_json, results
