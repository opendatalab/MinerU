# Copyright (c) Opendatalab. All rights reserved.
import gc
import math
import os
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image

from ..types import BBox, IntBBox

try:
    import torch
    import torch_npu
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _get_bbox(item: dict[str, Any]) -> BBox:
    bbox = item["bbox"]
    assert bbox is not None
    xmin, ymin, xmax, ymax = bbox
    return float(xmin), float(ymin), float(xmax), float(ymax)


def _get_int_bbox(item: dict[str, Any]) -> IntBBox:
    xmin, ymin, xmax, ymax = _get_bbox(item)
    return math.floor(xmin), math.floor(ymin), math.ceil(xmax), math.ceil(ymax)


def crop_img(
    input_res: dict[str, Any],
    input_img: Image.Image | np.ndarray,
    crop_paste_x: int = 0,
    crop_paste_y: int = 0,
) -> tuple[Image.Image | np.ndarray, list[int]]:
    crop_xmin, crop_ymin, crop_xmax, crop_ymax = _get_int_bbox(input_res)

    # Calculate new dimensions
    crop_new_width = crop_xmax - crop_xmin + crop_paste_x * 2
    crop_new_height = crop_ymax - crop_ymin + crop_paste_y * 2

    if isinstance(input_img, np.ndarray):
        # Create a white background array
        return_image = np.ones((crop_new_height, crop_new_width, 3), dtype=np.uint8) * 255

        # Crop the original image using numpy slicing
        cropped_img = input_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        # Paste the cropped image onto the white background
        return_image[
            crop_paste_y : crop_paste_y + (crop_ymax - crop_ymin), crop_paste_x : crop_paste_x + (crop_xmax - crop_xmin)
        ] = cropped_img
    else:
        # Create a white background array
        return_image = Image.new("RGB", (crop_new_width, crop_new_height), "white")
        # Crop image
        crop_box = (crop_xmin, crop_ymin, crop_xmax, crop_ymax)
        cropped_img = input_img.crop(crop_box)
        return_image.paste(cropped_img, (crop_paste_x, crop_paste_y))

    return_list = [crop_paste_x, crop_paste_y, crop_xmin, crop_ymin, crop_xmax, crop_ymax, crop_new_width, crop_new_height]
    return return_image, return_list


def clean_memory(device: str = "cuda") -> None:
    if not _TORCH_AVAILABLE:
        gc.collect()
        return
    if str(device).startswith("cuda"):
        if torch.cuda.is_available():  # type: ignore
            torch.cuda.empty_cache()  # type: ignore
            # torch.cuda.ipc_collect()
    elif str(device).startswith("npu"):
        if torch_npu.npu.is_available():  # type: ignore
            torch_npu.npu.empty_cache()  # type: ignore
    elif str(device).startswith("mps"):
        torch.mps.empty_cache()  # type: ignore
    elif str(device).startswith("gcu"):
        if torch.gcu.is_available():  # type: ignore
            torch.gcu.empty_cache()  # type: ignore
    elif str(device).startswith("musa"):
        if torch.musa.is_available():  # type: ignore
            torch.musa.empty_cache()  # type: ignore
    elif str(device).startswith("mlu"):
        if torch.mlu.is_available():  # type: ignore
            torch.mlu.empty_cache()  # type: ignore
    elif str(device).startswith("sdaa"):
        if torch.sdaa.is_available():  # type: ignore
            torch.sdaa.empty_cache()  # type: ignore
    gc.collect()


def get_vram(device: str) -> int:
    env_vram = os.getenv("MINERU_VIRTUAL_VRAM_SIZE")

    # 如果环境变量已配置,尝试解析并返回
    if env_vram is not None:
        try:
            total_memory = int(env_vram)
            if total_memory > 0:
                return total_memory
            else:
                logger.warning(f"MINERU_VIRTUAL_VRAM_SIZE value '{env_vram}' is not positive, falling back to auto-detection")
        except ValueError:
            logger.warning(
                f"MINERU_VIRTUAL_VRAM_SIZE value '{env_vram}' is not a valid integer, falling back to auto-detection"
            )

    # 环境变量未配置或配置错误,根据device自动获取
    total_memory = 1
    if not _TORCH_AVAILABLE:
        return total_memory
    if torch.cuda.is_available() and str(device).startswith("cuda"):  # type: ignore
        total_memory = round(torch.cuda.get_device_properties(device).total_memory / (1024**3))  # type: ignore  # 将字节转换为 GB
    elif str(device).startswith("npu"):
        if torch_npu.npu.is_available():  # type: ignore
            total_memory = round(torch_npu.npu.get_device_properties(device).total_memory / (1024**3))  # type: ignore  # 转为 GB
    elif str(device).startswith("gcu"):
        if torch.gcu.is_available():  # type: ignore
            total_memory = round(torch.gcu.get_device_properties(device).total_memory / (1024**3))  # type: ignore  # 转为 GB
    elif str(device).startswith("musa"):
        if torch.musa.is_available():  # type: ignore
            total_memory = round(torch.musa.get_device_properties(device).total_memory / (1024**3))  # type: ignore  # 转为 GB
    elif str(device).startswith("mlu"):
        if torch.mlu.is_available():  # type: ignore
            total_memory = round(torch.mlu.get_device_properties(device).total_memory / (1024**3))  # type: ignore  # 转为 GB
    elif str(device).startswith("sdaa"):
        if torch.sdaa.is_available():  # type: ignore
            total_memory = round(torch.sdaa.get_device_properties(device).total_memory / (1024**3))  # type: ignore  # 转为 GB

    return total_memory
