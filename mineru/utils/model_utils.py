import math
import os
import time
import gc
from PIL import Image
from loguru import logger
import numpy as np

try:
    import torch
    import torch_npu
except ImportError:
    pass

TEXT_REGION_LABELS = {
    "abstract",
    "algorithm",
    "aside_text",
    "content",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "number",
    "paragraph_title",
    "reference_content",
    "text",
    "vertical_text",
    "vision_footnote",
}


def _get_bbox(item):
    bbox = item.get("bbox")
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        return float(xmin), float(ymin), float(xmax), float(ymax)


def _get_int_bbox(item):
    xmin, ymin, xmax, ymax = _get_bbox(item)
    return math.floor(xmin), math.floor(ymin), math.ceil(xmax), math.ceil(ymax)


def crop_img(input_res, input_img, crop_paste_x=0, crop_paste_y=0):
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
        return_image[crop_paste_y:crop_paste_y + (crop_ymax - crop_ymin),
        crop_paste_x:crop_paste_x + (crop_xmax - crop_xmin)] = cropped_img
    else:
        # Create a white background array
        return_image = Image.new('RGB', (crop_new_width, crop_new_height), 'white')
        # Crop image
        crop_box = (crop_xmin, crop_ymin, crop_xmax, crop_ymax)
        cropped_img = input_img.crop(crop_box)
        return_image.paste(cropped_img, (crop_paste_x, crop_paste_y))

    return_list = [crop_paste_x, crop_paste_y, crop_xmin, crop_ymin, crop_xmax, crop_ymax, crop_new_width,
                   crop_new_height]
    return return_image, return_list


def get_coords_and_area(block_with_poly):
    """Extract coordinates and area from a table."""
    xmin, ymin, xmax, ymax = _get_bbox(block_with_poly)
    area = (xmax - xmin) * (ymax - ymin)
    return xmin, ymin, xmax, ymax, area


def calculate_intersection(box1, box2):
    """Calculate intersection coordinates between two boxes."""
    intersection_xmin = max(box1[0], box2[0])
    intersection_ymin = max(box1[1], box2[1])
    intersection_xmax = min(box1[2], box2[2])
    intersection_ymax = min(box1[3], box2[3])

    # Check if intersection is valid
    if intersection_xmax <= intersection_xmin or intersection_ymax <= intersection_ymin:
        return None

    return intersection_xmin, intersection_ymin, intersection_xmax, intersection_ymax


def is_inside(small_box, big_box, overlap_threshold=0.8):
    """Check if small_box is inside big_box by at least overlap_threshold."""
    intersection = calculate_intersection(small_box[:4], big_box[:4])

    if not intersection:
        return False

    intersection_xmin, intersection_ymin, intersection_xmax, intersection_ymax = intersection
    intersection_area = (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)

    # Check if overlap exceeds threshold
    return intersection_area >= overlap_threshold * small_box[4]


def remove_nested_ocr_text_blocks(
    ocr_res_list,
    layout_res,
    overlap_threshold=0.8,
    min_area_ratio=1.01,
):
    """Remove OCR candidate text blocks that are contained by any larger layout block."""
    if not ocr_res_list or len(layout_res) < 2:
        return ocr_res_list, []

    layout_info = [(block, get_coords_and_area(block)) for block in layout_res]
    blocks_to_remove = []

    for text_block in ocr_res_list:
        text_box = get_coords_and_area(text_block)
        text_area = text_box[4]
        for parent_block, parent_box in layout_info:
            if parent_block is text_block:
                continue
            if parent_box[4] <= text_area * min_area_ratio:
                continue
            if is_inside(text_box, parent_box, overlap_threshold):
                blocks_to_remove.append(text_block)
                break

    remove_ids = {id(block) for block in blocks_to_remove}
    filtered_ocr_res_list = [
        block for block in ocr_res_list if id(block) not in remove_ids
    ]
    return filtered_ocr_res_list, blocks_to_remove


def get_res_list_from_layout_res(layout_res, overlap_threshold=0.8):
    """Extract OCR, table and other regions from layout results."""
    ocr_res_list = []
    text_res_list = []
    table_res_list = []
    single_page_mfdetrec_res = []

    # Categorize regions
    for i, res in enumerate(layout_res):
        label = res.get("label")

        if label in ["display_formula", "inline_formula"]:
            xmin, ymin, xmax, ymax = _get_bbox(res)
            single_page_mfdetrec_res.append({
                "bbox": [xmin, ymin, xmax, ymax],
            })
        elif label == "table":
            table_res_list.append(res)
        elif label in TEXT_REGION_LABELS:
            text_res_list.append(res)

    ocr_res_list.extend(text_res_list)

    ocr_res_list, nested_text_need_remove = remove_nested_ocr_text_blocks(
        ocr_res_list,
        layout_res,
        overlap_threshold=overlap_threshold,
    )
    nested_remove_ids = {id(res) for res in nested_text_need_remove}
    if nested_remove_ids:
        layout_res[:] = [res for res in layout_res if id(res) not in nested_remove_ids]

    return ocr_res_list, table_res_list, single_page_mfdetrec_res


def clean_memory(device='cuda'):
    if str(device).startswith("cuda"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    elif str(device).startswith("npu"):
        if torch_npu.npu.is_available():
            torch_npu.npu.empty_cache()
    elif str(device).startswith("mps"):
        torch.mps.empty_cache()
    elif str(device).startswith("gcu"):
        if torch.gcu.is_available():
            torch.gcu.empty_cache()
    elif str(device).startswith("musa"):
        if torch.musa.is_available():
            torch.musa.empty_cache()
    elif str(device).startswith("mlu"):
        if torch.mlu.is_available():
            torch.mlu.empty_cache()
    elif str(device).startswith("sdaa"):
        if torch.sdaa.is_available():
            torch.sdaa.empty_cache()  
    gc.collect()


def clean_vram(device, vram_threshold=8):
    total_memory = get_vram(device)
    if total_memory and total_memory <= vram_threshold:
        gc_start = time.time()
        clean_memory(device)
        gc_time = round(time.time() - gc_start, 2)
        logger.debug(f"gc time: {gc_time}")


def get_vram(device) -> int:
    env_vram = os.getenv("MINERU_VIRTUAL_VRAM_SIZE")

    # 如果环境变量已配置,尝试解析并返回
    if env_vram is not None:
        try:
            total_memory = int(env_vram)
            if total_memory > 0:
                return total_memory
            else:
                logger.warning(
                    f"MINERU_VIRTUAL_VRAM_SIZE value '{env_vram}' is not positive, falling back to auto-detection")
        except ValueError:
            logger.warning(
                f"MINERU_VIRTUAL_VRAM_SIZE value '{env_vram}' is not a valid integer, falling back to auto-detection")

    # 环境变量未配置或配置错误,根据device自动获取
    total_memory = 1
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        total_memory = round(torch.cuda.get_device_properties(device).total_memory / (1024 ** 3))  # 将字节转换为 GB
    elif str(device).startswith("npu"):
        if torch_npu.npu.is_available():
            total_memory = round(torch_npu.npu.get_device_properties(device).total_memory / (1024 ** 3))  # 转为 GB
    elif str(device).startswith("gcu"):
        if torch.gcu.is_available():
            total_memory = round(torch.gcu.get_device_properties(device).total_memory / (1024 ** 3))  # 转为 GB
    elif str(device).startswith("musa"):
        if torch.musa.is_available():
            total_memory = round(torch.musa.get_device_properties(device).total_memory / (1024 ** 3))  # 转为 GB
    elif str(device).startswith("mlu"):
        if torch.mlu.is_available():
            total_memory = round(torch.mlu.get_device_properties(device).total_memory / (1024 ** 3))  # 转为 GB
    elif str(device).startswith("sdaa"):
        if torch.sdaa.is_available():
            total_memory = round(torch.sdaa.get_device_properties(device).total_memory / (1024 ** 3))  # 转为 GB          

    return total_memory
