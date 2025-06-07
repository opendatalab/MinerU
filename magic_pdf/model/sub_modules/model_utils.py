import time
import torch
from loguru import logger
import numpy as np

from magic_pdf.libs.boxbase import get_minbox_if_overlap_by_ratio
from magic_pdf.libs.clean_memory import clean_memory


def crop_img(input_res, input_np_img, crop_paste_x=0, crop_paste_y=0):

    crop_xmin, crop_ymin = int(input_res['poly'][0]), int(input_res['poly'][1])
    crop_xmax, crop_ymax = int(input_res['poly'][4]), int(input_res['poly'][5])

    # Calculate new dimensions
    crop_new_width = crop_xmax - crop_xmin + crop_paste_x * 2
    crop_new_height = crop_ymax - crop_ymin + crop_paste_y * 2

    # Create a white background array
    return_image = np.ones((crop_new_height, crop_new_width, 3), dtype=np.uint8) * 255

    # Crop the original image using numpy slicing
    cropped_img = input_np_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

    # Paste the cropped image onto the white background
    return_image[crop_paste_y:crop_paste_y + (crop_ymax - crop_ymin),
    crop_paste_x:crop_paste_x + (crop_xmax - crop_xmin)] = cropped_img

    return_list = [crop_paste_x, crop_paste_y, crop_xmin, crop_ymin, crop_xmax, crop_ymax, crop_new_width,
                   crop_new_height]
    return return_image, return_list


def get_coords_and_area(block_with_poly):
    """Extract coordinates and area from a table."""
    xmin, ymin = int(block_with_poly['poly'][0]), int(block_with_poly['poly'][1])
    xmax, ymax = int(block_with_poly['poly'][4]), int(block_with_poly['poly'][5])
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


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes."""
    intersection = calculate_intersection(box1[:4], box2[:4])

    if not intersection:
        return 0

    intersection_xmin, intersection_ymin, intersection_xmax, intersection_ymax = intersection
    intersection_area = (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)

    area1, area2 = box1[4], box2[4]
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0


def is_inside(small_box, big_box, overlap_threshold=0.8):
    """Check if small_box is inside big_box by at least overlap_threshold."""
    intersection = calculate_intersection(small_box[:4], big_box[:4])

    if not intersection:
        return False

    intersection_xmin, intersection_ymin, intersection_xmax, intersection_ymax = intersection
    intersection_area = (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)

    # Check if overlap exceeds threshold
    return intersection_area >= overlap_threshold * small_box[4]


def do_overlap(box1, box2):
    """Check if two boxes overlap."""
    return calculate_intersection(box1[:4], box2[:4]) is not None


def merge_high_iou_tables(table_res_list, layout_res, table_indices, iou_threshold=0.7):
    """Merge tables with IoU > threshold."""
    if len(table_res_list) < 2:
        return table_res_list, table_indices

    table_info = [get_coords_and_area(table) for table in table_res_list]
    merged = True

    while merged:
        merged = False
        i = 0
        while i < len(table_res_list) - 1:
            j = i + 1
            while j < len(table_res_list):
                iou = calculate_iou(table_info[i], table_info[j])

                if iou > iou_threshold:
                    # Merge tables by taking their union
                    x1_min, y1_min, x1_max, y1_max, _ = table_info[i]
                    x2_min, y2_min, x2_max, y2_max, _ = table_info[j]

                    union_xmin = min(x1_min, x2_min)
                    union_ymin = min(y1_min, y2_min)
                    union_xmax = max(x1_max, x2_max)
                    union_ymax = max(y1_max, y2_max)

                    # Create merged table
                    merged_table = table_res_list[i].copy()
                    merged_table['poly'][0] = union_xmin
                    merged_table['poly'][1] = union_ymin
                    merged_table['poly'][2] = union_xmax
                    merged_table['poly'][3] = union_ymin
                    merged_table['poly'][4] = union_xmax
                    merged_table['poly'][5] = union_ymax
                    merged_table['poly'][6] = union_xmin
                    merged_table['poly'][7] = union_ymax

                    # Update layout_res
                    to_remove = [table_indices[j], table_indices[i]]
                    for idx in sorted(to_remove, reverse=True):
                        del layout_res[idx]
                    layout_res.append(merged_table)

                    # Update tracking lists
                    table_indices = [k if k < min(to_remove) else
                                     k - 1 if k < max(to_remove) else
                                     k - 2 if k > max(to_remove) else
                                     len(layout_res) - 1
                                     for k in table_indices
                                     if k not in to_remove]
                    table_indices.append(len(layout_res) - 1)

                    # Update table lists
                    table_res_list.pop(j)
                    table_res_list.pop(i)
                    table_res_list.append(merged_table)

                    # Update table_info
                    table_info = [get_coords_and_area(table) for table in table_res_list]

                    merged = True
                    break
                j += 1

            if merged:
                break
            i += 1

    return table_res_list, table_indices


def filter_nested_tables(table_res_list, overlap_threshold=0.8, area_threshold=0.8):
    """Remove big tables containing multiple smaller tables within them."""
    if len(table_res_list) < 3:
        return table_res_list

    table_info = [get_coords_and_area(table) for table in table_res_list]
    big_tables_idx = []

    for i in range(len(table_res_list)):
        # Find tables inside this one
        tables_inside = [j for j in range(len(table_res_list))
                         if i != j and is_inside(table_info[j], table_info[i], overlap_threshold)]

        # Continue if there are at least 3 tables inside
        if len(tables_inside) >= 3:
            # Check if inside tables overlap with each other
            tables_overlap = any(do_overlap(table_info[tables_inside[idx1]], table_info[tables_inside[idx2]])
                                 for idx1 in range(len(tables_inside))
                                 for idx2 in range(idx1 + 1, len(tables_inside)))

            # If no overlaps, check area condition
            if not tables_overlap:
                total_inside_area = sum(table_info[j][4] for j in tables_inside)
                big_table_area = table_info[i][4]

                if total_inside_area > area_threshold * big_table_area:
                    big_tables_idx.append(i)

    return [table for i, table in enumerate(table_res_list) if i not in big_tables_idx]


def remove_overlaps_min_blocks(res_list):
    #  重叠block，小的不能直接删除，需要和大的那个合并成一个更大的。
    #  删除重叠blocks中较小的那些
    need_remove = []
    for res1 in res_list:
        for res2 in res_list:
            if res1 != res2:
                overlap_box = get_minbox_if_overlap_by_ratio(
                    res1['bbox'], res2['bbox'], 0.8
                )
                if overlap_box is not None:
                    res_to_remove = next(
                        (res for res in res_list if res['bbox'] == overlap_box),
                        None,
                    )
                    if (
                        res_to_remove is not None
                        and res_to_remove not in need_remove
                    ):
                        large_res = res1 if res1 != res_to_remove else res2
                        x1, y1, x2, y2 = large_res['bbox']
                        sx1, sy1, sx2, sy2 = res_to_remove['bbox']
                        x1 = min(x1, sx1)
                        y1 = min(y1, sy1)
                        x2 = max(x2, sx2)
                        y2 = max(y2, sy2)
                        large_res['bbox'] = [x1, y1, x2, y2]
                        need_remove.append(res_to_remove)

    if len(need_remove) > 0:
        for res in need_remove:
            res_list.remove(res)

    return res_list, need_remove


def get_res_list_from_layout_res(layout_res, iou_threshold=0.7, overlap_threshold=0.8, area_threshold=0.8):
    """Extract OCR, table and other regions from layout results."""
    ocr_res_list = []
    text_res_list = []
    table_res_list = []
    table_indices = []
    single_page_mfdetrec_res = []

    # Categorize regions
    for i, res in enumerate(layout_res):
        category_id = int(res['category_id'])

        if category_id in [13, 14]:  # Formula regions
            single_page_mfdetrec_res.append({
                "bbox": [int(res['poly'][0]), int(res['poly'][1]),
                         int(res['poly'][4]), int(res['poly'][5])],
            })
        elif category_id in [0, 2, 4, 6, 7, 3]:  # OCR regions
            ocr_res_list.append(res)
        elif category_id == 5:  # Table regions
            table_res_list.append(res)
            table_indices.append(i)
        elif category_id in [1]:  # Text regions
            res['bbox'] = [int(res['poly'][0]), int(res['poly'][1]), int(res['poly'][4]), int(res['poly'][5])]
            text_res_list.append(res)

    # Process tables: merge high IoU tables first, then filter nested tables
    table_res_list, table_indices = merge_high_iou_tables(
        table_res_list, layout_res, table_indices, iou_threshold)

    filtered_table_res_list = filter_nested_tables(
        table_res_list, overlap_threshold, area_threshold)

    # Remove filtered out tables from layout_res
    if len(filtered_table_res_list) < len(table_res_list):
        kept_tables = set(id(table) for table in filtered_table_res_list)
        to_remove = [table_indices[i] for i, table in enumerate(table_res_list)
                     if id(table) not in kept_tables]

        for idx in sorted(to_remove, reverse=True):
            del layout_res[idx]

    # Remove overlaps in OCR and text regions
    text_res_list, need_remove = remove_overlaps_min_blocks(text_res_list)
    for res in text_res_list:
        # 将res的poly使用bbox重构
        res['poly'] = [res['bbox'][0], res['bbox'][1], res['bbox'][2], res['bbox'][1],
                       res['bbox'][2], res['bbox'][3], res['bbox'][0], res['bbox'][3]]
        # 删除res的bbox
        del res['bbox']

    ocr_res_list.extend(text_res_list)

    if len(need_remove) > 0:
        for res in need_remove:
            del res['bbox']
            layout_res.remove(res)

    return ocr_res_list, filtered_table_res_list, single_page_mfdetrec_res


def clean_vram(device, vram_threshold=8):
    total_memory = get_vram(device)
    if total_memory and total_memory <= vram_threshold:
        gc_start = time.time()
        clean_memory(device)
        gc_time = round(time.time() - gc_start, 2)
        logger.info(f"gc time: {gc_time}")


def get_vram(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # 将字节转换为 GB
        return total_memory
    elif str(device).startswith("npu"):
        import torch_npu
        if torch_npu.npu.is_available():
            total_memory = torch_npu.npu.get_device_properties(device).total_memory / (1024 ** 3)  # 转为 GB
            return total_memory
    else:
        return None