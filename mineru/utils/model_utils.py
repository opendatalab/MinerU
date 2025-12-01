import os
import time
import gc
from PIL import Image
from loguru import logger
import numpy as np

from mineru.utils.boxbase import get_minbox_if_overlap_by_ratio

try:
    import torch
    import torch_npu
except ImportError:
    pass


def crop_img(input_res, input_img, crop_paste_x=0, crop_paste_y=0):

    crop_xmin, crop_ymin = int(input_res['poly'][0]), int(input_res['poly'][1])
    crop_xmax, crop_ymax = int(input_res['poly'][4]), int(input_res['poly'][5])

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
                    merged_table['poly'] = [
                        union_xmin, union_ymin, union_xmax, union_ymin,
                        union_xmax, union_ymax, union_xmin, union_ymax
                    ]
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

    for res in res_list:
        res['bbox'] = [int(res['poly'][0]), int(res['poly'][1]), int(res['poly'][4]), int(res['poly'][5])]

    # 重叠block，小的不能直接删除，需要和大的那个合并成一个更大的。
    # 删除重叠blocks中较小的那些
    need_remove = []
    for i in range(len(res_list)):
        # 如果当前元素已在需要移除列表中，则跳过
        if res_list[i] in need_remove:
            continue

        for j in range(i + 1, len(res_list)):
            # 如果比较对象已在需要移除列表中，则跳过
            if res_list[j] in need_remove:
                continue

            overlap_box = get_minbox_if_overlap_by_ratio(
                res_list[i]['bbox'], res_list[j]['bbox'], 0.8
            )

            if overlap_box is not None:

                # 根据重叠框确定哪个是小块，哪个是大块
                if overlap_box == res_list[i]['bbox']:
                    small_res, large_res = res_list[i], res_list[j]
                elif overlap_box == res_list[j]['bbox']:
                    small_res, large_res = res_list[j], res_list[i]
                else:
                    continue  # 如果重叠框与任一块都不匹配，跳过处理

                if small_res['score'] <= large_res['score']:
                    # 如果小块的分数低于大块，则小块为需要移除的块
                    if small_res is not None and small_res not in need_remove:
                        # 更新大块的边界为两者的并集
                        x1, y1, x2, y2 = large_res['bbox']
                        sx1, sy1, sx2, sy2 = small_res['bbox']
                        x1 = min(x1, sx1)
                        y1 = min(y1, sy1)
                        x2 = max(x2, sx2)
                        y2 = max(y2, sy2)
                        large_res['bbox'] = [x1, y1, x2, y2]
                        need_remove.append(small_res)
                else:
                    # 如果大块的分数低于小块，则大块为需要移除的块, 这时不需要更新小块的边界
                    if large_res is not None and large_res not in need_remove:
                        need_remove.append(large_res)

    # 从列表中移除标记的元素
    for res in need_remove:
        res_list.remove(res)
        del res['bbox']  # 删除bbox字段

    for res in res_list:
        # 将res的poly使用bbox重构
        res['poly'] = [res['bbox'][0], res['bbox'][1], res['bbox'][2], res['bbox'][1],
                       res['bbox'][2], res['bbox'][3], res['bbox'][0], res['bbox'][3]]
        # 删除res的bbox
        del res['bbox']

    return res_list, need_remove


def remove_overlaps_low_confidence_blocks(combined_res_list, overlap_threshold=0.8):
    """
    Remove low-confidence blocks that overlap with other blocks.

    This function identifies and removes blocks with low confidence scores that overlap
    with other blocks. It calculates the coordinates and area of each block, and checks
    for overlaps based on a specified threshold. Blocks that meet the criteria for removal
    are returned in a list.

    Parameters:
        combined_res_list (list): A list of blocks, where each block is a dictionary containing
            keys like 'poly' (polygon coordinates) and optionally 'score' (confidence score).
        overlap_threshold (float): The threshold for determining overlap between blocks. Default is 0.8.

    Returns:
        list: A list of blocks to be removed, based on the overlap and confidence criteria.
    """
    # 计算每个block的坐标和面积
    block_info = []
    for block in combined_res_list:
        xmin, ymin = int(block['poly'][0]), int(block['poly'][1])
        xmax, ymax = int(block['poly'][4]), int(block['poly'][5])
        area = (xmax - xmin) * (ymax - ymin)
        score = block.get('score', 0.5)  # 如果没有score字段，默认为0.5
        block_info.append((xmin, ymin, xmax, ymax, area, score, block))

    blocks_to_remove = []
    marked_indices = set()  # 跟踪已标记为删除的block索引

    # 检查每个block内部是否有3个及以上的小block
    for i, (xmin, ymin, xmax, ymax, area, score, block) in enumerate(block_info):
        # 如果当前block已标记为删除，则跳过
        if i in marked_indices:
            continue

        # 查找内部的小block (仅考虑尚未被标记为删除的block)
        blocks_inside = [(j, j_score, j_block) for j, (xj_min, yj_min, xj_max, yj_max, j_area, j_score, j_block) in
                         enumerate(block_info)
                         if i != j and j not in marked_indices and is_inside(block_info[j], block_info[i],
                                                                             overlap_threshold)]

        # 如果内部有3个及以上的小block
        if len(blocks_inside) >= 2:
            # 计算小block的平均分数
            avg_score = sum(s for _, s, _ in blocks_inside) / len(blocks_inside)

            # 比较大block的分数和小block的平均分数
            if score > avg_score:
                # 保留大block，扩展其边界
                # 首先将所有小block标记为要删除
                for j, _, j_block in blocks_inside:
                    if j_block not in blocks_to_remove:
                        blocks_to_remove.append(j_block)
                        marked_indices.add(j)  # 标记索引为已处理

                # 扩展大block的边界以包含所有小block
                new_xmin, new_ymin, new_xmax, new_ymax = xmin, ymin, xmax, ymax
                for _, _, j_block in blocks_inside:
                    j_xmin, j_ymin = int(j_block['poly'][0]), int(j_block['poly'][1])
                    j_xmax, j_ymax = int(j_block['poly'][4]), int(j_block['poly'][5])
                    new_xmin = min(new_xmin, j_xmin)
                    new_ymin = min(new_ymin, j_ymin)
                    new_xmax = max(new_xmax, j_xmax)
                    new_ymax = max(new_ymax, j_ymax)

                # 更新大block的边界
                block['poly'][0] = block['poly'][6] = new_xmin
                block['poly'][1] = block['poly'][3] = new_ymin
                block['poly'][2] = block['poly'][4] = new_xmax
                block['poly'][5] = block['poly'][7] = new_ymax
            else:
                # 保留小blocks，删除大block
                blocks_to_remove.append(block)
                marked_indices.add(i)  # 标记当前索引为已处理
    return blocks_to_remove


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
            text_res_list.append(res)

    # Process tables: merge high IoU tables first, then filter nested tables
    table_res_list, table_indices = merge_high_iou_tables(
        table_res_list, layout_res, table_indices, iou_threshold)

    filtered_table_res_list = filter_nested_tables(
        table_res_list, overlap_threshold, area_threshold)

    filtered_table_res_list, table_need_remove = remove_overlaps_min_blocks(filtered_table_res_list)

    for res in table_need_remove:
        if res in layout_res:
            layout_res.remove(res)

    # Remove filtered out tables from layout_res
    if len(filtered_table_res_list) < len(table_res_list):
        kept_tables = set(id(table) for table in filtered_table_res_list)
        tables_to_remove = [table for table in table_res_list if id(table) not in kept_tables]
        for table in tables_to_remove:
            if table in layout_res:
                layout_res.remove(table)

    # Remove overlaps in OCR and text regions
    text_res_list, need_remove = remove_overlaps_min_blocks(text_res_list)

    ocr_res_list.extend(text_res_list)

    for res in need_remove:
        if res in layout_res:
            layout_res.remove(res)

    # 检测大block内部是否包含多个小block, 合并ocr和table列表进行检测
    combined_res_list = ocr_res_list + filtered_table_res_list
    blocks_to_remove = remove_overlaps_low_confidence_blocks(combined_res_list, overlap_threshold)
    # 移除需要删除的blocks
    for block in blocks_to_remove:
        if block in ocr_res_list:
            ocr_res_list.remove(block)
        elif block in filtered_table_res_list:
            filtered_table_res_list.remove(block)
        # 同时从layout_res中删除
        if block in layout_res:
            layout_res.remove(block)

    return ocr_res_list, filtered_table_res_list, single_page_mfdetrec_res


def clean_memory(device='cuda'):
    if device == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    elif str(device).startswith("npu"):
        if torch_npu.npu.is_available():
            torch_npu.npu.empty_cache()
    elif str(device).startswith("mps"):
        torch.mps.empty_cache()
    gc.collect()


def clean_vram(device, vram_threshold=8):
    total_memory = get_vram(device)
    if total_memory and total_memory <= vram_threshold:
        gc_start = time.time()
        clean_memory(device)
        gc_time = round(time.time() - gc_start, 2)
        # logger.info(f"gc time: {gc_time}")


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

    return total_memory
