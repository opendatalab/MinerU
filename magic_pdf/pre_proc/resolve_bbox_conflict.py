"""
从pdf里提取出来api给出的bbox,然后根据重叠情况做出取舍
1. 首先去掉出现在图片上的bbox，图片包括表格和图片
2. 然后去掉出现在文字blcok上的图片bbox
"""

from magic_pdf.libs.boxbase import _is_in, _is_in_or_part_overlap, _is_left_overlap
from magic_pdf.libs.drop_tag import ON_IMAGE_TEXT, ON_TABLE_TEXT


def resolve_bbox_overlap_conflict(images: list, tables: list, interline_equations: list, inline_equations: list,
                                  text_raw_blocks: list):
    """
    text_raw_blocks结构是从pymupdf里直接取到的结构，具体样例参考test/assets/papre/pymu_textblocks.json
    当下采用一种粗暴的方式：
    1. 去掉图片上的公式
    2. 去掉table上的公式
    2. 图片和文字block部分重叠，首先丢弃图片
    3. 图片和图片重叠，修改图片的bbox，使得图片不重叠(暂时没这么做，先把图片都扔掉)
    4. 去掉文字bbox里位于图片、表格上的文字（一定要完全在图、表内部）
    5. 去掉表格上的文字
    """
    text_block_removed = []
    images_backup = []

    # 去掉位于图片上的文字block
    for image_box in images:
        for text_block in text_raw_blocks:
            text_bbox = text_block["bbox"]
            if _is_in(text_bbox, image_box):
                text_block['tag'] = ON_IMAGE_TEXT
                text_block_removed.append(text_block)
    # 去掉table上的文字block
    for table_box in tables:
        for text_block in text_raw_blocks:
            text_bbox = text_block["bbox"]
            if _is_in(text_bbox, table_box):
                text_block['tag'] = ON_TABLE_TEXT
                text_block_removed.append(text_block)

    for text_block in text_block_removed:
        if text_block in text_raw_blocks:
            text_raw_blocks.remove(text_block)

    # 第一步去掉在图片上出现的公式box
    temp = []
    for image_box in images:
        for eq1 in interline_equations:
            if _is_in_or_part_overlap(image_box, eq1[:4]):
                temp.append(eq1)
        for eq2 in inline_equations:
            if _is_in_or_part_overlap(image_box, eq2[:4]):
                temp.append(eq2)

    for eq in temp:
        if eq in interline_equations:
            interline_equations.remove(eq)
        if eq in inline_equations:
            inline_equations.remove(eq)

    # 第二步去掉在表格上出现的公式box
    temp = []
    for table_box in tables:
        for eq1 in interline_equations:
            if _is_in_or_part_overlap(table_box, eq1[:4]):
                temp.append(eq1)
        for eq2 in inline_equations:
            if _is_in_or_part_overlap(table_box, eq2[:4]):
                temp.append(eq2)

    for eq in temp:
        if eq in interline_equations:
            interline_equations.remove(eq)
        if eq in inline_equations:
            inline_equations.remove(eq)

    # 图片和文字重叠，丢掉图片
    for image_box in images:
        for text_block in text_raw_blocks:
            text_bbox = text_block["bbox"]
            if _is_in_or_part_overlap(image_box, text_bbox):
                images_backup.append(image_box)
                break
    for image_box in images_backup:
        images.remove(image_box)

    # 图片和图片重叠，两张都暂时不参与版面计算
    images_dup_index = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            if _is_in_or_part_overlap(images[i], images[j]):
                images_dup_index.append(i)
                images_dup_index.append(j)

    dup_idx = set(images_dup_index)
    for img_id in dup_idx:
        images_backup.append(images[img_id])
        images[img_id] = None

    images = [img for img in images if img is not None]

    # 如果行间公式和文字block重叠，放到临时的数据里，防止这些文字box影响到layout计算。通过计算IOU合并行间公式和文字block
    # 对于这样的文本块删除，然后保留行间公式的大小不变。
    # 当计算完毕layout，这部分再合并回来
    text_block_removed_2 = []
    # for text_block in text_raw_blocks:
    #     text_bbox = text_block["bbox"]
    #     for eq in interline_equations:
    #         ratio = calculate_overlap_area_2_minbox_area_ratio(text_bbox, eq[:4])
    #         if ratio>0.05:
    #             text_block['tag'] = "belong-to-interline-equation"
    #             text_block_removed_2.append(text_block)
    #             break

    # for tb in text_block_removed_2:
    #     if tb in text_raw_blocks:
    #         text_raw_blocks.remove(tb)

    # text_block_removed = text_block_removed + text_block_removed_2

    return images, tables, interline_equations, inline_equations, text_raw_blocks, text_block_removed, images_backup, text_block_removed_2


def check_text_block_horizontal_overlap(text_blocks: list, header, footer) -> bool:
    """
    检查文本block之间的水平重叠情况，这种情况如果发生，那么这个pdf就不再继续处理了。
    因为这种情况大概率发生了公式没有被检测出来。
    
    """
    if len(text_blocks) == 0:
        return False

    page_min_y = 0
    page_max_y = max(yy['bbox'][3] for yy in text_blocks)

    def __max_y(lst: list):
        if len(lst) > 0:
            return max([item[1] for item in lst])
        return page_min_y

    def __min_y(lst: list):
        if len(lst) > 0:
            return min([item[3] for item in lst])
        return page_max_y

    clip_y0 = __max_y(header)
    clip_y1 = __min_y(footer)

    txt_bboxes = []
    for text_block in text_blocks:
        bbox = text_block["bbox"]
        if bbox[1] >= clip_y0 and bbox[3] <= clip_y1:
            txt_bboxes.append(bbox)

    for i in range(len(txt_bboxes)):
        for j in range(i + 1, len(txt_bboxes)):
            if _is_left_overlap(txt_bboxes[i], txt_bboxes[j]) or _is_left_overlap(txt_bboxes[j], txt_bboxes[i]):
                return True

    return False


def check_useful_block_horizontal_overlap(useful_blocks: list) -> bool:
    """
    检查文本block之间的水平重叠情况，这种情况如果发生，那么这个pdf就不再继续处理了。
    因为这种情况大概率发生了公式没有被检测出来。

    """
    if len(useful_blocks) == 0:
        return False

    page_min_y = 0
    page_max_y = max(yy['bbox'][3] for yy in useful_blocks)

    useful_bboxes = []
    for text_block in useful_blocks:
        bbox = text_block["bbox"]
        if bbox[1] >= page_min_y and bbox[3] <= page_max_y:
            useful_bboxes.append(bbox)

    for i in range(len(useful_bboxes)):
        for j in range(i + 1, len(useful_bboxes)):
            area_i = (useful_bboxes[i][2] - useful_bboxes[i][0]) * (useful_bboxes[i][3] - useful_bboxes[i][1])
            area_j = (useful_bboxes[j][2] - useful_bboxes[j][0]) * (useful_bboxes[j][3] - useful_bboxes[j][1])
            if _is_left_overlap(useful_bboxes[i], useful_bboxes[j]) or _is_left_overlap(useful_bboxes[j], useful_bboxes[i]):
                if area_i > area_j:
                    return True, useful_bboxes[j], useful_bboxes[i]
                else:
                    return True, useful_bboxes[i], useful_bboxes[j]

    return False, None, None
