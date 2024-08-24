"""
对pymupdf返回的结构里的公式进行替换，替换为模型识别的公式结果
"""

from magic_pdf.libs.commons import fitz
import json
import os
from pathlib import Path
from loguru import logger
from magic_pdf.libs.ocr_content_type import ContentType

TYPE_INLINE_EQUATION = ContentType.InlineEquation
TYPE_INTERLINE_EQUATION = ContentType.InterlineEquation


def combine_chars_to_pymudict(block_dict, char_dict):
    """
    把block级别的pymupdf 结构里加入char结构
    """
    # 因为block_dict 被裁剪过，因此先把他和char_dict文字块对齐，才能进行补充
    char_map = {tuple(item["bbox"]): item for item in char_dict}

    for i in range(len(block_dict)):  # blcok
        block = block_dict[i]
        key = block["bbox"]
        char_dict_item = char_map[tuple(key)]
        char_dict_map = {tuple(item["bbox"]): item for item in char_dict_item["lines"]}
        for j in range(len(block["lines"])):
            lines = block["lines"][j]
            with_char_lines = char_dict_map[lines["bbox"]]
            for k in range(len(lines["spans"])):
                spans = lines["spans"][k]
                try:
                    chars = with_char_lines["spans"][k]["chars"]
                except Exception as e:
                    logger.error(char_dict[i]["lines"][j])

                spans["chars"] = chars

    return block_dict


def calculate_overlap_area_2_minbox_area_ratio(bbox1, min_bbox):
    """
    计算box1和box2的重叠面积占最小面积的box的比例
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], min_bbox[0])
    y_top = max(bbox1[1], min_bbox[1])
    x_right = min(bbox1[2], min_bbox[2])
    y_bottom = min(bbox1[3], min_bbox[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    min_box_area = (min_bbox[3] - min_bbox[1]) * (min_bbox[2] - min_bbox[0])
    if min_box_area == 0:
        return 0
    else:
        return intersection_area / min_box_area


def _is_xin(bbox1, bbox2):
    area1 = abs(bbox1[2] - bbox1[0]) * abs(bbox1[3] - bbox1[1])
    area2 = abs(bbox2[2] - bbox2[0]) * abs(bbox2[3] - bbox2[1])
    if area1 < area2:
        ratio = calculate_overlap_area_2_minbox_area_ratio(bbox2, bbox1)
    else:
        ratio = calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2)

    return ratio > 0.6


def remove_text_block_in_interline_equation_bbox(interline_bboxes, text_blocks):
    """消除掉整个块都在行间公式块内部的文本块"""
    for eq_bbox in interline_bboxes:
        removed_txt_blk = []
        for text_blk in text_blocks:
            text_bbox = text_blk["bbox"]
            if (
                calculate_overlap_area_2_minbox_area_ratio(eq_bbox["bbox"], text_bbox)
                >= 0.7
            ):
                removed_txt_blk.append(text_blk)
        for blk in removed_txt_blk:
            text_blocks.remove(blk)

    return text_blocks


def _is_in_or_part_overlap(box1, box2) -> bool:
    """
    两个bbox是否有部分重叠或者包含
    """
    if box1 is None or box2 is None:
        return False

    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    return not (
        x1_1 < x0_2  # box1在box2的左边
        or x0_1 > x1_2  # box1在box2的右边
        or y1_1 < y0_2  # box1在box2的上边
        or y0_1 > y1_2
    )  # box1在box2的下边


def remove_text_block_overlap_interline_equation_bbox(
    interline_eq_bboxes, pymu_block_list
):

    """消除掉行行内公式有部分重叠的文本块的内容。
    同时重新计算消除重叠之后文本块的大小"""
    deleted_block = []
    for text_block in pymu_block_list:
        deleted_line = []
        for line in text_block["lines"]:
            deleted_span = []
            for span in line["spans"]:
                deleted_chars = []
                for char in span["chars"]:
                    if any(
                            [
                                (calculate_overlap_area_2_minbox_area_ratio(eq_bbox["bbox"], char["bbox"]) > 0.5)
                                for eq_bbox in interline_eq_bboxes
                            ]
                    ):
                        deleted_chars.append(char)
                # 检查span里没有char则删除这个span
                for char in deleted_chars:
                    span["chars"].remove(char)
                # 重新计算这个span的大小
                if len(span["chars"]) == 0:  # 删除这个span
                    deleted_span.append(span)
                else:
                    span["bbox"] = (
                        min([b["bbox"][0] for b in span["chars"]]),
                        min([b["bbox"][1] for b in span["chars"]]),
                        max([b["bbox"][2] for b in span["chars"]]),
                        max([b["bbox"][3] for b in span["chars"]]),
                    )

            # 检查这个span
            for span in deleted_span:
                line["spans"].remove(span)
            if len(line["spans"]) == 0:  # 删除这个line
                deleted_line.append(line)
            else:
                line["bbox"] = (
                    min([b["bbox"][0] for b in line["spans"]]),
                    min([b["bbox"][1] for b in line["spans"]]),
                    max([b["bbox"][2] for b in line["spans"]]),
                    max([b["bbox"][3] for b in line["spans"]]),
                )

        # 检查这个block是否可以删除
        for line in deleted_line:
            text_block["lines"].remove(line)
        if len(text_block["lines"]) == 0:  # 删除block
            deleted_block.append(text_block)
        else:
            text_block["bbox"] = (
                min([b["bbox"][0] for b in text_block["lines"]]),
                min([b["bbox"][1] for b in text_block["lines"]]),
                max([b["bbox"][2] for b in text_block["lines"]]),
                max([b["bbox"][3] for b in text_block["lines"]]),
            )

    # 检查text block删除
    for block in deleted_block:
        pymu_block_list.remove(block)
    if len(pymu_block_list) == 0:
        return []

    return pymu_block_list


def insert_interline_equations_textblock(interline_eq_bboxes, pymu_block_list):
    """在行间公式对应的地方插上一个伪造的block"""
    for eq in interline_eq_bboxes:
        bbox = eq["bbox"]
        latex_content = eq["latex"]
        text_block = {
            "number": len(pymu_block_list),
            "type": 0,
            "bbox": bbox,
            "lines": [
                {
                    "spans": [
                        {
                            "size": 9.962599754333496,
                            "type": TYPE_INTERLINE_EQUATION,
                            "flags": 4,
                            "font": TYPE_INTERLINE_EQUATION,
                            "color": 0,
                            "ascender": 0.9409999847412109,
                            "descender": -0.3050000071525574,
                            "latex": latex_content,
                            "origin": [bbox[0], bbox[1]],
                            "bbox": bbox,
                        }
                    ],
                    "wmode": 0,
                    "dir": [1.0, 0.0],
                    "bbox": bbox,
                }
            ],
        }
        pymu_block_list.append(text_block)


def x_overlap_ratio(box1, box2):
    a, _, c, _ = box1
    e, _, g, _ = box2

    # 计算重叠宽度
    overlap_x = max(min(c, g) - max(a, e), 0)

    # 计算box1的宽度
    width1 = g - e

    # 计算重叠比例
    overlap_ratio = overlap_x / width1 if width1 != 0 else 0

    return overlap_ratio


def __is_x_dir_overlap(bbox1, bbox2):
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2])


def __y_overlap_ratio(box1, box2):
    """"""
    _, b, _, d = box1
    _, f, _, h = box2

    # 计算重叠高度
    overlap_y = max(min(d, h) - max(b, f), 0)

    # 计算box1的高度
    height1 = d - b

    # 计算重叠比例
    overlap_ratio = overlap_y / height1 if height1 != 0 else 0

    return overlap_ratio


def replace_line_v2(eqinfo, line):
    """
    扫描这一行所有的和公式框X方向重叠的char,然后计算char的左、右x0, x1,位于这个区间内的span删除掉。
    最后与这个x0,x1有相交的span0, span1内部进行分割。
    """
    first_overlap_span = -1
    first_overlap_span_idx = -1
    last_overlap_span = -1
    delete_chars = []
    for i in range(0, len(line["spans"])):
        if "chars" not in line["spans"][i]:
            continue

        if line["spans"][i].get("_type", None) is not None:
            continue  # 忽略，因为已经是插入的伪造span公式了

        for char in line["spans"][i]["chars"]:
            if __is_x_dir_overlap(eqinfo["bbox"], char["bbox"]):
                line_txt = ""
                for span in line["spans"]:
                    span_txt = "<span>"
                    for ch in span["chars"]:
                        span_txt = span_txt + ch["c"]

                    span_txt = span_txt + "</span>"

                    line_txt = line_txt + span_txt

                if first_overlap_span_idx == -1:
                    first_overlap_span = line["spans"][i]
                    first_overlap_span_idx = i
                last_overlap_span = line["spans"][i]
                delete_chars.append(char)

    # 第一个和最后一个char要进行检查，到底属于公式多还是属于正常span多
    if len(delete_chars) > 0:
        ch0_bbox = delete_chars[0]["bbox"]
        if x_overlap_ratio(eqinfo["bbox"], ch0_bbox) < 0.51:
            delete_chars.remove(delete_chars[0])
    if len(delete_chars) > 0:
        ch0_bbox = delete_chars[-1]["bbox"]
        if x_overlap_ratio(eqinfo["bbox"], ch0_bbox) < 0.51:
            delete_chars.remove(delete_chars[-1])

    # 计算x方向上被删除区间内的char的真实x0, x1
    if len(delete_chars):
        x0, x1 = min([b["bbox"][0] for b in delete_chars]), max(
            [b["bbox"][2] for b in delete_chars]
        )
    else:
        # logger.debug(f"行内公式替换没有发生，尝试下一行匹配, eqinfo={eqinfo}")
        return False

    # 删除位于x0, x1这两个中间的span
    delete_span = []
    for span in line["spans"]:
        span_box = span["bbox"]
        if x0 <= span_box[0] and span_box[2] <= x1:
            delete_span.append(span)
    for span in delete_span:
        line["spans"].remove(span)

    equation_span = {
        "size": 9.962599754333496,
        "type": TYPE_INLINE_EQUATION,
        "flags": 4,
        "font": TYPE_INLINE_EQUATION,
        "color": 0,
        "ascender": 0.9409999847412109,
        "descender": -0.3050000071525574,
        "latex": "",
        "origin": [337.1410153102337, 216.0205245153934],
        "bbox": eqinfo["bbox"]
    }
    # equation_span = line['spans'][0].copy()
    equation_span["latex"] = eqinfo['latex']
    equation_span["bbox"] = [x0, equation_span["bbox"][1], x1, equation_span["bbox"][3]]
    equation_span["origin"] = [equation_span["bbox"][0], equation_span["bbox"][1]]
    equation_span["chars"] = delete_chars
    equation_span["type"] = TYPE_INLINE_EQUATION
    equation_span["_eq_bbox"] = eqinfo["bbox"]
    line["spans"].insert(first_overlap_span_idx + 1, equation_span)  # 放入公式

    # logger.info(f"==>text is 【{line_txt}】, equation is 【{eqinfo['latex_text']}】")

    # 第一个、和最后一个有overlap的span进行分割,然后插入对应的位置
    first_span_chars = [
        char
        for char in first_overlap_span["chars"]
        if (char["bbox"][2] + char["bbox"][0]) / 2 < x0
    ]
    tail_span_chars = [
        char
        for char in last_overlap_span["chars"]
        if (char["bbox"][0] + char["bbox"][2]) / 2 > x1
    ]

    if len(first_span_chars) > 0:
        first_overlap_span["chars"] = first_span_chars
        first_overlap_span["text"] = "".join([char["c"] for char in first_span_chars])
        first_overlap_span["bbox"] = (
            first_overlap_span["bbox"][0],
            first_overlap_span["bbox"][1],
            max([chr["bbox"][2] for chr in first_span_chars]),
            first_overlap_span["bbox"][3],
        )
        # first_overlap_span['_type'] = "first"
    else:
        # 删掉
        if first_overlap_span not in delete_span:
            line["spans"].remove(first_overlap_span)

    if len(tail_span_chars) > 0:
        min_of_tail_span_x0 = min([chr["bbox"][0] for chr in tail_span_chars])
        min_of_tail_span_y0 = min([chr["bbox"][1] for chr in tail_span_chars])
        max_of_tail_span_x1 = max([chr["bbox"][2] for chr in tail_span_chars])
        max_of_tail_span_y1 = max([chr["bbox"][3] for chr in tail_span_chars])

        if last_overlap_span == first_overlap_span:  # 这个时候应该插入一个新的
            tail_span_txt = "".join([char["c"] for char in tail_span_chars])
            last_span_to_insert = last_overlap_span.copy()
            last_span_to_insert["chars"] = tail_span_chars
            last_span_to_insert["text"] = "".join(
                [char["c"] for char in tail_span_chars]
            )
            if equation_span["bbox"][2] >= last_overlap_span["bbox"][2]:
                last_span_to_insert["bbox"] = (
                    min_of_tail_span_x0,
                    min_of_tail_span_y0,
                    max_of_tail_span_x1,
                    max_of_tail_span_y1
                )
            else:
                last_span_to_insert["bbox"] = (
                    min([chr["bbox"][0] for chr in tail_span_chars]),
                    last_overlap_span["bbox"][1],
                    last_overlap_span["bbox"][2],
                    last_overlap_span["bbox"][3],
                )
            # 插入到公式对象之后
            equation_idx = line["spans"].index(equation_span)
            line["spans"].insert(equation_idx + 1, last_span_to_insert)  # 放入公式
        else:  # 直接修改原来的span
            last_overlap_span["chars"] = tail_span_chars
            last_overlap_span["text"] = "".join([char["c"] for char in tail_span_chars])
            last_overlap_span["bbox"] = (
                min([chr["bbox"][0] for chr in tail_span_chars]),
                last_overlap_span["bbox"][1],
                last_overlap_span["bbox"][2],
                last_overlap_span["bbox"][3],
            )
    else:
        # 删掉
        if (
            last_overlap_span not in delete_span
            and last_overlap_span != first_overlap_span
        ):
            line["spans"].remove(last_overlap_span)

    remain_txt = ""
    for span in line["spans"]:
        span_txt = "<span>"
        for char in span["chars"]:
            span_txt = span_txt + char["c"]

        span_txt = span_txt + "</span>"

        remain_txt = remain_txt + span_txt

    # logger.info(f"<== succ replace, text is 【{remain_txt}】, equation is 【{eqinfo['latex_text']}】")

    return True


def replace_eq_blk(eqinfo, text_block):
    """替换行内公式"""
    for line in text_block["lines"]:
        line_bbox = line["bbox"]
        if (
            _is_xin(eqinfo["bbox"], line_bbox)
            or __y_overlap_ratio(eqinfo["bbox"], line_bbox) > 0.6
        ):  # 定位到行, 使用y方向重合率是因为有的时候，一个行的宽度会小于公式位置宽度：行很高，公式很窄，
            replace_succ = replace_line_v2(eqinfo, line)
            if (
                not replace_succ
            ):  # 有的时候，一个pdf的line高度从API里会计算的有问题，因此在行内span级别会替换不成功，这就需要继续重试下一行
                continue
            else:
                break
    else:
        return False
    return True


def replace_inline_equations(inline_equation_bboxes, raw_text_blocks):
    """替换行内公式"""
    for eqinfo in inline_equation_bboxes:
        eqbox = eqinfo["bbox"]
        for blk in raw_text_blocks:
            if _is_xin(eqbox, blk["bbox"]):
                if not replace_eq_blk(eqinfo, blk):
                    logger.warning(f"行内公式没有替换成功：{eqinfo} ")
                else:
                    break

    return raw_text_blocks


def remove_chars_in_text_blocks(text_blocks):
    """删除text_blocks里的char"""
    for blk in text_blocks:
        for line in blk["lines"]:
            for span in line["spans"]:
                _ = span.pop("chars", "no such key")
    return text_blocks


def replace_equations_in_textblock(
    raw_text_blocks, inline_equation_bboxes, interline_equation_bboxes
):
    """
    替换行间和和行内公式为latex
    """
    raw_text_blocks = remove_text_block_in_interline_equation_bbox(
        interline_equation_bboxes, raw_text_blocks
    )  # 消除重叠：第一步，在公式内部的

    raw_text_blocks = remove_text_block_overlap_interline_equation_bbox(
        interline_equation_bboxes, raw_text_blocks
    )  # 消重，第二步，和公式覆盖的

    insert_interline_equations_textblock(interline_equation_bboxes, raw_text_blocks)
    raw_text_blocks = replace_inline_equations(inline_equation_bboxes, raw_text_blocks)
    return raw_text_blocks


def draw_block_on_pdf_with_txt_replace_eq_bbox(json_path, pdf_path):
    """ """
    new_pdf = f"{Path(pdf_path).parent}/{Path(pdf_path).stem}.step3-消除行内公式text_block.pdf"
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.loads(f.read())

    if os.path.exists(new_pdf):
        os.remove(new_pdf)
    new_doc = fitz.open("")

    doc = fitz.open(pdf_path)
    new_doc = fitz.open(pdf_path)
    for i in range(len(new_doc)):
        page = new_doc[i]
        inline_equation_bboxes = obj[f"page_{i}"]["inline_equations"]
        interline_equation_bboxes = obj[f"page_{i}"]["interline_equations"]
        raw_text_blocks = obj[f"page_{i}"]["preproc_blocks"]
        raw_text_blocks = remove_text_block_in_interline_equation_bbox(
            interline_equation_bboxes, raw_text_blocks
        )  # 消除重叠：第一步，在公式内部的
        raw_text_blocks = remove_text_block_overlap_interline_equation_bbox(
            interline_equation_bboxes, raw_text_blocks
        )  # 消重，第二步，和公式覆盖的
        insert_interline_equations_textblock(interline_equation_bboxes, raw_text_blocks)
        raw_text_blocks = replace_inline_equations(
            inline_equation_bboxes, raw_text_blocks
        )

        # 为了检验公式是否重复，把每一行里，含有公式的span背景改成黄色的
        color_map = [fitz.pdfcolor["blue"], fitz.pdfcolor["green"]]
        j = 0
        for blk in raw_text_blocks:
            for i, line in enumerate(blk["lines"]):

                # line_box = line['bbox']
                # shape = page.new_shape()
                # shape.draw_rect(line_box)
                # shape.finish(color=fitz.pdfcolor['red'], fill=color_map[j%2], fill_opacity=0.3)
                # shape.commit()
                # j = j+1

                for i, span in enumerate(line["spans"]):
                    shape_page = page.new_shape()
                    span_type = span.get("_type")
                    color = fitz.pdfcolor["blue"]
                    if span_type == "first":
                        color = fitz.pdfcolor["blue"]
                    elif span_type == "tail":
                        color = fitz.pdfcolor["green"]
                    elif span_type == TYPE_INLINE_EQUATION:
                        color = fitz.pdfcolor["black"]
                    else:
                        color = None

                    b = span["bbox"]
                    shape_page.draw_rect(b)

                    shape_page.finish(color=None, fill=color, fill_opacity=0.3)
                    shape_page.commit()

    new_doc.save(new_pdf)
    logger.info(f"save ok {new_pdf}")
    final_json = json.dumps(obj, ensure_ascii=False, indent=2)
    with open("equations_test/final_json.json", "w") as f:
        f.write(final_json)

    return new_pdf


if __name__ == "__main__":
    # draw_block_on_pdf_with_txt_replace_eq_bbox(new_json_path, equation_color_pdf)
    pass
