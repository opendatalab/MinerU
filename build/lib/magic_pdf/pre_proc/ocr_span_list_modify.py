from loguru import logger

from magic_pdf.libs.boxbase import calculate_overlap_area_in_bbox1_area_ratio, get_minbox_if_overlap_by_ratio, \
    __is_overlaps_y_exceeds_threshold, calculate_iou
from magic_pdf.libs.drop_tag import DropTag
from magic_pdf.libs.ocr_content_type import ContentType, BlockType


def remove_overlaps_low_confidence_spans(spans):
    dropped_spans = []
    #  删除重叠spans中置信度低的的那些
    for span1 in spans:
        for span2 in spans:
            if span1 != span2:
                # span1 或 span2 任何一个都不应该在 dropped_spans 中
                if span1 in dropped_spans or span2 in dropped_spans:
                    continue
                else:
                    if calculate_iou(span1['bbox'], span2['bbox']) > 0.9:
                        if span1['score'] < span2['score']:
                            span_need_remove = span1
                        else:
                            span_need_remove = span2
                        if span_need_remove is not None and span_need_remove not in dropped_spans:
                            dropped_spans.append(span_need_remove)

    if len(dropped_spans) > 0:
        for span_need_remove in dropped_spans:
            spans.remove(span_need_remove)
            span_need_remove['tag'] = DropTag.SPAN_OVERLAP

    return spans, dropped_spans


def remove_overlaps_min_spans(spans):
    dropped_spans = []
    #  删除重叠spans中较小的那些
    for span1 in spans:
        for span2 in spans:
            if span1 != span2:
                overlap_box = get_minbox_if_overlap_by_ratio(span1['bbox'], span2['bbox'], 0.65)
                if overlap_box is not None:
                    span_need_remove = next((span for span in spans if span['bbox'] == overlap_box), None)
                    if span_need_remove is not None and span_need_remove not in dropped_spans:
                        dropped_spans.append(span_need_remove)

    if len(dropped_spans) > 0:
        for span_need_remove in dropped_spans:
            spans.remove(span_need_remove)
            span_need_remove['tag'] = DropTag.SPAN_OVERLAP

    return spans, dropped_spans


def remove_spans_by_bboxes(spans, need_remove_spans_bboxes):
    # 遍历spans, 判断是否在removed_span_block_bboxes中
    # 如果是, 则删除该span 否则, 保留该span
    need_remove_spans = []
    for span in spans:
        for removed_bbox in need_remove_spans_bboxes:
            if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], removed_bbox) > 0.5:
                if span not in need_remove_spans:
                    need_remove_spans.append(span)
                    break

    if len(need_remove_spans) > 0:
        for span in need_remove_spans:
            spans.remove(span)

    return spans


def remove_spans_by_bboxes_dict(spans, need_remove_spans_bboxes_dict):
    dropped_spans = []
    for drop_tag, removed_bboxes in need_remove_spans_bboxes_dict.items():
        # logger.info(f"remove spans by bbox dict, drop_tag: {drop_tag}, removed_bboxes: {removed_bboxes}")
        need_remove_spans = []
        for span in spans:
            # 通过判断span的bbox是否在removed_bboxes中, 判断是否需要删除该span
            for removed_bbox in removed_bboxes:
                if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], removed_bbox) > 0.5:
                    need_remove_spans.append(span)
                    break
                # 当drop_tag为DropTag.FOOTNOTE时, 判断span是否在removed_bboxes中任意一个的下方，如果是,则删除该span
                elif drop_tag == DropTag.FOOTNOTE and (span['bbox'][1] + span['bbox'][3]) / 2 > removed_bbox[3] and \
                        removed_bbox[0] < (span['bbox'][0] + span['bbox'][2]) / 2 < removed_bbox[2]:
                    need_remove_spans.append(span)
                    break

        for span in need_remove_spans:
            spans.remove(span)
            span['tag'] = drop_tag
            dropped_spans.append(span)

    return spans, dropped_spans


def adjust_bbox_for_standalone_block(spans):
    # 对tpye=["interline_equation", "image", "table"]进行额外处理,如果左边有字的话,将该span的bbox中y0调整至不高于文字的y0
    for sb_span in spans:
        if sb_span['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table]:
            for text_span in spans:
                if text_span['type'] in [ContentType.Text, ContentType.InlineEquation]:
                    # 判断span2的纵向高度是否被span所覆盖
                    if sb_span['bbox'][1] < text_span['bbox'][1] and sb_span['bbox'][3] > text_span['bbox'][3]:
                        # 判断span2是否在span左边
                        if text_span['bbox'][0] < sb_span['bbox'][0]:
                            # 调整span的y0和span2的y0一致
                            sb_span['bbox'][1] = text_span['bbox'][1]
    return spans


def modify_y_axis(spans: list, displayed_list: list, text_inline_lines: list):
    # displayed_list = []
    # 如果spans为空,则不处理
    if len(spans) == 0:
        pass
    else:
        spans.sort(key=lambda span: span['bbox'][1])

        lines = []
        current_line = [spans[0]]
        if spans[0]["type"] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table]:
            displayed_list.append(spans[0])

        line_first_y0 = spans[0]["bbox"][1]
        line_first_y = spans[0]["bbox"][3]
        # 用于给行间公式搜索
        # text_inline_lines = []
        for span in spans[1:]:
            # if span.get("content","") == "78.":
            #     print("debug")
            # 如果当前的span类型为"interline_equation" 或者 当前行中已经有"interline_equation"
            # image和table类型，同上
            if span['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table] or any(
                    s['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table] for s in
                    current_line):
                # 传入
                if span["type"] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table]:
                    displayed_list.append(span)
                # 则开始新行
                lines.append(current_line)
                if len(current_line) > 1 or current_line[0]["type"] in [ContentType.Text, ContentType.InlineEquation]:
                    text_inline_lines.append((current_line, (line_first_y0, line_first_y)))
                current_line = [span]
                line_first_y0 = span["bbox"][1]
                line_first_y = span["bbox"][3]
                continue

            # 如果当前的span与当前行的最后一个span在y轴上重叠，则添加到当前行
            if __is_overlaps_y_exceeds_threshold(span['bbox'], current_line[-1]['bbox']):
                if span["type"] == "text":
                    line_first_y0 = span["bbox"][1]
                    line_first_y = span["bbox"][3]
                current_line.append(span)

            else:
                # 否则，开始新行
                lines.append(current_line)
                text_inline_lines.append((current_line, (line_first_y0, line_first_y)))
                current_line = [span]
                line_first_y0 = span["bbox"][1]
                line_first_y = span["bbox"][3]

            # 添加最后一行
        if current_line:
            lines.append(current_line)
            if len(current_line) > 1 or current_line[0]["type"] in [ContentType.Text, ContentType.InlineEquation]:
                text_inline_lines.append((current_line, (line_first_y0, line_first_y)))
        for line in text_inline_lines:
            # 按照x0坐标排序
            current_line = line[0]
            current_line.sort(key=lambda span: span['bbox'][0])

        # 调整每一个文字行内bbox统一
        for line in text_inline_lines:
            current_line, (line_first_y0, line_first_y) = line
            for span in current_line:
                span["bbox"][1] = line_first_y0
                span["bbox"][3] = line_first_y

        # return spans, displayed_list, text_inline_lines


def modify_inline_equation(spans: list, displayed_list: list, text_inline_lines: list):
    # 错误行间公式转行内公式
    j = 0
    for i in range(len(displayed_list)):
        # if i == 8:
        #     print("debug")
        span = displayed_list[i]
        span_y0, span_y = span["bbox"][1], span["bbox"][3]

        while j < len(text_inline_lines):
            text_line = text_inline_lines[j]
            y0, y1 = text_line[1]
            if (
                    span_y0 < y0 < span_y or span_y0 < y1 < span_y or span_y0 < y0 and span_y > y1
            ) and __is_overlaps_y_exceeds_threshold(
                span['bbox'], (0, y0, 0, y1)
            ):
                # 调整公式类型
                if span["type"] == ContentType.InterlineEquation:
                    # 最后一行是行间公式
                    if j + 1 >= len(text_inline_lines):
                        span["type"] = ContentType.InlineEquation
                        span["bbox"][1] = y0
                        span["bbox"][3] = y1
                    else:
                        # 行间公式旁边有多行文字或者行间公式比文字高3倍则不转换
                        y0_next, y1_next = text_inline_lines[j + 1][1]
                        if not __is_overlaps_y_exceeds_threshold(span['bbox'], (0, y0_next, 0, y1_next)) and 3 * (
                                y1 - y0) > span_y - span_y0:
                            span["type"] = ContentType.InlineEquation
                            span["bbox"][1] = y0
                            span["bbox"][3] = y1
                break
            elif span_y < y0 or span_y0 < y0 < span_y and not __is_overlaps_y_exceeds_threshold(span['bbox'],
                                                                                                (0, y0, 0, y1)):
                break
            else:
                j += 1

    return spans


def get_qa_need_list(blocks):
    # 创建 images, tables, interline_equations, inline_equations 的副本
    images = []
    tables = []
    interline_equations = []
    inline_equations = []

    for block in blocks:
        for line in block["lines"]:
            for span in line["spans"]:
                if span["type"] == ContentType.Image:
                    images.append(span)
                elif span["type"] == ContentType.Table:
                    tables.append(span)
                elif span["type"] == ContentType.InlineEquation:
                    inline_equations.append(span)
                elif span["type"] == ContentType.InterlineEquation:
                    interline_equations.append(span)
                else:
                    continue
    return images, tables, interline_equations, inline_equations


def get_qa_need_list_v2(blocks):
    # 创建 images, tables, interline_equations, inline_equations 的副本
    images = []
    tables = []
    interline_equations = []

    for block in blocks:
        if block["type"] == BlockType.Image:
            images.append(block)
        elif block["type"] == BlockType.Table:
            tables.append(block)
        elif block["type"] == BlockType.InterlineEquation:
            interline_equations.append(block)
    return images, tables, interline_equations
