from loguru import logger

from magic_pdf.libs.boxbase import calculate_overlap_area_in_bbox1_area_ratio, get_minbox_if_overlap_by_ratio, \
    __is_overlaps_y_exceeds_threshold


def remove_overlaps_min_spans(spans):
    #  删除重叠spans中较小的那些
    for span1 in spans.copy():
        for span2 in spans.copy():
            if span1 != span2:
                overlap_box = get_minbox_if_overlap_by_ratio(span1['bbox'], span2['bbox'], 0.65)
                if overlap_box is not None:
                    bbox_to_remove = next((span for span in spans if span['bbox'] == overlap_box), None)
                    if bbox_to_remove is not None:
                        spans.remove(bbox_to_remove)
    return spans


def remove_spans_by_bboxes(spans, need_remove_spans_bboxes):
    # 遍历spans, 判断是否在removed_span_block_bboxes中
    # 如果是, 则删除该span 否则, 保留该span
    need_remove_spans = []
    for span in spans:
        for removed_bbox in need_remove_spans_bboxes:
            if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], removed_bbox) > 0.5:
                need_remove_spans.append(span)
                break

    for span in need_remove_spans:
        spans.remove(span)

    return spans


def remove_spans_by_bboxes_dict(spans, need_remove_spans_bboxes_dict):
    dropped_text_block = []
    dropped_image_block = []
    dropped_table_block = []
    for drop_tag, removed_bboxes in need_remove_spans_bboxes_dict.items():
        # logger.info(f"remove spans by bbox dict, drop_tag: {drop_tag}, removed_bboxes: {removed_bboxes}")
        need_remove_spans = []
        for span in spans:
            for removed_bbox in removed_bboxes:
                if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], removed_bbox) > 0.5:
                    need_remove_spans.append(span)
                    break

        for span in need_remove_spans:
            spans.remove(span)
            span['tag'] = drop_tag
            if span['type'] in ['text', 'inline_equation', 'displayed_equation']:
                dropped_text_block.append(span)
            elif span['type'] == 'image':
                dropped_image_block.append(span)
            elif span['type'] == 'table':
                dropped_table_block.append(span)

    return spans, dropped_text_block, dropped_image_block, dropped_table_block


def adjust_bbox_for_standalone_block(spans):
    # 对tpye=["displayed_equation", "image", "table"]进行额外处理,如果左边有字的话,将该span的bbox中y0调整至不高于文字的y0
    for sb_span in spans:
        if sb_span['type'] in ["displayed_equation", "image", "table"]:
            for text_span in spans:
                if text_span['type'] in ['text', 'inline_equation']:
                    # 判断span2的纵向高度是否被span所覆盖
                    if sb_span['bbox'][1] < text_span['bbox'][1] and sb_span['bbox'][3] > text_span['bbox'][3]:
                        # 判断span2是否在span左边
                        if text_span['bbox'][0] < sb_span['bbox'][0]:
                            # 调整span的y0和span2的y0一致
                            sb_span['bbox'][1] = text_span['bbox'][1]
    return spans


def modify_y_axis(spans: list, displayed_list: list, text_inline_lines: list):
    # displayed_list = []

    spans.sort(key=lambda span: span['bbox'][1])

    lines = []
    current_line = [spans[0]]
    if spans[0]["type"] in ["displayed_equation", "image", "table"]:
        displayed_list.append(spans[0])

    line_first_y0 = spans[0]["bbox"][1]
    line_first_y = spans[0]["bbox"][3]
    # 用于给行间公式搜索
    # text_inline_lines = []
    for span in spans[1:]:
        # if span.get("content","") == "78.":
        #     print("debug")
        # 如果当前的span类型为"displayed_equation" 或者 当前行中已经有"displayed_equation"
        # image和table类型，同上
        if span['type'] in ["displayed_equation", "image", "table"] or any(
                s['type'] in ["displayed_equation", "image", "table"] for s in current_line):
            # 传入
            if span["type"] in ["displayed_equation", "image", "table"]:
                displayed_list.append(span)
            # 则开始新行
            lines.append(current_line)
            if len(current_line) > 1 or current_line[0]["type"] in ["text", "inline_equation"]:
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
        if len(current_line) > 1 or current_line[0]["type"] in ["text", "inline_equation"]:
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
                    span_y0 < y0 and span_y > y0 or span_y0 < y1 and span_y > y1 or span_y0 < y0 and span_y > y1) and __is_overlaps_y_exceeds_threshold(
                    span['bbox'], (0, y0, 0, y1)):

                # 调整公式类型
                if span["type"] == "displayed_equation":
                    # 最后一行是行间公式
                    if j + 1 >= len(text_inline_lines):
                        span["type"] = "inline_equation"
                        span["bbox"][1] = y0
                        span["bbox"][3] = y1
                    else:
                        # 行间公式旁边有多行文字或者行间公式比文字高3倍则不转换
                        y0_next, y1_next = text_inline_lines[j + 1][1]
                        if not __is_overlaps_y_exceeds_threshold(span['bbox'], (0, y0_next, 0, y1_next)) and 3 * (
                                y1 - y0) > span_y - span_y0:
                            span["type"] = "inline_equation"
                            span["bbox"][1] = y0
                            span["bbox"][3] = y1
                break
            elif span_y < y0 or span_y0 < y0 and span_y > y0 and not __is_overlaps_y_exceeds_threshold(span['bbox'],
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
                if span["type"] == "image":
                    images.append(span)
                elif span["type"] == "table":
                    tables.append(span)
                elif span["type"] == "inline_equation":
                    inline_equations.append(span)
                elif span["type"] == "displayed_equation":
                    interline_equations.append(span)
                else:
                    continue
    return images, tables, interline_equations, inline_equations
