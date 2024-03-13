from loguru import logger

from magic_pdf.libs.boxbase import __is_overlaps_y_exceeds_threshold, get_minbox_if_overlap_by_ratio, \
    calculate_overlap_area_in_bbox1_area_ratio


# 删除重叠spans中较小的那些
def remove_overlaps_min_spans(spans):
    for span1 in spans.copy():
        for span2 in spans.copy():
            if span1 != span2:
                overlap_box = get_minbox_if_overlap_by_ratio(span1['bbox'], span2['bbox'], 0.65)
                if overlap_box is not None:
                    bbox_to_remove = next((span for span in spans if span['bbox'] == overlap_box), None)
                    if bbox_to_remove is not None:
                        spans.remove(bbox_to_remove)
    return spans


# 将每一个line中的span从左到右排序
def line_sort_spans_by_left_to_right(lines):
    line_objects = []
    for line in lines:
        # 按照x0坐标排序
        line.sort(key=lambda span: span['bbox'][0])
        line_bbox = [
            min(span['bbox'][0] for span in line),  # x0
            min(span['bbox'][1] for span in line),  # y0
            max(span['bbox'][2] for span in line),  # x1
            max(span['bbox'][3] for span in line),  # y1
        ]
        line_objects.append({
            "bbox": line_bbox,
            "spans": line,
        })
    return line_objects

def merge_spans_to_line(spans):
    # 按照y0坐标排序
    spans.sort(key=lambda span: span['bbox'][1])

    lines = []
    current_line = [spans[0]]
    for span in spans[1:]:
        # 如果当前的span类型为"displayed_equation" 或者 当前行中已经有"displayed_equation"
        # image和table类型，同上
        if span['type'] in ["displayed_equation", "image", "table"] or any(
                s['type'] in ["displayed_equation", "image", "table"] for s in current_line):
            # 则开始新行
            lines.append(current_line)
            current_line = [span]
            continue

        # 如果当前的span与当前行的最后一个span在y轴上重叠，则添加到当前行
        if __is_overlaps_y_exceeds_threshold(span['bbox'], current_line[-1]['bbox']):
            current_line.append(span)
        else:
            # 否则，开始新行
            lines.append(current_line)
            current_line = [span]

    # 添加最后一行
    if current_line:
        lines.append(current_line)

    return lines

def merge_spans_to_line_by_layout(spans, layout_bboxes):
    lines = []
    new_spans = []
    for item in layout_bboxes:
        layout_bbox = item['layout_bbox']
        # 遍历spans,将每个span放入对应的layout中
        layout_sapns = []
        for span in spans:
            if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], layout_bbox) > 0.8:
                layout_sapns.append(span)
        # 如果layout_sapns不为空，则放入new_spans中
        if len(layout_sapns) > 0:
            new_spans.append(layout_sapns)
            # 从spans删除已经放入layout_sapns中的span
            for layout_sapn in layout_sapns:
                spans.remove(layout_sapn)

    if len(new_spans) > 0:
        for layout_sapns in new_spans:
            layout_lines = merge_spans_to_line(layout_sapns)
            lines.extend(layout_lines)

    #对line中的span进行排序
    lines = line_sort_spans_by_left_to_right(lines)

    return lines



def modify_y_axis(spans: list):
    inline_list = []
    displayed_list = []
    text_list = []
    image_list = []
    table_list = []

    spans.sort(key=lambda span: span['bbox'][1])

    lines = []
    current_line = [spans[0]]
    if spans[0]["type"] in ["displayed_equation", "image", "table"]:
        displayed_list.append(spans[0])

    line_first_y0 = spans[0]["bbox"][1]
    line_first_y = spans[0]["bbox"][3]
    #用于给行间公式搜索
    text_inline_lines = []
    for span in spans[1:]:
        # if span.get("content","") == "78.":
        #     print("debug")
        # 如果当前的span类型为"displayed_equation" 或者 当前行中已经有"displayed_equation"
        # image和table类型，同上
        if span['type'] in ["displayed_equation", "image", "table"] or any(
                s['type'] in ["displayed_equation", "image", "table"] for s in current_line):
            #传入
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
            if span["bbox"][1] < line_first_y0:
                line_first_y0 = span["bbox"][1]
            if span["bbox"][3] > line_first_y:
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


    #调整每一个文字行内bbox统一
    for line in text_inline_lines:
        current_line, (line_first_y0, line_first_y) = line
        for span in current_line:
            span["bbox"][1] = line_first_y0
            span["bbox"][3] = line_first_y
    #错误行间公式转行内公式
    j = 0
    for i in range(len(displayed_list)):
        # if i == 8:
        #     print("debug")
        span = displayed_list[i]
        span_y0, span_y = span["bbox"][1], span["bbox"][3]

        while j < len(text_inline_lines):
            text_line = text_inline_lines[j]
            y0, y1 = text_line[1]
            if (span_y0 < y0 and span_y > y0 or span_y0 < y1 and span_y > y1 or span_y0 < y0 and span_y > y1) and __is_overlaps_y_exceeds_threshold(span['bbox'], (0, y0, 0, y1)):
                span["bbox"][1] = y0
                # span["bbox"][3] = y1
                #调整公式类型
                if span["type"] == "displayed_equation":
                    span["type"] = "inline_equation"
                break
            elif span_y < y0 or span_y0 < y0 and span_y > y0 and not __is_overlaps_y_exceeds_threshold(span['bbox'], (0, y0, 0, y1)):
                break
            else:
                j += 1

    return spans




