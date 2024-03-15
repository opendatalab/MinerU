from loguru import logger

from magic_pdf.libs.boxbase import __is_overlaps_y_exceeds_threshold, get_minbox_if_overlap_by_ratio, \
    calculate_overlap_area_in_bbox1_area_ratio
from magic_pdf.libs.ocr_content_type import ContentType


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
    if len(spans) == 0:
        return []
    else:
        # 按照y0坐标排序
        spans.sort(key=lambda span: span['bbox'][1])

        lines = []
        current_line = [spans[0]]
        for span in spans[1:]:
            # 如果当前的span类型为"interline_equation" 或者 当前行中已经有"interline_equation"
            # image和table类型，同上
            if span['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table] or any(
                    s['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table] for s in current_line):
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
            if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], layout_bbox) > 0.65:
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


def merge_lines_to_block(lines):
    # 目前不做block拼接,先做个结构,每个block中只有一个line,block的bbox就是line的bbox
    blocks = []
    for line in lines:
        blocks.append(
            {
                "bbox": line["bbox"],
                "lines": [line],
            }
        )
    return blocks






