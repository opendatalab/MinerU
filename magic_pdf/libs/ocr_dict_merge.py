from magic_pdf.libs.boxbase import __is_overlaps_y_exceeds_threshold


def merge_spans(spans):
    # 按照y0坐标排序
    spans.sort(key=lambda span: span['bbox'][1])

    lines = []
    current_line = [spans[0]]
    for span in spans[1:]:
        # 如果当前的span类型为"displayed_equation" 或者 当前行中已经有"displayed_equation"
        if span['type'] == "displayed_equation" or any(s['type'] == "displayed_equation" for s in current_line):
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

    # 计算每行的边界框，并对每行中的span按照x0进行排序
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
