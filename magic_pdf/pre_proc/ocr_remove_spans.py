from magic_pdf.libs.boxbase import calculate_overlap_area_in_bbox1_area_ratio


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
