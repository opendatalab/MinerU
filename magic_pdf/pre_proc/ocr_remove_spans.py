from magic_pdf.libs.boxbase import _is_in_or_part_overlap


def remove_spans_by_bboxes(spans, need_remove_spans_bboxes):
    # 遍历spans, 判断是否在removed_span_block_bboxes中
    # 如果是, 则删除该span
    # 否则, 保留该span
    need_remove_spans = []
    for span in spans:
        for bbox in need_remove_spans_bboxes:
            if _is_in_or_part_overlap(span['bbox'], bbox):
                need_remove_spans.append(span)
                break

    for span in need_remove_spans:
        spans.remove(span)

    return spans
