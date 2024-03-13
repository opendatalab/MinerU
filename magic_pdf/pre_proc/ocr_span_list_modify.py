from magic_pdf.libs.boxbase import calculate_overlap_area_in_bbox1_area_ratio, get_minbox_if_overlap_by_ratio


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
