
from magic_pdf.config.drop_tag import DropTag
from magic_pdf.config.ocr_content_type import BlockType
from magic_pdf.libs.boxbase import calculate_iou, get_minbox_if_overlap_by_ratio


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
                        if (
                            span_need_remove is not None
                            and span_need_remove not in dropped_spans
                        ):
                            dropped_spans.append(span_need_remove)

    if len(dropped_spans) > 0:
        for span_need_remove in dropped_spans:
            spans.remove(span_need_remove)
            span_need_remove['tag'] = DropTag.SPAN_OVERLAP

    return spans, dropped_spans


def check_chars_is_overlap_in_span(chars):
    for i in range(len(chars)):
        for j in range(i + 1, len(chars)):
            if calculate_iou(chars[i]['bbox'], chars[j]['bbox']) > 0.9:
                return True
    return False


def remove_overlaps_min_spans(spans):
    dropped_spans = []
    #  删除重叠spans中较小的那些
    for span1 in spans:
        for span2 in spans:
            if span1 != span2:
                # span1 或 span2 任何一个都不应该在 dropped_spans 中
                if span1 in dropped_spans or span2 in dropped_spans:
                    continue
                else:
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


def get_qa_need_list_v2(blocks):
    # 创建 images, tables, interline_equations, inline_equations 的副本
    images = []
    tables = []
    interline_equations = []

    for block in blocks:
        if block['type'] == BlockType.Image:
            images.append(block)
        elif block['type'] == BlockType.Table:
            tables.append(block)
        elif block['type'] == BlockType.InterlineEquation:
            interline_equations.append(block)
    return images, tables, interline_equations
