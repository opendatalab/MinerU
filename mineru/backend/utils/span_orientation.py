# Copyright (c) Opendatalab. All rights reserved.
from ...types import Span

VERTICAL_SPAN_HEIGHT_TO_WIDTH_RATIO_THRESHOLD = 2
VERTICAL_SPAN_IN_BLOCK_THRESHOLD = 0.8


def is_vertical_text_block_by_spans(spans: list[Span]) -> bool:
    """根据块内文本 span 的高宽比判断文本块是否更像竖排文本。"""
    valid_span_count = 0
    vertical_span_count = 0
    for span in spans:
        bbox = span.bbox
        if not bbox or len(bbox) < 4:
            continue

        span_width = bbox[2] - bbox[0]
        span_height = bbox[3] - bbox[1]
        if span_width <= 0 or span_height <= 0:
            continue

        valid_span_count += 1
        if span_height / span_width > VERTICAL_SPAN_HEIGHT_TO_WIDTH_RATIO_THRESHOLD:
            vertical_span_count += 1

    if valid_span_count == 0:
        return False

    return vertical_span_count / valid_span_count > VERTICAL_SPAN_IN_BLOCK_THRESHOLD
