# Copyright (c) Opendatalab. All rights reserved.
import copy

from mineru.utils.enum_class import BlockType, SplitFlag


LINE_STOP_FLAG = ('.', '!', '?', '。', '！', '？', ')', '）', '"', '”', ':', '：', ';', '；')
MERGE_BARRIER_TYPES = {
    BlockType.TITLE,
    BlockType.INTERLINE_EQUATION,
    BlockType.LIST,
}
_CROSS_PAGE_MERGE_KEY = "_cross_page_merge_prev"
_EDGE_TEXT_LINE_HINTS_KEY = "_edge_text_line_hints"


def iter_block_spans(block):
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            yield span

    for sub_block in block.get("blocks", []):
        yield from iter_block_spans(sub_block)


def build_para_blocks_from_preproc(pdf_info_list):
    for page_info in pdf_info_list:
        page_info["para_blocks"] = copy.deepcopy(page_info.get("preproc_blocks", []))


def merge_para_text_blocks(pdf_info_list, allow_cross_page=False):
    ordered_blocks = []
    for page_info in pdf_info_list:
        page_idx = page_info.get("page_idx")
        for order_idx, block in enumerate(page_info.get("para_blocks", [])):
            ordered_blocks.append((page_idx, order_idx, block))

    for current_index in range(len(ordered_blocks) - 1, -1, -1):
        current_page_idx, _, current_block = ordered_blocks[current_index]
        if current_block.get("type") != BlockType.TEXT:
            continue
        if not current_block.get("merge_prev"):
            continue
        if not _block_has_lines(current_block):
            continue

        previous_block = _find_previous_text_block(
            ordered_blocks,
            current_index,
            current_block,
            current_page_idx,
            allow_cross_page=allow_cross_page,
        )
        if previous_block is None:
            continue

        previous_page_idx, _, previous_text_block = previous_block
        _merge_text_block(
            current_block,
            previous_text_block,
            is_cross_page=current_page_idx != previous_page_idx,
        )


def annotate_hybrid_cross_page_merge_prev(pdf_info_list, prefer_edge_line_hints=False):
    for page_index in range(1, len(pdf_info_list)):
        previous_page_info = pdf_info_list[page_index - 1]
        current_page_info = pdf_info_list[page_index]

        previous_text_block = _find_last_page_edge_text_block(previous_page_info)
        current_text_block = _find_first_page_edge_text_block(current_page_info)
        if previous_text_block is None or current_text_block is None:
            continue

        previous_metric_lines = _resolve_metric_lines(
            previous_page_info,
            previous_text_block,
            edge_name="last",
            prefer_edge_line_hints=prefer_edge_line_hints,
        )
        current_metric_lines = _resolve_metric_lines(
            current_page_info,
            current_text_block,
            edge_name="first",
            prefer_edge_line_hints=prefer_edge_line_hints,
        )
        if not previous_metric_lines or not current_metric_lines:
            continue

        if can_merge_text_blocks(
            current_text_block,
            previous_text_block,
            current_metric_lines=current_metric_lines,
            previous_metric_lines=previous_metric_lines,
        ):
            current_text_block["merge_prev"] = True
            current_text_block[_CROSS_PAGE_MERGE_KEY] = True


def can_merge_text_blocks(current_block, previous_block, current_metric_lines=None, previous_metric_lines=None):
    current_lines = current_block.get("lines", [])
    previous_lines = previous_block.get("lines", [])
    if not current_lines or not previous_lines:
        return False

    current_metric_lines = current_metric_lines or current_lines
    previous_metric_lines = previous_metric_lines or previous_lines
    if not current_metric_lines or not previous_metric_lines:
        return False

    first_metric_line = current_metric_lines[0]
    first_line_height = _line_height(first_metric_line)
    if first_line_height <= 0:
        return False

    current_bbox_fs = _build_bbox_fs(current_block, current_metric_lines)
    if abs(current_bbox_fs[0] - first_metric_line["bbox"][0]) >= first_line_height / 2:
        return False

    last_metric_line = previous_metric_lines[-1]
    last_line_height = _line_height(last_metric_line)
    if last_line_height <= 0:
        return False

    previous_bbox_fs = _build_bbox_fs(previous_block, previous_metric_lines)

    first_span = _first_span(current_lines[0])
    last_span = _last_span(previous_lines[-1])
    if first_span is None or last_span is None:
        return False

    first_content = first_span.get("content", "")
    last_content = last_span.get("content", "")
    if not first_content:
        return False

    current_block_width = current_block["bbox"][2] - current_block["bbox"][0]
    previous_block_width = previous_block["bbox"][2] - previous_block["bbox"][0]
    min_block_width = min(current_block_width, previous_block_width)
    if min_block_width <= 0:
        return False

    if abs(previous_bbox_fs[2] - last_metric_line["bbox"][2]) >= last_line_height:
        return False
    if last_content.endswith(LINE_STOP_FLAG):
        return False
    if abs(current_block_width - previous_block_width) >= min_block_width:
        return False
    if first_content[0].isdigit() or first_content[0].isupper():
        return False
    if current_block["bbox"][1] >= previous_block["bbox"][3]:
        return False
    if len(current_metric_lines) <= 1 and len(previous_metric_lines) <= 1:
        return False

    return True


def cleanup_internal_para_block_metadata(pdf_info_list):
    for page_info in pdf_info_list:
        page_info.pop(_EDGE_TEXT_LINE_HINTS_KEY, None)
        for block in page_info.get("para_blocks", []):
            block.pop(_CROSS_PAGE_MERGE_KEY, None)


def edge_text_line_hints_key():
    return _EDGE_TEXT_LINE_HINTS_KEY


def _find_previous_text_block(ordered_blocks, current_index, current_block, current_page_idx, allow_cross_page):
    cross_page_allowed = allow_cross_page and current_block.get(_CROSS_PAGE_MERGE_KEY, False)

    for previous_index in range(current_index - 1, -1, -1):
        previous_page_idx, _, previous_block = ordered_blocks[previous_index]
        if previous_page_idx != current_page_idx and not cross_page_allowed:
            return None

        previous_type = previous_block.get("type")
        if previous_type in MERGE_BARRIER_TYPES:
            return None
        if previous_type != BlockType.TEXT:
            continue
        return ordered_blocks[previous_index]

    return None


def _find_first_page_edge_text_block(page_info):
    for block in page_info.get("para_blocks", []):
        block_type = block.get("type")
        if block_type in MERGE_BARRIER_TYPES:
            return None
        if block_type == BlockType.TEXT and _block_has_lines(block):
            return block
    return None


def _find_last_page_edge_text_block(page_info):
    for block in reversed(page_info.get("para_blocks", [])):
        block_type = block.get("type")
        if block_type in MERGE_BARRIER_TYPES:
            return None
        if block_type == BlockType.TEXT and _block_has_lines(block):
            return block
    return None


def _resolve_metric_lines(page_info, block, edge_name, prefer_edge_line_hints):
    if prefer_edge_line_hints:
        edge_line_hints = page_info.get(_EDGE_TEXT_LINE_HINTS_KEY, {})
        edge_hint = edge_line_hints.get(edge_name)
        if edge_hint and edge_hint.get("index") == block.get("index"):
            return edge_hint.get("lines", [])
        return []
    return block.get("lines", [])


def _merge_text_block(current_block, previous_block, is_cross_page):
    if is_cross_page:
        for line in current_block.get("lines", []):
            for span in line.get("spans", []):
                span[SplitFlag.CROSS_PAGE] = True

    previous_block.setdefault("lines", []).extend(current_block.get("lines", []))
    current_block["lines"] = []
    current_block[SplitFlag.LINES_DELETED] = True


def _line_height(line):
    bbox = line.get("bbox")
    if not bbox:
        return 0
    return bbox[3] - bbox[1]


def _build_bbox_fs(block, lines):
    if lines:
        return [
            min(line["bbox"][0] for line in lines),
            min(line["bbox"][1] for line in lines),
            max(line["bbox"][2] for line in lines),
            max(line["bbox"][3] for line in lines),
        ]
    return list(block.get("bbox", []))


def _block_has_lines(block):
    return any(line.get("spans") for line in block.get("lines", []))


def _first_span(line):
    spans = line.get("spans", [])
    return spans[0] if spans else None


def _last_span(line):
    spans = line.get("spans", [])
    return spans[-1] if spans else None
