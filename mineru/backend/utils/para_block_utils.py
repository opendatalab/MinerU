# Copyright (c) Opendatalab. All rights reserved.
import copy

from mineru.utils.enum_class import BlockType, SplitFlag


LINE_STOP_FLAG = ('.', '!', '?', '。', '！', '？', ')', '）', '"', '”', ':', '：', ';', '；')
SECTION_MERGE_BARRIER_TYPES = {
    BlockType.TITLE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
    BlockType.INTERLINE_EQUATION,
}
TEXT_MERGE_BARRIER_TYPES = {
    *SECTION_MERGE_BARRIER_TYPES,
    BlockType.LIST,
}
_CROSS_PAGE_MERGE_KEY = "_cross_page_merge_prev"
OCR_DET_LINES_KEY = "_ocr_det_lines"


def iter_block_spans(block):
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            yield span

    for sub_block in block.get("blocks", []):
        yield from iter_block_spans(sub_block)


def build_para_blocks_from_preproc(pdf_info_list):
    for page_info in pdf_info_list:
        page_info["para_blocks"] = copy.deepcopy(page_info.get("preproc_blocks", []))


def merge_para_text_blocks(pdf_info_list, allow_cross_page=False, auto_merge_by_det=False):
    ordered_blocks = []
    for page_info in pdf_info_list:
        page_idx = page_info.get("page_idx")
        for order_idx, block in enumerate(page_info.get("para_blocks", [])):
            ordered_blocks.append((page_idx, order_idx, block))

    for current_index in range(len(ordered_blocks) - 1, -1, -1):
        current_page_idx, _, current_block = ordered_blocks[current_index]
        current_type = current_block.get("type")
        if current_type == BlockType.TEXT:
            if not _block_has_lines(current_block):
                continue
            _merge_current_text_block(
                ordered_blocks,
                current_index,
                current_page_idx,
                current_block,
                allow_cross_page,
                auto_merge_by_det,
            )
        elif current_type == BlockType.LIST:
            if not current_block.get("blocks"):
                continue
            _merge_current_ref_text_list_block(
                ordered_blocks,
                current_index,
                current_page_idx,
                current_block,
                allow_cross_page,
            )


def _merge_current_text_block(
    ordered_blocks,
    current_index,
    current_page_idx,
    current_block,
    allow_cross_page,
    auto_merge_by_det,
):
    """处理当前 text block 的 merge_prev 强制合并和 Hybrid det 自动合并。"""
    previous_block = None
    if current_block.get("merge_prev"):
        previous_block = _find_previous_text_block(
            ordered_blocks,
            current_index,
            current_block,
            current_page_idx,
            allow_cross_page=allow_cross_page,
        )

    if previous_block is None and auto_merge_by_det:
        previous_block = _find_previous_text_block(
            ordered_blocks,
            current_index,
            current_block,
            current_page_idx,
            allow_cross_page=allow_cross_page,
            force_cross_page=allow_cross_page,
        )
        if previous_block is not None:
            previous_page_idx, _, previous_text_block = previous_block
            is_cross_page = current_page_idx != previous_page_idx
            if can_auto_merge_text_blocks(
                current_block,
                previous_text_block,
                is_cross_page=is_cross_page,
            ):
                if is_cross_page:
                    current_block[_CROSS_PAGE_MERGE_KEY] = True
            else:
                previous_block = None

    if previous_block is None:
        return

    previous_page_idx, _, previous_text_block = previous_block
    _merge_text_block(
        current_block,
        previous_text_block,
        is_cross_page=current_page_idx != previous_page_idx,
    )


def _merge_current_ref_text_list_block(
    ordered_blocks,
    current_index,
    current_page_idx,
    current_block,
    allow_cross_page,
):
    """处理当前 ref_text list 与前一个相邻 ref_text list 的合并。"""
    previous_block = _find_previous_ref_text_list_block(
        ordered_blocks,
        current_index,
        current_page_idx,
        current_block,
        allow_cross_page,
    )
    if previous_block is None:
        return

    previous_page_idx, _, previous_list_block = previous_block
    _merge_ref_text_list_block(
        current_block,
        previous_list_block,
        is_cross_page=current_page_idx != previous_page_idx,
    )


def can_auto_merge_text_blocks(current_block, previous_block, is_cross_page=False):
    """基于 OCR det 行几何和 canonical 文本首尾规则判断 Hybrid text 是否应自动合并。"""
    current_lines = current_block.get("lines", [])
    previous_lines = previous_block.get("lines", [])
    current_metric_lines = _resolve_auto_metric_lines(current_block)
    previous_metric_lines = _resolve_auto_metric_lines(previous_block)
    if not current_lines or not previous_lines or not current_metric_lines or not previous_metric_lines:
        return False

    first_metric_line = current_metric_lines[0]
    last_metric_line = previous_metric_lines[-1]
    first_line_height = _line_height(first_metric_line)
    last_line_height = _line_height(last_metric_line)
    if first_line_height <= 0 or last_line_height <= 0:
        return False

    current_bbox_fs = _build_bbox_fs(current_block, current_metric_lines)
    previous_bbox_fs = _build_bbox_fs(previous_block, previous_metric_lines)
    if abs(current_bbox_fs[0] - first_metric_line["bbox"][0]) >= first_line_height / 2:
        return False
    if abs(previous_bbox_fs[2] - last_metric_line["bbox"][2]) >= last_line_height:
        return False

    first_content = _first_non_empty_content(current_lines)
    last_content = _last_non_empty_content(previous_lines)
    if not first_content or not last_content:
        return False
    if last_content.endswith(LINE_STOP_FLAG):
        return False
    if first_content[0].isdigit() or first_content[0].isupper():
        return False

    current_metric_width = current_bbox_fs[2] - current_bbox_fs[0]
    previous_metric_width = previous_bbox_fs[2] - previous_bbox_fs[0]
    min_metric_width = min(current_metric_width, previous_metric_width)
    if min_metric_width <= 0:
        return False
    if abs(current_metric_width - previous_metric_width) >= min_metric_width:
        return False

    if len(current_metric_lines) <= 1 and len(previous_metric_lines) <= 1:
        return False
    return _has_mergeable_block_bbox_relation(current_block, previous_block)


def cleanup_internal_para_block_metadata(pdf_info_list):
    for page_info in pdf_info_list:
        for block_key in ["preproc_blocks", "para_blocks", "discarded_blocks"]:
            for block in page_info.get(block_key, []):
                _cleanup_block_internal_metadata(block)


def _find_previous_text_block(
    ordered_blocks,
    current_index,
    current_block,
    current_page_idx,
    allow_cross_page,
    force_cross_page=False,
):
    cross_page_allowed = allow_cross_page and (
        force_cross_page or current_block.get(_CROSS_PAGE_MERGE_KEY, False)
    )

    for previous_index in range(current_index - 1, -1, -1):
        previous_page_idx, _, previous_block = ordered_blocks[previous_index]
        if previous_page_idx != current_page_idx and not cross_page_allowed:
            return None

        previous_type = previous_block.get("type")
        if previous_type in TEXT_MERGE_BARRIER_TYPES:
            return None
        if previous_type != BlockType.TEXT:
            continue
        return ordered_blocks[previous_index]

    return None


def _find_previous_ref_text_list_block(
    ordered_blocks,
    current_index,
    current_page_idx,
    current_block,
    allow_cross_page,
):
    """查找紧邻当前 list 的前一个 ref_text list，避免跨正文或标题误合并。"""
    previous_index = current_index - 1
    if previous_index < 0:
        return None

    previous_page_idx, _, previous_block = ordered_blocks[previous_index]
    if previous_page_idx != current_page_idx and not allow_cross_page:
        return None
    if previous_block.get("type") in SECTION_MERGE_BARRIER_TYPES:
        return None
    if previous_block.get("type") != BlockType.LIST:
        return None
    if not _is_ref_text_list_block(current_block) or not _is_ref_text_list_block(previous_block):
        return None
    return ordered_blocks[previous_index]


def _is_ref_text_list_block(block):
    """判断 list block 是否为引用文本列表，只允许这种列表自动拼接。"""
    return block.get("type") == BlockType.LIST and block.get("sub_type") == BlockType.REF_TEXT


def _resolve_auto_metric_lines(block):
    """优先使用 OCR det 行提示；没有提示时退回 block 自身 lines。"""
    return block.get(OCR_DET_LINES_KEY) or block.get("lines", [])


def _merge_text_block(current_block, previous_block, is_cross_page):
    if is_cross_page:
        for line in current_block.get("lines", []):
            for span in line.get("spans", []):
                span[SplitFlag.CROSS_PAGE] = True

    previous_block.setdefault("lines", []).extend(current_block.get("lines", []))
    if current_block.get(OCR_DET_LINES_KEY):
        previous_block.setdefault(OCR_DET_LINES_KEY, []).extend(
            current_block.get(OCR_DET_LINES_KEY, [])
        )
    current_block["lines"] = []
    current_block[OCR_DET_LINES_KEY] = []
    current_block[SplitFlag.LINES_DELETED] = True


def _merge_ref_text_list_block(current_block, previous_block, is_cross_page):
    """合并相邻 ref_text list，并在跨页时给当前 list 内 span 标记跨页。"""
    if is_cross_page:
        for span in iter_block_spans(current_block):
            span[SplitFlag.CROSS_PAGE] = True

    previous_block.setdefault("blocks", []).extend(current_block.get("blocks", []))
    current_block["blocks"] = []
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


def _first_non_empty_content(lines):
    """从行列表中提取第一个非空 span 文本，用于段落起始字符规则判断。"""
    for line in lines:
        for span in line.get("spans", []):
            content = span.get("content", "")
            if content:
                return content
    return ""


def _last_non_empty_content(lines):
    """从行列表中提取最后一个非空 span 文本，用于段落结尾字符规则判断。"""
    for line in reversed(lines):
        for span in reversed(line.get("spans", [])):
            content = span.get("content", "")
            if content:
                return content
    return ""


def _has_mergeable_block_bbox_relation(current_block, previous_block):
    """复刻 pipeline text 合并的核心几何条件：当前块上边界进入前块范围。"""
    return current_block["bbox"][1] < previous_block["bbox"][3]


def _cleanup_block_internal_metadata(block):
    """递归清理只供 Hybrid 内部段落合并使用的临时字段。"""
    block.pop(_CROSS_PAGE_MERGE_KEY, None)
    block.pop(OCR_DET_LINES_KEY, None)
    for sub_block in block.get("blocks", []):
        _cleanup_block_internal_metadata(sub_block)
