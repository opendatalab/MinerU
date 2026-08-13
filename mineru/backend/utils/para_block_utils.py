# Copyright (c) Opendatalab. All rights reserved.
import copy
import math
from typing import Any, TypeAlias

from ...types import Block, BlockType, PageInfo

LINE_STOP_FLAG = (".", "!", "?", "。", "！", "？", ")", "）", '"', "”", ":", "：", ";", "；")
SECTION_MERGE_BARRIER_TYPES = {
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
    BlockType.EQUATION,
}
TEXT_MERGE_BARRIER_TYPES = {
    *SECTION_MERGE_BARRIER_TYPES,
    BlockType.LIST,
}
# 文本段落合并只允许跨过这些视觉根块，避免 ref_text/phonetic 等语义块被当作透明块。
TEXT_MERGE_TRANSPARENT_TYPES = {
    BlockType.IMAGE,
    BlockType.TABLE,
    BlockType.CHART,
    BlockType.CODE,
}
VERTICAL_LINE_HEIGHT_TO_WIDTH_RATIO_THRESHOLD = 2
VERTICAL_LINE_IN_BLOCK_THRESHOLD = 0.8

BlockDict: TypeAlias = dict[str, Any]
CalculationBBox: TypeAlias = tuple[int, int, int, int]
OrderedBlock: TypeAlias = tuple[int, int, BlockDict]


def build_para_blocks_from_preproc(pages: list[PageInfo]) -> None:
    for page_info in pages:
        page_info.para_blocks = copy.deepcopy(page_info.preproc_blocks)


def merge_para_text_blocks(pages: list[dict[str, Any]]) -> None:
    """按页面阅读顺序给可延续的文本或参考文献列表写入 continues_prev 标记。"""
    ordered_blocks: list[OrderedBlock] = []
    for page_info in pages:
        blocks = page_info.get("blocks")
        if not isinstance(blocks, list):
            continue

        for block in blocks:
            if isinstance(block, dict):
                _clear_nested_continues_prev(block)
                if block.get("type") != BlockType.TEXT:
                    block.pop("continues_prev", None)

        page_idx = page_info.get("page_idx")
        if not isinstance(page_idx, int):
            continue
        for order_idx, block in enumerate(blocks):
            if isinstance(block, dict):
                ordered_blocks.append((page_idx, order_idx, block))

    for current_index in range(len(ordered_blocks) - 1, -1, -1):
        current_page_idx, _, current_block = ordered_blocks[current_index]
        current_type = current_block.get("type")
        if current_type == BlockType.TEXT:
            # 已清理过 lines 的结果视为 finalize 完成，保留其既有标记以支持幂等调用。
            if "lines" not in current_block:
                continue
            current_block.pop("continues_prev", None)
            previous_block = _find_previous_text_block(ordered_blocks, current_index)
            if previous_block is None:
                continue
            previous_page_idx, _, previous_text_block = previous_block
            if not _is_same_or_consecutive_page(current_page_idx, previous_page_idx):
                continue
            if can_auto_merge_text_blocks(current_block, previous_text_block):
                current_block["continues_prev"] = True
        elif current_type == BlockType.LIST:
            previous_block = _find_previous_ref_text_list_block(
                ordered_blocks,
                current_index,
                current_block,
            )
            if previous_block is None:
                continue
            previous_page_idx, _, _ = previous_block
            if _is_same_or_consecutive_page(current_page_idx, previous_page_idx):
                current_block["continues_prev"] = True

    for page_info in pages:
        blocks = page_info.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if isinstance(block, dict):
                _remove_line_metadata(block)


def can_auto_merge_text_blocks(current_block: BlockDict, previous_block: BlockDict) -> bool:
    """按文本首尾、行方向和几何关系判断两个 dict text block 是否可连续。"""
    current_metric_lines = _metric_line_bboxes(current_block)
    previous_metric_lines = _metric_line_bboxes(previous_block)
    if not current_metric_lines or not previous_metric_lines:
        return False

    current_bbox = _bbox_for_calculation(current_block.get("bbox"))
    previous_bbox = _bbox_for_calculation(previous_block.get("bbox"))
    if current_bbox is None or previous_bbox is None:
        return False

    current_content = _normalized_text_content(current_block)
    previous_content = _normalized_text_content(previous_block)
    if not current_content or not previous_content:
        return False

    current_is_vertical = _is_vertical_text_block_by_lines(current_metric_lines)
    previous_is_vertical = _is_vertical_text_block_by_lines(previous_metric_lines)
    if current_is_vertical != previous_is_vertical:
        return False
    if current_is_vertical:
        return _can_auto_merge_vertical_text_blocks(
            current_content,
            previous_content,
            current_bbox,
            previous_bbox,
            current_metric_lines,
            previous_metric_lines,
        )
    return _can_auto_merge_horizontal_text_blocks(
        current_content,
        previous_content,
        current_bbox,
        previous_bbox,
        current_metric_lines,
        previous_metric_lines,
    )


def cleanup_internal_para_block_metadata(pages: list[PageInfo]) -> None:
    for page_info in pages:
        for block in page_info.preproc_blocks:
            _cleanup_block_internal_metadata(block)
        for block in page_info.para_blocks:
            _cleanup_block_internal_metadata(block)
        for block in page_info.discarded_blocks:
            _cleanup_block_internal_metadata(block)


def _find_previous_text_block(
    ordered_blocks: list[OrderedBlock],
    current_index: int,
) -> OrderedBlock | None:
    """向前查找 text，视觉根块可跨过，结构或其他语义块会阻断查找。"""
    for previous_index in range(current_index - 1, -1, -1):
        previous_block = ordered_blocks[previous_index][2]
        previous_type = previous_block.get("type")
        if previous_type in TEXT_MERGE_BARRIER_TYPES:
            return None
        if previous_type != BlockType.TEXT:
            if previous_type not in TEXT_MERGE_TRANSPARENT_TYPES:
                return None
            continue
        return ordered_blocks[previous_index]
    return None


def _find_previous_ref_text_list_block(
    ordered_blocks: list[OrderedBlock],
    current_index: int,
    current_block: BlockDict,
) -> OrderedBlock | None:
    """查找紧邻当前块的前一个 ref_text list，列表之间不允许跨过其他块。"""
    previous_index = current_index - 1
    if previous_index < 0:
        return None
    previous_block = ordered_blocks[previous_index][2]
    if not _is_ref_text_list_block(current_block) or not _is_ref_text_list_block(previous_block):
        return None
    return ordered_blocks[previous_index]


def _is_ref_text_list_block(block: BlockDict) -> bool:
    """判断当前 dict block 是否为参考文献列表。"""
    return block.get("type") == BlockType.LIST and block.get("sub_type") == BlockType.REF_TEXT


def _is_same_or_consecutive_page(current_page_idx: int, previous_page_idx: int) -> bool:
    """只允许同页或页码严格连续的前后页建立延续关系。"""
    return current_page_idx == previous_page_idx or current_page_idx == previous_page_idx + 1


def _bbox_for_calculation(bbox: Any) -> CalculationBBox | None:
    """复制并将 0～1 bbox 放大为千分位整数，原始 bbox 保持不变。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        return None

    x0, y0, x1, y1 = (int(round(value * 1000)) for value in values)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _metric_line_bboxes(block: BlockDict) -> list[CalculationBBox]:
    """读取 block.lines 的全部合法行框，任一行非法时整块按不可合并处理。"""
    lines = block.get("lines")
    if not isinstance(lines, list) or not lines:
        return []

    line_bboxes: list[CalculationBBox] = []
    for line in lines:
        if not isinstance(line, dict):
            return []
        line_bbox = _bbox_for_calculation(line.get("bbox"))
        if line_bbox is None:
            return []
        line_bboxes.append(line_bbox)
    return line_bboxes


def _normalized_text_content(block: BlockDict) -> str:
    """读取并去除 text block 内容首尾空白，非字符串内容按无文本处理。"""
    content = block.get("content")
    return content.strip() if isinstance(content, str) else ""


def _bbox_union(line_bboxes: list[CalculationBBox]) -> CalculationBBox:
    """聚合全部行框，得到只用于几何判断的文本覆盖范围。"""
    return (
        min(bbox[0] for bbox in line_bboxes),
        min(bbox[1] for bbox in line_bboxes),
        max(bbox[2] for bbox in line_bboxes),
        max(bbox[3] for bbox in line_bboxes),
    )


def _line_height(line_bbox: CalculationBBox) -> int:
    """计算千分位行框高度。"""
    return line_bbox[3] - line_bbox[1]


def _line_width(line_bbox: CalculationBBox) -> int:
    """计算千分位行框宽度。"""
    return line_bbox[2] - line_bbox[0]


def _is_vertical_text_block_by_lines(line_bboxes: list[CalculationBBox]) -> bool:
    """按行框高宽比判断 block 是否为竖排文本。"""
    vertical_line_count = sum(
        _line_height(line_bbox) / _line_width(line_bbox) > VERTICAL_LINE_HEIGHT_TO_WIDTH_RATIO_THRESHOLD
        for line_bbox in line_bboxes
    )
    return vertical_line_count / len(line_bboxes) > VERTICAL_LINE_IN_BLOCK_THRESHOLD


def _has_mergeable_text_boundary(current_content: str, previous_content: str) -> bool:
    """使用前块结尾和后块开头字符排除明显的新段落边界。"""
    if previous_content.endswith(LINE_STOP_FLAG):
        return False
    first_char = current_content[0]
    return not first_char.isdigit() and not first_char.isupper()


def _can_auto_merge_horizontal_text_blocks(
    current_content: str,
    previous_content: str,
    current_bbox: CalculationBBox,
    previous_bbox: CalculationBBox,
    current_lines: list[CalculationBBox],
    previous_lines: list[CalculationBBox],
) -> bool:
    """使用横排段落的首行、末行、宽度和 block 相交规则判断是否连续。"""
    first_line = current_lines[0]
    last_line = previous_lines[-1]
    first_line_height = _line_height(first_line)
    last_line_height = _line_height(last_line)
    if first_line_height <= 0 or last_line_height <= 0:
        return False

    current_lines_bbox = _bbox_union(current_lines)
    previous_lines_bbox = _bbox_union(previous_lines)
    if abs(current_lines_bbox[0] - first_line[0]) >= first_line_height / 2:
        return False
    if abs(previous_lines_bbox[2] - last_line[2]) >= last_line_height:
        return False
    if not _has_mergeable_text_boundary(current_content, previous_content):
        return False

    current_width = current_lines_bbox[2] - current_lines_bbox[0]
    previous_width = previous_lines_bbox[2] - previous_lines_bbox[0]
    min_width = min(current_width, previous_width)
    if min_width <= 0 or abs(current_width - previous_width) >= min_width:
        return False
    if len(current_lines) <= 1 and len(previous_lines) <= 1:
        return False
    return current_bbox[1] < previous_bbox[3]


def _can_auto_merge_vertical_text_blocks(
    current_content: str,
    previous_content: str,
    current_bbox: CalculationBBox,
    previous_bbox: CalculationBBox,
    current_lines: list[CalculationBBox],
    previous_lines: list[CalculationBBox],
) -> bool:
    """使用竖排段落的首列、末列、高度和 block 相交规则判断是否连续。"""
    first_line = current_lines[0]
    last_line = previous_lines[-1]
    first_line_width = _line_width(first_line)
    last_line_width = _line_width(last_line)
    if first_line_width <= 0 or last_line_width <= 0:
        return False

    current_lines_bbox = _bbox_union(current_lines)
    previous_lines_bbox = _bbox_union(previous_lines)
    if abs(current_lines_bbox[1] - first_line[1]) >= first_line_width / 2:
        return False
    if abs(previous_lines_bbox[3] - last_line[3]) >= last_line_width:
        return False
    if not _has_mergeable_text_boundary(current_content, previous_content):
        return False

    current_height = current_lines_bbox[3] - current_lines_bbox[1]
    previous_height = previous_lines_bbox[3] - previous_lines_bbox[1]
    min_height = min(current_height, previous_height)
    if min_height <= 0 or abs(current_height - previous_height) >= min_height:
        return False
    return current_bbox[2] > previous_bbox[0]


def _clear_nested_continues_prev(block: BlockDict) -> None:
    """递归清理子块旧标记，顶层 text 标记由是否仍有 lines 决定是否重算。"""
    content = block.get("content")
    if not isinstance(content, list):
        return
    for child_block in content:
        if isinstance(child_block, dict):
            child_block.pop("continues_prev", None)
            _clear_nested_continues_prev(child_block)


def _remove_line_metadata(block: BlockDict) -> None:
    """递归删除顶层及嵌套 block 的临时 lines 字段。"""
    block.pop("lines", None)
    content = block.get("content")
    if not isinstance(content, list):
        return
    for child_block in content:
        if isinstance(child_block, dict):
            _remove_line_metadata(child_block)


def _cleanup_block_internal_metadata(block: Block) -> None:
    """递归清理只供 finalize 内部流程使用的临时字段。"""
    block._ocr_det_lines = []
    block._line_avg_height = 0
    for sub_block in block.blocks:
        _cleanup_block_internal_metadata(sub_block)
