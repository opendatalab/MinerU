# Copyright (c) Opendatalab. All rights reserved.
"""PDF 段落延续关系的 raw model-list 后处理。"""

import math
from typing import Any, TypeAlias

from ...types import PAGE_AUXILIARY_BLOCK_TYPES, BlockType

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
# 文本段落合并允许跨过视觉根块和页面装饰块，其他语义块仍会阻断候选查找。
TEXT_MERGE_TRANSPARENT_TYPES = {
    BlockType.IMAGE,
    BlockType.TABLE,
    BlockType.CHART,
    BlockType.CODE,
    *PAGE_AUXILIARY_BLOCK_TYPES,
}
VERTICAL_LINE_HEIGHT_TO_WIDTH_RATIO_THRESHOLD = 2
VERTICAL_LINE_IN_BLOCK_THRESHOLD = 0.8
SINGLE_LINE_LOOKAHEAD_LIMIT = 5
SINGLE_LINE_MIN_ALIGNED_LOOKAHEAD = 3
SINGLE_LINE_THICKNESS_RATIO_MAX = 1.5

BlockDict: TypeAlias = dict[str, Any]
CalculationBBox: TypeAlias = tuple[int, int, int, int]
OrderedBlock: TypeAlias = tuple[int, int, BlockDict]


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
            if can_auto_merge_text_blocks(
                current_block,
                previous_text_block,
            ) or _can_auto_merge_multiline_to_single_line(
                current_block,
                previous_text_block,
                ordered_blocks=ordered_blocks,
                current_index=current_index,
                current_page_idx=current_page_idx,
            ):
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


def _find_previous_text_block(
    ordered_blocks: list[OrderedBlock],
    current_index: int,
) -> OrderedBlock | None:
    """向前查找 text，视觉根块和页面装饰块可跨过，其他语义块会阻断查找。"""
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
    """跳过页面辅助块查找前一个 ref_text list，其他语义块保持阻断。"""
    if not _is_ref_text_list_block(current_block):
        return None
    for previous_index in range(current_index - 1, -1, -1):
        previous_block = ordered_blocks[previous_index][2]
        if previous_block.get("type") in PAGE_AUXILIARY_BLOCK_TYPES:
            continue
        if _is_ref_text_list_block(previous_block):
            return ordered_blocks[previous_index]
        return None
    return None


def _is_ref_text_list_block(block: BlockDict) -> bool:
    """判断当前 dict block 是否为参考文献列表。"""
    return block.get("type") == BlockType.LIST and block.get("sub_type") == BlockType.REF_TEXT


def _is_same_or_consecutive_page(current_page_idx: int, previous_page_idx: int) -> bool:
    """只允许同页或页码严格连续的前后页建立延续关系。"""
    return current_page_idx == previous_page_idx or current_page_idx == previous_page_idx + 1


def _positive_values_have_max_ratio(first: float, second: float, max_ratio: float) -> bool:
    """判断两个正值的较大较小比是否不超过给定上限。"""
    if first <= 0 or second <= 0:
        return False
    return max(first, second) / min(first, second) <= max_ratio


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


def _collect_following_same_orientation_lines(
    ordered_blocks: list[OrderedBlock],
    current_index: int,
    *,
    current_page_idx: int,
    is_vertical: bool,
) -> list[CalculationBBox]:
    """在当前页向后读取至多五条同方向正文行，语义屏障或非法文本会终止读取。"""
    following_lines: list[CalculationBBox] = []
    for page_idx, _, block in ordered_blocks[current_index + 1 :]:
        if page_idx != current_page_idx:
            break
        block_type = block.get("type")
        if block_type in TEXT_MERGE_BARRIER_TYPES:
            break
        if block_type != BlockType.TEXT:
            if block_type in TEXT_MERGE_TRANSPARENT_TYPES:
                continue
            break

        block_lines = _metric_line_bboxes(block)
        if not block_lines or _is_vertical_text_block_by_lines(block_lines) != is_vertical:
            break
        for line_bbox in block_lines:
            following_lines.append(line_bbox)
            if len(following_lines) >= SINGLE_LINE_LOOKAHEAD_LIMIT:
                return following_lines
    return following_lines


def _aligned_following_lines(
    current_line: CalculationBBox,
    following_lines: list[CalculationBBox],
    *,
    is_vertical: bool,
) -> list[CalculationBBox]:
    """按横排左边界或竖排上边界筛选同一虚拟栏内的后续行列。"""
    current_start = current_line[1] if is_vertical else current_line[0]
    current_thickness = _line_width(current_line) if is_vertical else _line_height(current_line)
    aligned_lines: list[CalculationBBox] = []
    for line_bbox in following_lines:
        line_start = line_bbox[1] if is_vertical else line_bbox[0]
        line_thickness = _line_width(line_bbox) if is_vertical else _line_height(line_bbox)
        if not _positive_values_have_max_ratio(
            current_thickness,
            line_thickness,
            SINGLE_LINE_THICKNESS_RATIO_MAX,
        ):
            continue
        if abs(line_start - current_start) <= max(current_thickness, line_thickness):
            aligned_lines.append(line_bbox)
    return aligned_lines


def _virtual_single_line_bbox(
    current_line: CalculationBBox,
    aligned_lines: list[CalculationBBox],
    *,
    is_vertical: bool,
) -> CalculationBBox:
    """仅沿文本主轴扩展单行计算框，原始 line 和 block bbox 保持不变。"""
    if is_vertical:
        return (
            current_line[0],
            current_line[1],
            current_line[2],
            max(current_line[3], *(line_bbox[3] for line_bbox in aligned_lines)),
        )
    return (
        current_line[0],
        current_line[1],
        max(current_line[2], *(line_bbox[2] for line_bbox in aligned_lines)),
        current_line[3],
    )


def _can_auto_merge_multiline_to_single_line(
    current_block: BlockDict,
    previous_block: BlockDict,
    *,
    ordered_blocks: list[OrderedBlock],
    current_index: int,
    current_page_idx: int,
) -> bool:
    """用后续五行或列补足单行主轴尺寸，再复用原横排或竖排连接规则。"""
    current_lines = _metric_line_bboxes(current_block)
    previous_lines = _metric_line_bboxes(previous_block)
    if len(current_lines) != 1 or len(previous_lines) <= 1:
        return False

    current_is_vertical = _is_vertical_text_block_by_lines(current_lines)
    if _is_vertical_text_block_by_lines(previous_lines) != current_is_vertical:
        return False
    following_lines = _collect_following_same_orientation_lines(
        ordered_blocks,
        current_index,
        current_page_idx=current_page_idx,
        is_vertical=current_is_vertical,
    )
    aligned_lines = _aligned_following_lines(
        current_lines[0],
        following_lines,
        is_vertical=current_is_vertical,
    )
    if len(aligned_lines) < SINGLE_LINE_MIN_ALIGNED_LOOKAHEAD:
        return False

    current_bbox = _bbox_for_calculation(current_block.get("bbox"))
    previous_bbox = _bbox_for_calculation(previous_block.get("bbox"))
    current_content = _normalized_text_content(current_block)
    previous_content = _normalized_text_content(previous_block)
    if current_bbox is None or previous_bbox is None or not current_content or not previous_content:
        return False

    virtual_current_line = _virtual_single_line_bbox(
        current_lines[0],
        aligned_lines,
        is_vertical=current_is_vertical,
    )
    if current_is_vertical:
        return _can_auto_merge_vertical_text_blocks(
            current_content,
            previous_content,
            current_bbox,
            previous_bbox,
            [virtual_current_line],
            previous_lines,
        )
    return _can_auto_merge_horizontal_text_blocks(
        current_content,
        previous_content,
        current_bbox,
        previous_bbox,
        [virtual_current_line],
        previous_lines,
    )


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
