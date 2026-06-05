# Copyright (c) Opendatalab. All rights reserved.
import copy
from collections.abc import Generator

from ...types import BBox, Block, Line, PageInfo, Span
from ...utils.enum_class import BlockType

LINE_STOP_FLAG = (".", "!", "?", "。", "！", "？", ")", "）", '"', "”", ":", "：", ";", "；")
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
# 文本段落合并只允许跨过这些视觉根块，避免 ref_text/phonetic 等语义块被当作透明块。
TEXT_MERGE_TRANSPARENT_TYPES = {
    BlockType.IMAGE,
    BlockType.TABLE,
    BlockType.CHART,
    BlockType.CODE,
}


def iter_block_spans(block: Block) -> Generator[Span, None, None]:
    for line in block.lines:
        for span in line.spans:
            yield span

    for sub_block in block.blocks:
        yield from iter_block_spans(sub_block)


def build_para_blocks_from_preproc(pages: list[PageInfo]) -> None:
    for page_info in pages:
        page_info.para_blocks = copy.deepcopy(page_info.preproc_blocks)


def merge_para_text_blocks(pages: list[PageInfo], auto_merge_by_det: bool = False) -> None:
    ordered_blocks: list[tuple[int, int, Block]] = []
    for page_info in pages:
        page_idx = page_info.page_idx
        for order_idx, block in enumerate(page_info.para_blocks):
            ordered_blocks.append((page_idx, order_idx, block))

    for current_index in range(len(ordered_blocks) - 1, -1, -1):
        current_page_idx, _, current_block = ordered_blocks[current_index]
        current_type = current_block.type
        if current_type == BlockType.TEXT:
            if not _block_has_lines(current_block):
                continue
            _merge_current_text_block(
                ordered_blocks,
                current_index,
                current_page_idx,
                current_block,
                auto_merge_by_det,
            )
        elif current_type == BlockType.LIST:
            if not current_block.blocks:
                continue
            _merge_current_ref_text_list_block(
                ordered_blocks,
                current_index,
                current_block,
            )


def _merge_current_text_block(
    ordered_blocks: list[tuple[int, int, Block]],
    current_index: int,
    current_page_idx: int,
    current_block: Block,
    auto_merge_by_det: bool,
) -> None:
    """处理当前 text block 的 merge_prev 候选合并和 Hybrid det 自动合并。"""
    previous_block = None
    if current_block.merge_prev:
        previous_block = _find_previous_merge_prev_text_block(
            ordered_blocks,
            current_index,
            current_page_idx,
        )
        if previous_block is not None:
            _, _, previous_text_block = previous_block
            if not can_auto_merge_text_blocks(
                current_block,
                previous_text_block,
                allow_single_line_blocks=True,
            ):
                previous_block = None

    if previous_block is None and auto_merge_by_det:
        previous_block = _find_previous_text_block(
            ordered_blocks,
            current_index,
        )
        if previous_block is not None:
            _, _, previous_text_block = previous_block
            if not can_auto_merge_text_blocks(
                current_block,
                previous_text_block,
            ):
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
    ordered_blocks: list[tuple[int, int, Block]],
    current_index: int,
    current_block: Block,
) -> None:
    """处理当前 ref_text list 与前一个相邻 ref_text list 的合并。"""
    previous_block = _find_previous_ref_text_list_block(
        ordered_blocks,
        current_index,
        current_block,
    )
    if previous_block is None:
        return

    current_page_idx, _, _ = ordered_blocks[current_index]
    previous_page_idx, _, previous_list_block = previous_block
    _merge_ref_text_list_block(
        current_block,
        previous_list_block,
        is_cross_page=current_page_idx != previous_page_idx,
    )


def can_auto_merge_text_blocks(
    current_block: Block,
    previous_block: Block,
    allow_single_line_blocks: bool = False,
) -> bool:
    """按段落首尾文本和行几何规则判断 text 是否可合并。"""
    current_lines = current_block.lines
    previous_lines = previous_block.lines
    current_metric_lines = _resolve_local_metric_lines(current_block)
    previous_metric_lines = _resolve_local_metric_lines(previous_block)
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
    if abs(current_bbox_fs[0] - first_metric_line.bbox[0]) >= first_line_height / 2:
        return False
    if abs(previous_bbox_fs[2] - last_metric_line.bbox[2]) >= last_line_height:
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

    if not allow_single_line_blocks and len(current_metric_lines) <= 1 and len(previous_metric_lines) <= 1:
        return False
    return _has_mergeable_block_bbox_relation(current_block, previous_block)


def cleanup_internal_para_block_metadata(pages: list[PageInfo]) -> None:
    for page_info in pages:
        for block in page_info.preproc_blocks:
            _cleanup_block_internal_metadata(block)
        for block in page_info.para_blocks:
            _cleanup_block_internal_metadata(block)
        for block in page_info.discarded_blocks:
            _cleanup_block_internal_metadata(block)


def _find_previous_text_block(
    ordered_blocks: list[tuple[int, int, Block]],
    current_index: int,
) -> tuple[int, int, Block] | None:
    """查找前序 text；除 merge_prev 专用路径外默认允许跨页查找。"""
    for previous_index in range(current_index - 1, -1, -1):
        _, _, previous_block = ordered_blocks[previous_index]

        previous_type = previous_block.type
        if previous_type in TEXT_MERGE_BARRIER_TYPES:
            return None
        if previous_type != BlockType.TEXT:
            if previous_type not in TEXT_MERGE_TRANSPARENT_TYPES:
                return None
            continue
        return ordered_blocks[previous_index]

    return None


def _find_previous_merge_prev_text_block(
    ordered_blocks: list[tuple[int, int, Block]],
    current_index: int,
    current_page_idx: int,
) -> tuple[int, int, Block] | None:
    """查找同页 merge_prev 提示对应的前序 text，但不跨越段落合并屏障。"""
    for previous_index in range(current_index - 1, -1, -1):
        previous_page_idx, _, previous_block = ordered_blocks[previous_index]
        if previous_page_idx != current_page_idx:
            return None

        previous_type = previous_block.type
        if previous_type in TEXT_MERGE_BARRIER_TYPES:
            return None
        if previous_type != BlockType.TEXT:
            if previous_type not in TEXT_MERGE_TRANSPARENT_TYPES:
                return None
            continue
        return ordered_blocks[previous_index]

    return None


def _find_previous_ref_text_list_block(
    ordered_blocks: list[tuple[int, int, Block]],
    current_index: int,
    current_block: Block,
) -> tuple[int, int, Block] | None:
    """查找紧邻当前 list 的前一个 ref_text list，默认允许跨页拼接。"""
    previous_index = current_index - 1
    if previous_index < 0:
        return None

    _, _, previous_block = ordered_blocks[previous_index]
    if previous_block.type in SECTION_MERGE_BARRIER_TYPES:
        return None
    if previous_block.type != BlockType.LIST:
        return None
    if not _is_ref_text_list_block(current_block) or not _is_ref_text_list_block(previous_block):
        return None
    return ordered_blocks[previous_index]


def _is_ref_text_list_block(block: Block) -> bool:
    """判断 list block 是否为引用文本列表，只允许这种列表自动拼接。"""
    return block.type == BlockType.LIST and block.sub_type == BlockType.REF_TEXT


def _resolve_auto_metric_lines(block: Block) -> list[Line]:
    """优先使用 OCR det 行提示；没有提示时退回 block 自身 lines。"""
    return block._ocr_det_lines or block.lines


def _resolve_local_metric_lines(block: Block) -> list[Line]:
    """过滤跨页追加行，避免已合并内容污染后续本地几何判定。"""
    metric_lines = _resolve_auto_metric_lines(block)
    local_metric_lines = [line for line in metric_lines if not _is_cross_page_line(line)]
    return local_metric_lines or metric_lines


def _merge_text_block(current_block: Block, previous_block: Block, is_cross_page: bool) -> None:
    if is_cross_page:
        _mark_lines_cross_page(current_block.lines)
        _mark_lines_cross_page(current_block._ocr_det_lines)

    previous_block.lines.extend(current_block.lines)
    previous_block._ocr_det_lines.extend(current_block._ocr_det_lines)

    current_block.lines = []
    current_block._ocr_det_lines = []
    current_block._lines_deleted = True


def _mark_lines_cross_page(lines: list[Line]) -> None:
    """给跨页合并进来的文本行和 det hint 行同步打跨页标记。"""
    for line in lines:
        for span in line.spans:
            span._cross_page = True


def _is_cross_page_line(line: Line) -> bool:
    """判断整行是否来自跨页追加，供几何度量时排除。"""
    spans = line.spans
    return bool(spans) and all(span._cross_page for span in spans)


def _merge_ref_text_list_block(current_block: Block, previous_block: Block, is_cross_page: bool) -> None:
    """合并相邻 ref_text list，并在跨页时给当前 list 内 span 标记跨页。"""
    if is_cross_page:
        for span in iter_block_spans(current_block):
            span._cross_page = True

    previous_block.blocks.extend(current_block.blocks)
    current_block.blocks = []
    current_block._lines_deleted = True


def _line_height(line: Line) -> float:
    bbox = line.bbox
    if not bbox:
        return 0
    return bbox[3] - bbox[1]


def _build_bbox_fs(block: Block, lines: list[Line]) -> BBox:
    if lines:
        return (
            min(line.bbox[0] for line in lines),
            min(line.bbox[1] for line in lines),
            max(line.bbox[2] for line in lines),
            max(line.bbox[3] for line in lines),
        )
    return block.bbox


def _block_has_lines(block: Block) -> bool:
    return any(line.spans for line in block.lines)


def _first_non_empty_content(lines: list[Line]) -> str:
    """从行列表中提取第一个非空 span 文本，用于段落起始字符规则判断。"""
    for line in lines:
        for span in line.spans:
            content = span.content or ""
            if content:
                return content
    return ""


def _last_non_empty_content(lines: list[Line]) -> str:
    """从行列表中提取最后一个非空 span 文本，用于段落结尾字符规则判断。"""
    for line in reversed(lines):
        for span in reversed(line.spans):
            content = span.content or ""
            if content:
                return content
    return ""


def _has_mergeable_block_bbox_relation(current_block: Block, previous_block: Block) -> bool:
    """复刻 pipeline text 合并的核心几何条件：当前块上边界进入前块范围。"""
    return current_block.bbox[1] < previous_block.bbox[3]


def _cleanup_block_internal_metadata(block: Block) -> None:
    """递归清理只供 finalize 内部流程使用的临时字段。"""
    block._ocr_det_lines = []
    block._line_avg_height = 0
    for sub_block in block.blocks:
        _cleanup_block_internal_metadata(sub_block)
