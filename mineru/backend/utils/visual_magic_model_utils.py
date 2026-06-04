# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from ...types import BBox, Block
from ...utils.enum_class import BlockType
from .boxbase import bbox_center_distance, bbox_distance, calculate_overlap_area_in_bbox1_area_ratio
from .table_continuation import is_table_continuation_text

IMAGE_BLOCK_BODY = "image_block_body"
GENERIC_CHILD_TYPES = (BlockType.CAPTION, BlockType.FOOTNOTE)
INLINE_CAPTION_FRAGMENT_TYPES = {BlockType.TEXT, BlockType.FOOTNOTE}
STACKED_TABLE_CAPTION_CLUSTER_TYPES = {
    BlockType.CAPTION,
    BlockType.TEXT,
    BlockType.FOOTNOTE,
}
VISUAL_RELATION_IGNORED_TYPES = {
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.PAGE_FOOTNOTE,
    BlockType.ASIDE_TEXT,
}
VISUAL_MAIN_TYPES = {
    BlockType.IMAGE_BODY: BlockType.IMAGE,
    IMAGE_BLOCK_BODY: BlockType.IMAGE,
    BlockType.TABLE_BODY: BlockType.TABLE,
    BlockType.CHART_BODY: BlockType.CHART,
    BlockType.CODE_BODY: BlockType.CODE,
}
VISUAL_TYPE_MAPPING = {
    BlockType.IMAGE: {
        "body": BlockType.IMAGE_BODY,
        "caption": BlockType.IMAGE_CAPTION,
        "footnote": BlockType.IMAGE_FOOTNOTE,
    },
    BlockType.TABLE: {
        "body": BlockType.TABLE_BODY,
        "caption": BlockType.TABLE_CAPTION,
        "footnote": BlockType.TABLE_FOOTNOTE,
    },
    BlockType.CHART: {
        "body": BlockType.CHART_BODY,
        "caption": BlockType.CHART_CAPTION,
        "footnote": BlockType.CHART_FOOTNOTE,
    },
    BlockType.CODE: {
        "body": BlockType.CODE_BODY,
        "caption": BlockType.CODE_CAPTION,
        "footnote": BlockType.CODE_FOOTNOTE,
    },
}


def isolated_formula_clean(txt: str) -> str:
    latex = txt[:]
    if latex.startswith("\\["):
        latex = latex[2:]
    if latex.endswith("\\]"):
        latex = latex[:-2]
    latex = latex.strip()
    return latex


def code_content_clean(content: str | None) -> str:
    """清理代码内容，移除Markdown代码块的开始和结束标记"""
    if not content:
        return ""

    lines = content.splitlines()
    start_idx = 0
    end_idx = len(lines)

    if lines and lines[0].startswith("```"):
        start_idx = 1

    if lines and end_idx > start_idx and lines[end_idx - 1].strip() == "```":
        end_idx -= 1

    if start_idx < end_idx:
        return "\n".join(lines[start_idx:end_idx]).strip()
    return ""


def clean_content(content: str | None) -> str | None:
    if content and content.count("\\[") == content.count("\\]") and content.count("\\[") > 0:

        def replace_pattern(match: re.Match) -> str:
            inner_content = match.group(1)
            return f"[{inner_content}]"

        pattern = r"\\\[(.*?)\\\]"
        content = re.sub(pattern, replace_pattern, content)

    return content


def fallback_inline_caption_fragments(blocks: list[Block], visual_main_types: dict[Any, Any] | set[Any]) -> None:
    """将紧贴视觉主体上方的同行 text/footnote 片段兜底为通用 caption。"""
    if len(blocks) < 3:
        return

    main_types = set(visual_main_types)
    ordered_blocks = sorted(blocks, key=lambda x: x.index or 0)
    for pos, block in enumerate(ordered_blocks):
        if block.type not in INLINE_CAPTION_FRAGMENT_TYPES:
            continue

        previous_block = find_previous_effective_block(ordered_blocks, pos)
        next_block = find_next_effective_block(ordered_blocks, pos)
        if not (previous_block and next_block and previous_block.type == BlockType.CAPTION and next_block.type in main_types):
            continue

        if not is_inline_caption_fragment(previous_block, block, next_block):
            continue

        block.type = BlockType.CAPTION
        # fallback 后该块已是视觉 caption 片段，不再参与正文跨块合并。
        block.merge_prev = False

    fallback_stacked_table_caption_fragments(blocks, visual_main_types)


def fallback_leading_table_continuation_captions(blocks: list[Block], visual_main_types: dict[Any, Any] | set[Any]) -> None:
    """将页首紧贴表格的续表文本兜底为通用 caption。

    该规则只处理页面有效块开头的 text，避免正文中出现“续表”时被误挂。
    后续 regroup_visual_blocks() 会根据表格主体类型将通用 caption 落成
    table_caption 子块。
    """
    table_main_types = get_table_main_types(visual_main_types)
    if not table_main_types:
        return

    effective_blocks = [
        block for block in sorted(blocks, key=lambda x: x.index or 0) if block.type not in VISUAL_RELATION_IGNORED_TYPES
    ]
    if len(effective_blocks) < 2:
        return

    leading_blocks = []
    cursor = 0
    while cursor < len(effective_blocks):
        block = effective_blocks[cursor]
        if not _is_leading_continuation_text_block(block):
            break
        leading_blocks.append(block)
        cursor += 1

    if not leading_blocks or cursor >= len(effective_blocks):
        return

    table_block = effective_blocks[cursor]
    if table_block.type not in table_main_types:
        return

    if not _is_leading_continuation_cluster_near_table(leading_blocks, table_block):
        return

    for block in leading_blocks:
        block.type = BlockType.CAPTION
        # fallback 后该块已是视觉 caption，不再参与正文跨块合并。
        block.merge_prev = False


def _is_leading_continuation_text_block(block: Block) -> bool:
    """判断页首候选块是否是单行续表文本。"""
    return (
        block.type in INLINE_CAPTION_FRAGMENT_TYPES
        and is_single_line_caption_fragment(block)
        and is_table_continuation_text(_block_text_content(block))
    )


def _block_text_content(block: Block) -> str:
    """提取视觉块中的可见文本，用于续表 marker 判断。"""
    return "".join(span.content for line in block.lines for span in line.spans)


def is_transparent_visual_relation_block(block: Block) -> bool:
    """判断视觉关系中可忽略的结构性空块。"""
    if block.type != BlockType.LIST:
        return False

    if block.blocks:
        return False

    return not _block_text_content(block).strip()


def _is_leading_continuation_cluster_near_table(leading_blocks: list[Block], table_block: Block) -> bool:
    """判断页首续表文本簇是否与后续 table 在几何上相邻。"""
    next_top = table_block.bbox[1]
    max_child_height = 1

    for block in reversed(leading_blocks):
        if not is_horizontally_near_table(block, table_block):
            return False

        block_height = max(block.bbox[3] - block.bbox[1], 1)
        vertical_gap = next_top - block.bbox[3]
        max_gap = stacked_caption_max_gap(max(max_child_height, block_height))
        max_overlap = max(2, block_height * 0.3)
        if vertical_gap > max_gap or vertical_gap < -max_overlap:
            return False

        next_top = block.bbox[1]
        max_child_height = max(max_child_height, block_height)

    return True


def fallback_stacked_table_caption_fragments(blocks: list[Block], visual_main_types: dict[Any, Any] | set[Any]) -> None:
    """将 table 上方紧贴标题簇里的 text/footnote 片段兜底为 caption。"""
    table_main_types = get_table_main_types(visual_main_types)
    if not table_main_types:
        return

    for table_block in blocks:
        if table_block.type not in table_main_types:
            continue

        caption_cluster = find_stacked_table_caption_cluster(table_block, blocks)
        if not caption_cluster:
            continue

        last_caption_pos = find_last_caption_position(caption_cluster)
        if last_caption_pos is None:
            continue

        for block in caption_cluster[last_caption_pos + 1 :]:
            if block.type in INLINE_CAPTION_FRAGMENT_TYPES and is_single_line_caption_fragment(block):
                block.type = BlockType.CAPTION
                # fallback 后该块已是视觉 caption 片段，不再参与正文跨块合并。
                block.merge_prev = False


def get_table_main_types(visual_main_types: dict[Any, Any] | set[Any]) -> set[Any]:
    """根据调用方传入的视觉主体类型，找出 table 对应的主体类型。"""
    if isinstance(visual_main_types, dict):
        return {block_type for block_type, visual_type in visual_main_types.items() if visual_type == BlockType.TABLE}

    main_types = set(visual_main_types)
    return main_types & {BlockType.TABLE, BlockType.TABLE_BODY}


def find_stacked_table_caption_cluster(table_block: Block, blocks: list[Block]) -> list[Block]:
    """按几何位置收集紧贴 table 上方的 caption/text/footnote 标题簇。"""
    table_bbox = table_block.bbox
    table_top = table_bbox[1]
    above_candidates = [
        block
        for block in blocks
        if block is not table_block
        and block.type in STACKED_TABLE_CAPTION_CLUSTER_TYPES
        and block.bbox[3] <= table_top
        and is_horizontally_near_table(block, table_block)
    ]
    if not above_candidates:
        return []

    caption_cluster = []
    next_top = table_top
    max_child_height = 1
    for block in sorted(
        above_candidates,
        key=lambda x: (x.bbox[1], x.index),
        reverse=True,
    ):
        block_height = max(block.bbox[3] - block.bbox[1], 1)
        max_allowed_gap = stacked_caption_max_gap(max(max_child_height, block_height))
        vertical_gap = next_top - block.bbox[3]
        if not 0 <= vertical_gap <= max_allowed_gap:
            break

        caption_cluster.append(block)
        next_top = block.bbox[1]
        max_child_height = max(max_child_height, block_height)

    return list(reversed(caption_cluster))


def find_last_caption_position(caption_cluster: list[Block]) -> int | None:
    """定位标题簇里的最后一个 caption，避免吸收上一张表的尾注。"""
    for pos in range(len(caption_cluster) - 1, -1, -1):
        if caption_cluster[pos].type == BlockType.CAPTION:
            return pos
    return None


def is_horizontally_near_table(block: Block, table_block: Block) -> bool:
    """判断标题簇候选块是否横向落在 table 范围附近。"""
    table_bbox = table_block.bbox
    block_bbox = block.bbox
    table_width = max(table_bbox[2] - table_bbox[0], 1)
    tolerance = max(12, table_width * 0.03)
    return not (block_bbox[2] < table_bbox[0] - tolerance or block_bbox[0] > table_bbox[2] + tolerance)


def is_single_line_caption_fragment(block: Block) -> bool:
    """判断待兜底片段是否是单行块，避免吞掉多行正文。"""
    return len(block.lines or [None]) <= 1


def stacked_caption_max_gap(block_height: float) -> float:
    """计算堆叠标题簇允许的最大纵向间距。"""
    return max(12, block_height * 1.5)


def find_previous_effective_block(ordered_blocks: list[Block], pos: int) -> Block | None:
    """向前查找参与视觉关系判断的有效块，跳过页眉页脚等外围块。"""
    for index in range(pos - 1, -1, -1):
        block = ordered_blocks[index]
        if block.type not in VISUAL_RELATION_IGNORED_TYPES:
            return block
    return None


def find_next_effective_block(ordered_blocks: list[Block], pos: int) -> Block | None:
    """向后查找参与视觉关系判断的有效块，跳过页眉页脚等外围块。"""
    for index in range(pos + 1, len(ordered_blocks)):
        block = ordered_blocks[index]
        if block.type not in VISUAL_RELATION_IGNORED_TYPES:
            return block
    return None


def is_inline_caption_fragment(previous_caption: Block, text_block: Block, next_visual: Block) -> bool:
    """判断当前块是否是前一 caption 的同行补充片段。"""
    caption_bbox = previous_caption.bbox
    text_bbox = text_block.bbox
    visual_bbox = next_visual.bbox

    caption_height = max(caption_bbox[3] - caption_bbox[1], 1)
    text_height = max(text_bbox[3] - text_bbox[1], 1)
    min_text_height = max(min(caption_height, text_height), 1)

    vertical_overlap = min(caption_bbox[3], text_bbox[3]) - max(caption_bbox[1], text_bbox[1])
    center_y_diff = abs(((caption_bbox[1] + caption_bbox[3]) / 2) - ((text_bbox[1] + text_bbox[3]) / 2))
    is_same_line = vertical_overlap / min_text_height >= 0.6 or center_y_diff <= max(caption_height, text_height) * 0.5
    if not is_same_line:
        return False

    vertical_gap_to_visual = visual_bbox[1] - max(caption_bbox[3], text_bbox[3])
    max_allowed_gap = max(12, max(caption_height, text_height) * 1.5)
    return 0 <= vertical_gap_to_visual <= max_allowed_gap


def regroup_visual_blocks(blocks: list[Block]) -> tuple[dict[Any, list[Block]], list[Block]]:
    ordered_blocks = sorted(blocks, key=lambda x: x.index or 0)
    absorbed_member_indices, sub_images_by_index = absorb_image_block_members(ordered_blocks)
    effective_blocks = [block for block in ordered_blocks if block.index not in absorbed_member_indices]
    visual_relation_blocks = [block for block in effective_blocks if not is_transparent_visual_relation_block(block)]
    position_by_index = {block.index: pos for pos, block in enumerate(visual_relation_blocks)}
    main_blocks = [block for block in visual_relation_blocks if block.type in VISUAL_MAIN_TYPES]
    child_blocks = [block for block in visual_relation_blocks if block.type in GENERIC_CHILD_TYPES]

    grouped_children: dict[int, dict[str, list[Block]]] = {
        block.index: {"captions": [], "footnotes": []} for block in main_blocks
    }
    unmatched_child_blocks = []

    for main_block in main_blocks:
        if main_block.index in sub_images_by_index:
            main_block._extra["sub_images"] = sub_images_by_index[main_block.index]

    for child_block in child_blocks:
        parent_block = find_best_visual_parent(
            child_block,
            main_blocks,
            visual_relation_blocks,
            position_by_index,
        )
        if parent_block is None:
            unmatched_child_blocks.append(child_block)
            continue

        child_kind = child_kind_from_type(child_block.type)
        grouped_children[parent_block.index][f"{child_kind}s"].append(child_block)

    grouped_blocks: dict[str, list[Block]] = {
        BlockType.IMAGE: [],
        BlockType.TABLE: [],
        BlockType.CHART: [],
        BlockType.CODE: [],
    }

    for main_block in main_blocks:
        visual_type = VISUAL_MAIN_TYPES[main_block.type]
        mapping = VISUAL_TYPE_MAPPING[visual_type]
        body_block = deepcopy(main_block)
        body_block.type = mapping["body"]
        body_block.pop("sub_images", None)
        body_block.sub_type = ""

        captions = []
        for caption in sorted(
            grouped_children[main_block.index]["captions"],
            key=lambda x: x.index or 0,
        ):
            child_block = deepcopy(caption)
            child_block.type = mapping["caption"]
            captions.append(child_block)

        footnotes = []
        for footnote in sorted(
            grouped_children[main_block.index]["footnotes"],
            key=lambda x: x.index or 0,
        ):
            child_block = deepcopy(footnote)
            child_block.type = mapping["footnote"]
            footnotes.append(child_block)

        two_layer_block = Block(
            index=body_block.index,
            type=visual_type,
            bbox=body_block.bbox,
            blocks=[body_block, *captions, *footnotes],
        )
        if visual_type in [BlockType.IMAGE, BlockType.CHART] and main_block.sub_type:
            two_layer_block.sub_type = main_block.sub_type
        if visual_type == BlockType.IMAGE and main_block._extra.get("sub_images"):
            two_layer_block["sub_images"] = main_block._extra["sub_images"]
        if visual_type == BlockType.TABLE and main_block._cell_merge:
            two_layer_block._cell_merge = main_block._cell_merge
        two_layer_block.blocks.sort(key=lambda x: x.index or 0)

        grouped_blocks[visual_type].append(two_layer_block)

    for blocks_of_type in grouped_blocks.values():
        blocks_of_type.sort(key=lambda x: x.index or 0)

    return grouped_blocks, unmatched_child_blocks


def absorb_image_block_members(blocks: list[Block]) -> tuple[set[int], dict[int, list[Block]]]:
    image_block_bodies = [block for block in blocks if block.type == IMAGE_BLOCK_BODY]
    member_candidates = [block for block in blocks if block.type in [BlockType.IMAGE_BODY, BlockType.CHART_BODY]]

    assignments = {}
    for member in member_candidates:
        best_key = None
        best_parent_index = None
        for image_block in image_block_bodies:
            overlap_ratio = calculate_overlap_area_in_bbox1_area_ratio(
                member.bbox,
                image_block.bbox,
            )
            if overlap_ratio < 0.9:
                continue

            candidate_key = (
                -overlap_ratio,
                bbox_area(image_block.bbox),
                image_block.index,
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_parent_index = image_block.index

        if best_parent_index is not None:
            assignments[member.index] = best_parent_index

    absorbed_member_indices = set()
    sub_images_by_index = {}
    for image_block in image_block_bodies:
        members = [member for member in member_candidates if assignments.get(member.index) == image_block.index]
        if not members:
            continue

        members.sort(key=lambda x: x.index or 0)
        absorbed_member_indices.update(member.index for member in members)
        sub_images_by_index[image_block.index] = [
            {
                "type": child_visual_type(member.type),
                "bbox": relative_bbox(member.bbox, image_block.bbox),
            }
            for member in members
        ]

    return absorbed_member_indices, sub_images_by_index


def find_best_visual_parent(
    child_block: Block,
    main_blocks: list[Block],
    ordered_blocks: list[Block],
    position_by_index: dict[int, int],
    main_type_to_visual_type: dict[Any, Any] | None = None,
    type_by_index: dict[int, str] | None = None,
) -> Block | None:
    """为通用 caption/footnote 查找最合适的视觉主体。"""
    if main_type_to_visual_type is None:
        main_type_to_visual_type = VISUAL_MAIN_TYPES
    candidates = []
    for main_block in main_blocks:
        if not is_visual_neighbor(
            child_block,
            main_block,
            ordered_blocks,
            position_by_index,
            type_by_index=type_by_index,
        ):
            continue

        candidates.append(main_block)

    if not candidates:
        return None

    min_effective_index_diff = min(
        effective_visual_index_diff(
            child_block,
            main_block,
            ordered_blocks,
            type_by_index=type_by_index,
        )
        for main_block in candidates
    )
    closest_index_candidates = [
        main_block
        for main_block in candidates
        if effective_visual_index_diff(
            child_block,
            main_block,
            ordered_blocks,
            type_by_index=type_by_index,
        )
        == min_effective_index_diff
    ]

    if len(closest_index_candidates) == 1:
        return closest_index_candidates[0]

    edge_distances = [(main_block, bbox_distance(child_block.bbox, main_block.bbox)) for main_block in closest_index_candidates]
    edge_values = [edge_distance for _, edge_distance in edge_distances]
    if max(edge_values) - min(edge_values) > 2:
        return min(
            edge_distances,
            key=lambda item: (item[1], item[0].index),
        )[0]

    child_type = block_type(child_block, type_by_index)
    if child_type == BlockType.CAPTION and all(
        main_type_to_visual_type.get(block_type(main_block, type_by_index)) == BlockType.TABLE
        for main_block in closest_index_candidates
    ):
        # 表格 caption 位于两个表之间且距离接近时，优先归属后一个表。
        return max(closest_index_candidates, key=lambda x: x.index or 0)

    if child_type == BlockType.FOOTNOTE:
        # 视觉脚注位于两个主体之间且距离接近时，优先归属前一个主体。
        return min(closest_index_candidates, key=lambda x: x.index or 0)

    return min(
        closest_index_candidates,
        key=lambda main_block: (
            bbox_center_distance(child_block.bbox, main_block.bbox),
            main_block.index,
        ),
    )


def effective_visual_index_diff(
    child_block: Block,
    main_block: Block,
    ordered_blocks: list[Block],
    type_by_index: dict[int, str] | None = None,
) -> int:
    """按有效块序列计算视觉子块与主体距离，吸收的 image 子成员视为零成本。"""
    position_by_index = {block.index: position for position, block in enumerate(ordered_blocks)}
    child_pos = position_by_index[child_block.index]
    main_pos = position_by_index[main_block.index]
    start_pos = min(child_pos, main_pos)
    end_pos = max(child_pos, main_pos)
    skipped_child_count = 0
    child_type = block_type(child_block, type_by_index)

    for block in ordered_blocks[start_pos + 1 : end_pos]:
        if block_type(block, type_by_index) == child_type:
            skipped_child_count += 1

    return end_pos - start_pos - skipped_child_count


def is_visual_neighbor(
    child_block: Block,
    main_block: Block,
    ordered_blocks: list[Block],
    position_by_index: dict[int, int],
    type_by_index: dict[int, str] | None = None,
) -> bool:
    child_type = block_type(child_block, type_by_index)
    if child_type == BlockType.FOOTNOTE and child_block.index < main_block.index:
        return False

    if child_type == BlockType.CAPTION:
        allowed_between_types = {BlockType.CAPTION}
    else:
        allowed_between_types = set(GENERIC_CHILD_TYPES)

    child_pos = position_by_index[child_block.index]
    main_pos = position_by_index[main_block.index]
    start_pos = min(child_pos, main_pos) + 1
    end_pos = max(child_pos, main_pos)

    for pos in range(start_pos, end_pos):
        between_block = ordered_blocks[pos]
        if block_type(between_block, type_by_index) in allowed_between_types:
            continue
        if is_block_outside_visual_gap(between_block, child_block, main_block):
            continue
        return False

    return True


def is_block_outside_visual_gap(between_block: Block, child_block: Block, main_block: Block) -> bool:
    """判断阅读顺序夹在中间的块是否没有落入视觉父子块的垂直间隔。"""
    visual_gap = vertical_gap_between_blocks(child_block, main_block)
    if visual_gap is None:
        return False

    if is_bbox_overlapping_visual_relation_block(
        between_block.bbox,
        child_block.bbox,
        main_block.bbox,
    ):
        return False

    if not is_bbox_intersecting_vertical_gap(between_block.bbox, visual_gap):
        return True

    return False


def vertical_gap_between_blocks(first_block: Block, second_block: Block) -> tuple[float, float] | None:
    """计算两个块上下分离时的垂直间隔；发生纵向重叠时保持严格阻断。"""
    first_bbox = first_block.bbox
    second_bbox = second_block.bbox
    if first_bbox[3] <= second_bbox[1]:
        return first_bbox[3], second_bbox[1]
    if second_bbox[3] <= first_bbox[1]:
        return second_bbox[3], first_bbox[1]
    return None


def is_bbox_intersecting_vertical_gap(bbox: BBox, vertical_gap: tuple[float, float]) -> bool:
    """判断 bbox 是否与视觉父子块之间的垂直间隔相交。"""
    gap_top, gap_bottom = vertical_gap
    return bbox[1] < gap_bottom and bbox[3] > gap_top


def is_bbox_overlapping_visual_relation_block(bbox: BBox, child_bbox: BBox, main_bbox: BBox) -> bool:
    """判断 bbox 是否覆盖到父子块本身；覆盖时不能当作普通 index 噪声跳过。"""
    return are_bboxes_overlapping(bbox, child_bbox) or are_bboxes_overlapping(bbox, main_bbox)


def are_bboxes_overlapping(first_bbox: BBox, second_bbox: BBox) -> bool:
    """判断两个 bbox 是否存在二维相交。"""
    return not (
        first_bbox[2] <= second_bbox[0]
        or first_bbox[0] >= second_bbox[2]
        or first_bbox[3] <= second_bbox[1]
        or first_bbox[1] >= second_bbox[3]
    )


def block_type(block: Block, type_by_index: dict[int, str] | None = None) -> str:
    """读取块类型；pipeline 会传入改写前的原始类型映射。"""
    if type_by_index is not None:
        return type_by_index[block.index]
    return block.type


def child_kind_from_type(block_type: str) -> str:
    if block_type == BlockType.CAPTION:
        return "caption"
    return "footnote"


def child_visual_type(block_type: str) -> str:
    if block_type == BlockType.CHART_BODY:
        return BlockType.CHART
    return BlockType.IMAGE


def bbox_area(bbox: BBox) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def relative_bbox(child_bbox: BBox, parent_bbox: BBox) -> BBox:
    parent_x0, parent_y0, parent_x1, parent_y1 = parent_bbox
    parent_w = max(parent_x1 - parent_x0, 1)
    parent_h = max(parent_y1 - parent_y0, 1)
    return (
        clamp_and_round((child_bbox[0] - parent_x0) / parent_w),
        clamp_and_round((child_bbox[1] - parent_y0) / parent_h),
        clamp_and_round((child_bbox[2] - parent_x0) / parent_w),
        clamp_and_round((child_bbox[3] - parent_y0) / parent_h),
    )


def clamp_and_round(value: float) -> float:
    return round(min(max(value, 0.0), 1.0), 3)
