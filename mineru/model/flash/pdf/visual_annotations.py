# Copyright (c) Opendatalab. All rights reserved.

"""识别独立视觉块的强规则 caption/footnote，并构造局部阅读区域。"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal


from .._shared.xycut import sort_entries
from ....types import BBox

from .geometry import (
    _bbox_center_x,
    _bbox_center_y,
    _bbox_union_many,
    _coerce_bbox,
    _rotate_bbox_to_upright,
)


_VISUAL_BLOCK_TYPES = {"image", "table", "code"}
_CAPTION_MAX_GAP_IN_LINE_HEIGHTS = 2.5
_FOOTNOTE_MAX_GAP_IN_LINE_HEIGHTS = 6.0
_MIN_PROJECTION_OVERLAP = 0.8
_MIN_ANNOTATION_COVERAGE = 0.65
_COMPONENT_COVERAGE_GAIN = 0.15
_CROSS_LANE_CAPTION_ROW_TOP_TOLERANCE = 0.25
_CROSS_LANE_CAPTION_MIN_LINE_HEIGHT_RATIO = 0.85
_CROSS_LANE_CAPTION_MAX_LINE_HEIGHT_RATIO = 1.15
_TABLE_FOOTNOTE_CONTINUATION_MAX_GAP = 1.5
_TABLE_FOOTNOTE_FIRST_INDENT_MIN = 0.5
_TABLE_FOOTNOTE_FIRST_INDENT_MAX = 2.0
_TABLE_FOOTNOTE_FOLLOWING_INDENT_TOLERANCE = 0.5
_TABLE_FOOTNOTE_LANE_TOLERANCE = 0.5
_TABLE_FOOTNOTE_MIN_LINE_HEIGHT_RATIO = 0.75
_TABLE_FOOTNOTE_MAX_LINE_HEIGHT_RATIO = 1.25

_IDENTIFIER_PATTERN = (
    r"(?:"
    r"[A-Z]?\d+(?:\s*[-./–—]\s*[A-Z]?\d+)*[A-Z]?"
    r"|[IVXLCDM]+"
    r"|[零〇一二三四五六七八九十百千两]+"
    r")"
)
_ENGLISH_CAPTION_RE = re.compile(
    rf"^\s*(?:fig(?:ure)?|tab(?:le)?|alg(?:orithm)?|listing|chart|scheme)\.?(?:\s*)"
    rf"(?P<identifier>{_IDENTIFIER_PATTERN})(?P<tail>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_CHINESE_CAPTION_RE = re.compile(
    rf"^\s*(?:程序清单|图表|表格|算法|图|表)\s*"
    rf"(?P<identifier>{_IDENTIFIER_PATTERN})(?P<tail>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_FOOTNOTE_RE = re.compile(
    r"^\s*(?:source(?:s|\(s\))?|data\s+source|note(?:s|\(s\))?"
    r"|资料来源|数据来源|来源|注|备注)\s*[:：]\s*\S",
    re.IGNORECASE | re.DOTALL,
)
_ENGLISH_REFERENCE_TAIL_RE = re.compile(
    r"^(?:[,;]|and\b|or\b|&|shows?\b|illustrates?\b|presents?\b|depicts?\b|"
    r"demonstrates?\b|lists?\b|summari[sz]es?\b|reports?\b|compares?\b|"
    r"provides?\b|is\b|are\b|was\b|were\b|has\b|have\b|can\b)",
    re.IGNORECASE,
)
_CHINESE_REFERENCE_TAIL_RE = re.compile(
    r"^(?:[,;，；、]|和|及|与|为|是|展示|显示|给出|说明|列出|分别|可见|表明|描述|呈现|汇总|所示|中(?:的|为)?)",
)
_SUBFIGURE_REFERENCE_TAIL_RE = re.compile(
    r"^[（(][^）)]+[）)]\s*(?:[、,，]|和|及|与)",
)

_Direction = Literal["above", "below", "left", "right"]
_AnnotationKind = Literal["caption", "footnote"]
_TextBlockGroupMerger = Callable[
    [list[dict[str, Any]], list[int]],
    dict[str, Any],
]


@dataclass(frozen=True)
class _VisualParent:
    """记录单个视觉块或只用于关联的多图片组件。"""

    member_indices: tuple[int, ...]
    angle: int
    local_bbox: BBox


@dataclass(frozen=True)
class _AnnotationRelation:
    """记录注释与候选父块之间的方向和归一化几何代价。"""

    parent: _VisualParent
    direction: _Direction
    normalized_gap: float
    projection_overlap: float
    annotation_coverage: float
    center_offset: float


def _normalize_annotation_text(text: str) -> str:
    """统一全角字符和兼容罗马数字，保留原文仅供规则判断。"""

    return unicodedata.normalize("NFKC", text).strip()


def _caption_tail_is_reference(tail: str) -> bool:
    """排除编号后紧接叙述谓语、并列编号或正文连接词的引用句。"""

    stripped = tail.lstrip()
    if not stripped:
        return False
    if _SUBFIGURE_REFERENCE_TAIL_RE.match(stripped):
        return True
    if stripped[0] in ".:：-–—()[]（）":
        return False
    if _ENGLISH_REFERENCE_TAIL_RE.match(stripped) or _CHINESE_REFERENCE_TAIL_RE.match(stripped):
        return True
    return stripped[0].isascii() and stripped[0].isalpha() and stripped[0].islower()


def _is_strong_caption_text(text: str) -> bool:
    """判断文本是否以带编号的中英文强图表标题标记开头。"""

    normalized = _normalize_annotation_text(text)
    match = _ENGLISH_CAPTION_RE.match(normalized) or _CHINESE_CAPTION_RE.match(normalized)
    return match is not None and not _caption_tail_is_reference(match.group("tail"))


def _is_strong_footnote_text(text: str) -> bool:
    """判断文本是否以带冒号的来源或注释强标记开头。"""

    return bool(_FOOTNOTE_RE.match(_normalize_annotation_text(text)))


def _block_angle(block: dict[str, Any]) -> int:
    """将块角度规范到四个正交方向。"""

    return int(block.get("angle", 0) or 0) % 360


def _block_local_bbox(
    block: dict[str, Any],
    page_size: tuple[float, float],
    angle: int | None = None,
) -> BBox | None:
    """读取有效 bbox 并转换到指定文本方向的局部坐标。"""

    bbox = _coerce_bbox(block.get("bbox"))
    if bbox is None:
        return None
    return _rotate_bbox_to_upright(
        bbox,
        page_size,
        _block_angle(block) if angle is None else angle,
    )


def _block_median_line_height(
    block: dict[str, Any],
    page_size: tuple[float, float],
) -> float:
    """优先使用原生行高，缺失时按局部块高和行框数量保守估计。"""

    heights = [
        float(height)
        for height in block.get("_line_heights", [])
        if isinstance(height, (int, float)) and float(height) > 0
    ]
    if heights:
        return statistics.median(heights)
    local_bbox = _block_local_bbox(block, page_size)
    if local_bbox is None:
        return 1.0
    local_rows = block.get("_local_line_bboxes")
    row_count = len(local_rows) if isinstance(local_rows, list) and local_rows else 1
    return max(1.0, (local_bbox[3] - local_bbox[1]) / row_count)


def _collect_annotation_candidates(
    blocks: list[dict[str, Any]],
) -> dict[int, _AnnotationKind]:
    """从独立 text 及预分类 caption/footnote 中收集视觉绑定候选。"""

    output: dict[int, _AnnotationKind] = {}
    for index, block in enumerate(blocks):
        block_type = block.get("type")
        if block_type in {"caption", "footnote"}:
            output[index] = block_type
            continue
        if block_type != "text" or not isinstance(block.get("content"), str):
            continue
        content = str(block["content"])
        if _is_strong_caption_text(content):
            output[index] = "caption"
        elif _is_strong_footnote_text(content):
            output[index] = "footnote"
    return output


def _coerce_lane_interval(value: object) -> tuple[float, float] | None:
    """读取内部栏带区间，拒绝缺失、逆序或非数值元数据。"""

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return start, end


def _block_first_local_row_bbox(
    block: dict[str, Any],
    page_size: tuple[float, float],
    angle: int,
) -> BBox | None:
    """返回块在指定方向局部坐标中的首个文本行框。"""

    rows = block.get("_local_line_bboxes")
    local_rows = [bbox for row in rows if (bbox := _coerce_bbox(row)) is not None] if isinstance(rows, list) else []
    if local_rows:
        return min(local_rows, key=lambda bbox: (bbox[1], bbox[0]))
    return _block_local_bbox(block, page_size, angle)


def _blocks_share_table_footnote_lane(
    anchor: dict[str, Any],
    candidate: dict[str, Any],
    line_height: float,
) -> bool:
    """要求表注锚点与续块来自同一内部栏带和 span 层级。"""

    if anchor.get("_lane_is_span") != candidate.get("_lane_is_span"):
        return False
    anchor_interval = _coerce_lane_interval(anchor.get("_lane_interval"))
    candidate_interval = _coerce_lane_interval(candidate.get("_lane_interval"))
    if anchor_interval is None or candidate_interval is None:
        return False
    tolerance = _TABLE_FOOTNOTE_LANE_TOLERANCE * line_height
    return (
        abs(anchor_interval[0] - candidate_interval[0]) <= tolerance
        and abs(anchor_interval[1] - candidate_interval[1]) <= tolerance
    )


def _block_overlaps_table_footnote_lane(
    block: dict[str, Any],
    lane_interval: tuple[float, float],
    page_size: tuple[float, float],
    angle: int,
) -> bool:
    """判断任意块是否横穿表注栏带，用于保证收集过程不跨越障碍。"""

    local_bbox = _block_local_bbox(block, page_size, angle)
    if local_bbox is None:
        return False
    overlap = max(
        0.0,
        min(local_bbox[2], lane_interval[1]) - max(local_bbox[0], lane_interval[0]),
    )
    shorter_width = max(
        0.1,
        min(
            local_bbox[2] - local_bbox[0],
            lane_interval[1] - lane_interval[0],
        ),
    )
    return overlap / shorter_width >= 0.5


def _table_footnote_fonts_are_compatible(
    previous: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    """仅在两侧都有可靠字体信息且完全不相交时拒绝续块。"""

    previous_fonts = previous.get("_font_signatures")
    candidate_fonts = candidate.get("_font_signatures")
    return not (
        isinstance(previous_fonts, set)
        and isinstance(candidate_fonts, set)
        and previous_fonts
        and candidate_fonts
        and previous_fonts.isdisjoint(candidate_fonts)
    )


def _table_footnote_candidate_fits_parent(
    candidate_bbox: BBox,
    parent_bbox: BBox,
    line_height: float,
) -> bool:
    """要求续块位于父表格下方并保持既有横向投影覆盖标准。"""

    if _bbox_center_y(candidate_bbox) < _bbox_center_y(parent_bbox):
        return False
    projection_overlap, annotation_coverage, _center_offset = _axis_overlap_metrics(
        candidate_bbox,
        parent_bbox,
        axis="x",
        float_margin=line_height,
    )
    return projection_overlap >= _MIN_PROJECTION_OVERLAP and annotation_coverage >= _MIN_ANNOTATION_COVERAGE


def _table_footnote_continuation_is_compatible(
    anchor: dict[str, Any],
    previous: dict[str, Any],
    candidate: dict[str, Any],
    parent_bbox: BBox,
    page_size: tuple[float, float],
    angle: int,
    continuation_left: float | None,
) -> tuple[bool, float | None]:
    """按栏带、字体、行高、净空和悬挂缩进确认单个表注续块。"""

    previous_bbox = _block_local_bbox(previous, page_size, angle)
    candidate_bbox = _block_local_bbox(candidate, page_size, angle)
    anchor_first_row = _block_first_local_row_bbox(anchor, page_size, angle)
    candidate_first_row = _block_first_local_row_bbox(candidate, page_size, angle)
    if previous_bbox is None or candidate_bbox is None or anchor_first_row is None or candidate_first_row is None:
        return False, continuation_left

    previous_height = _block_median_line_height(previous, page_size)
    candidate_height = _block_median_line_height(candidate, page_size)
    if previous_height <= 0 or candidate_height <= 0:
        return False, continuation_left
    height_ratio = candidate_height / previous_height
    pair_height = statistics.median((previous_height, candidate_height))
    vertical_gap = max(0.0, candidate_bbox[1] - previous_bbox[3])
    if (
        not _TABLE_FOOTNOTE_MIN_LINE_HEIGHT_RATIO <= height_ratio <= _TABLE_FOOTNOTE_MAX_LINE_HEIGHT_RATIO
        or vertical_gap > _TABLE_FOOTNOTE_CONTINUATION_MAX_GAP * pair_height
        or not _blocks_share_table_footnote_lane(anchor, candidate, pair_height)
        or not _table_footnote_fonts_are_compatible(previous, candidate)
        or not _table_footnote_candidate_fits_parent(
            candidate_bbox,
            parent_bbox,
            pair_height,
        )
    ):
        return False, continuation_left

    candidate_left = candidate_first_row[0]
    if continuation_left is None:
        indent = candidate_left - anchor_first_row[0]
        if not (_TABLE_FOOTNOTE_FIRST_INDENT_MIN * pair_height <= indent <= _TABLE_FOOTNOTE_FIRST_INDENT_MAX * pair_height):
            return False, continuation_left
        return True, candidate_left
    if abs(candidate_left - continuation_left) > _TABLE_FOOTNOTE_FOLLOWING_INDENT_TOLERANCE * pair_height:
        return False, continuation_left
    return True, continuation_left


def _collect_table_footnote_continuation_indices(
    anchor_index: int,
    relation: _AnnotationRelation,
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
    annotation_indices: set[int],
    consumed_indices: set[int],
) -> list[int]:
    """从表注锚点向下连续收集同栏悬挂缩进文本，不跨越首个障碍。"""

    anchor = blocks[anchor_index]
    angle = relation.parent.angle
    anchor_bbox = _block_local_bbox(anchor, page_size, angle)
    lane_interval = _coerce_lane_interval(anchor.get("_lane_interval"))
    if anchor_bbox is None or lane_interval is None:
        return []

    ordered_indices = sorted(
        (
            index
            for index, block in enumerate(blocks)
            if index != anchor_index
            and index not in relation.parent.member_indices
            and index not in consumed_indices
            and _block_angle(block) == angle
            and (local_bbox := _block_local_bbox(block, page_size, angle)) is not None
            and _bbox_center_y(local_bbox) > _bbox_center_y(anchor_bbox)
            and _block_overlaps_table_footnote_lane(
                block,
                lane_interval,
                page_size,
                angle,
            )
        ),
        key=lambda index: (
            _block_local_bbox(blocks[index], page_size, angle)[1],
            _block_local_bbox(blocks[index], page_size, angle)[0],
            index,
        ),
    )

    output: list[int] = []
    previous = anchor
    continuation_left: float | None = None
    for index in ordered_indices:
        candidate = blocks[index]
        if index in annotation_indices or candidate.get("type") != "text":
            break
        compatible, continuation_left = _table_footnote_continuation_is_compatible(
            anchor,
            previous,
            candidate,
            relation.parent.local_bbox,
            page_size,
            angle,
            continuation_left,
        )
        if not compatible:
            break
        output.append(index)
        previous = candidate
    return output


def _merge_table_footnote_continuations(
    blocks: list[dict[str, Any]],
    candidates: dict[int, _AnnotationKind],
    assignments: dict[int, _AnnotationRelation],
    page_size: tuple[float, float],
    merge_text_block_group: _TextBlockGroupMerger | None,
) -> set[int]:
    """合并已绑定单表格的强标记脚注及其页内续块，并返回被消费索引。"""

    if merge_text_block_group is None:
        return set()
    annotation_indices = set(candidates)
    consumed_indices: set[int] = set()
    for anchor_index in sorted(
        assignments,
        key=lambda index: _annotation_local_sort_key(index, blocks, page_size),
    ):
        relation = assignments[anchor_index]
        parent_indices = relation.parent.member_indices
        anchor = blocks[anchor_index]
        if (
            candidates.get(anchor_index) != "footnote"
            or anchor.get("_table_annotation_complete") is True
            or anchor.get("type") != "text"
            or not isinstance(anchor.get("content"), str)
            or not _is_strong_footnote_text(str(anchor["content"]))
            or relation.direction != "below"
            or len(parent_indices) != 1
            or blocks[parent_indices[0]].get("type") != "table"
        ):
            continue
        continuation_indices = _collect_table_footnote_continuation_indices(
            anchor_index,
            relation,
            blocks,
            page_size,
            annotation_indices,
            consumed_indices,
        )
        if not continuation_indices:
            continue
        blocks[anchor_index] = merge_text_block_group(
            blocks,
            [anchor_index, *continuation_indices],
        )
        consumed_indices.update(continuation_indices)
    return consumed_indices


def _axis_overlap_metrics(
    annotation_bbox: BBox,
    parent_bbox: BBox,
    *,
    axis: Literal["x", "y"],
    float_margin: float,
) -> tuple[float, float, float]:
    """在正交轴允许一行高浮动后计算较短投影、注释覆盖和中心偏移。"""

    if axis == "x":
        annotation_start, annotation_end = annotation_bbox[0], annotation_bbox[2]
        parent_start, parent_end = parent_bbox[0], parent_bbox[2]
    else:
        annotation_start, annotation_end = annotation_bbox[1], annotation_bbox[3]
        parent_start, parent_end = parent_bbox[1], parent_bbox[3]
    expanded_parent_start = parent_start - float_margin
    expanded_parent_end = parent_end + float_margin
    overlap = max(
        0.0,
        min(annotation_end, expanded_parent_end)
        - max(annotation_start, expanded_parent_start),
    )
    annotation_length = max(0.1, annotation_end - annotation_start)
    parent_length = max(0.1, expanded_parent_end - expanded_parent_start)
    projection_overlap = overlap / min(annotation_length, parent_length)
    annotation_coverage = overlap / annotation_length
    center_offset = abs(
        (annotation_start + annotation_end) / 2.0
        - (parent_start + parent_end) / 2.0
    ) / max(annotation_length, parent_end - parent_start, 0.1)
    return projection_overlap, annotation_coverage, center_offset


def _direction_relation(
    parent: _VisualParent,
    annotation_bbox: BBox,
    line_height: float,
    direction: _Direction,
    max_gap_in_line_heights: float,
) -> _AnnotationRelation | None:
    """按一个方向检查边缘距离、深入量和正交投影约束。"""

    parent_bbox = parent.local_bbox
    if direction == "above":
        if _bbox_center_y(annotation_bbox) > _bbox_center_y(parent_bbox):
            return None
        signed_gap = parent_bbox[1] - annotation_bbox[3]
        projection_axis: Literal["x", "y"] = "x"
    elif direction == "below":
        if _bbox_center_y(annotation_bbox) < _bbox_center_y(parent_bbox):
            return None
        signed_gap = annotation_bbox[1] - parent_bbox[3]
        projection_axis = "x"
    elif direction == "left":
        if _bbox_center_x(annotation_bbox) > _bbox_center_x(parent_bbox):
            return None
        signed_gap = parent_bbox[0] - annotation_bbox[2]
        projection_axis = "y"
    else:
        if _bbox_center_x(annotation_bbox) < _bbox_center_x(parent_bbox):
            return None
        signed_gap = annotation_bbox[0] - parent_bbox[2]
        projection_axis = "y"
    gap = max(0.0, signed_gap)
    penetration = max(0.0, -signed_gap)
    if gap > max_gap_in_line_heights * line_height or penetration > line_height:
        return None
    projection_overlap, annotation_coverage, center_offset = _axis_overlap_metrics(
        annotation_bbox,
        parent_bbox,
        axis=projection_axis,
        float_margin=line_height,
    )
    if (
        projection_overlap < _MIN_PROJECTION_OVERLAP
        or annotation_coverage < _MIN_ANNOTATION_COVERAGE
    ):
        return None
    return _AnnotationRelation(
        parent=parent,
        direction=direction,
        normalized_gap=gap / line_height,
        projection_overlap=projection_overlap,
        annotation_coverage=annotation_coverage,
        center_offset=center_offset,
    )


def _best_parent_relation(
    parent: _VisualParent,
    annotation_bbox: BBox,
    line_height: float,
    kind: _AnnotationKind,
) -> _AnnotationRelation | None:
    """为单个父块选择几何代价最小的合法注释方向。"""

    directions: tuple[_Direction, ...]
    if kind == "footnote":
        directions = ("below",)
        max_gap = _FOOTNOTE_MAX_GAP_IN_LINE_HEIGHTS
    else:
        directions = ("above", "below", "left", "right")
        max_gap = _CAPTION_MAX_GAP_IN_LINE_HEIGHTS
    relations = [
        relation
        for direction in directions
        if (
            relation := _direction_relation(
                parent,
                annotation_bbox,
                line_height,
                direction,
                max_gap,
            )
        )
        is not None
    ]
    if not relations:
        return None
    return min(
        relations,
        key=lambda relation: (
            relation.normalized_gap,
            -relation.annotation_coverage,
            -relation.projection_overlap,
            relation.center_offset,
        ),
    )


def _image_blocks_form_component(
    first_bbox: BBox,
    second_bbox: BBox,
    line_height: float,
) -> bool:
    """判断两张图片是否以小净空和足够正交投影组成相邻面板。"""

    horizontal_gap = max(first_bbox[0] - second_bbox[2], second_bbox[0] - first_bbox[2], 0.0)
    vertical_gap = max(first_bbox[1] - second_bbox[3], second_bbox[1] - first_bbox[3], 0.0)
    x_overlap = max(0.0, min(first_bbox[2], second_bbox[2]) - max(first_bbox[0], second_bbox[0]))
    y_overlap = max(0.0, min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1]))
    min_width = max(0.1, min(first_bbox[2] - first_bbox[0], second_bbox[2] - second_bbox[0]))
    min_height = max(0.1, min(first_bbox[3] - first_bbox[1], second_bbox[3] - second_bbox[1]))
    return (
        horizontal_gap <= 2.0 * line_height and y_overlap / min_height >= 0.5
    ) or (
        vertical_gap <= 2.0 * line_height and x_overlap / min_width >= 0.5
    )


def _build_visual_parents(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
    component_line_height: float,
) -> list[_VisualParent]:
    """构造单视觉块父候选，并为相邻图片补充只用于关联的连通组件。"""

    parents: list[_VisualParent] = []
    image_indices_by_angle: dict[int, list[int]] = {}
    local_bboxes: dict[int, BBox] = {}
    for index, block in enumerate(blocks):
        if block.get("type") not in _VISUAL_BLOCK_TYPES:
            continue
        angle = _block_angle(block)
        local_bbox = _block_local_bbox(block, page_size, angle)
        if local_bbox is None:
            continue
        local_bboxes[index] = local_bbox
        parents.append(_VisualParent((index,), angle, local_bbox))
        if block.get("type") == "image":
            image_indices_by_angle.setdefault(angle, []).append(index)

    for angle, image_indices in image_indices_by_angle.items():
        remaining = set(image_indices)
        while remaining:
            seed = min(remaining)
            component = {seed}
            remaining.remove(seed)
            changed = True
            while changed:
                changed = False
                for candidate in list(remaining):
                    if any(
                        _image_blocks_form_component(
                            local_bboxes[member],
                            local_bboxes[candidate],
                            component_line_height,
                        )
                        for member in component
                    ):
                        component.add(candidate)
                        remaining.remove(candidate)
                        changed = True
            if len(component) < 2:
                continue
            member_indices = tuple(sorted(component))
            parents.append(
                _VisualParent(
                    member_indices,
                    angle,
                    _bbox_union_many([local_bboxes[index] for index in member_indices]),
                )
            )
    return parents


def _relation_has_intervening_block(
    relation: _AnnotationRelation,
    annotation_index: int,
    annotation_bbox: BBox,
    line_height: float,
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
    annotation_indices: set[int],
) -> bool:
    """检查注释与父块净空走廊内是否横隔正文或另一个视觉块。"""

    if relation.normalized_gap <= 0:
        return False
    parent_bbox = relation.parent.local_bbox
    if relation.direction in {"above", "below"}:
        orth_start = max(annotation_bbox[0], parent_bbox[0]) - 0.25 * line_height
        orth_end = min(annotation_bbox[2], parent_bbox[2]) + 0.25 * line_height
        if relation.direction == "above":
            corridor = (orth_start, annotation_bbox[3], orth_end, parent_bbox[1])
        else:
            corridor = (orth_start, parent_bbox[3], orth_end, annotation_bbox[1])
    else:
        orth_start = max(annotation_bbox[1], parent_bbox[1]) - 0.25 * line_height
        orth_end = min(annotation_bbox[3], parent_bbox[3]) + 0.25 * line_height
        if relation.direction == "left":
            corridor = (annotation_bbox[2], orth_start, parent_bbox[0], orth_end)
        else:
            corridor = (parent_bbox[2], orth_start, annotation_bbox[0], orth_end)
    if corridor[2] <= corridor[0] or corridor[3] <= corridor[1]:
        return False
    ignored_indices = {
        annotation_index,
        *relation.parent.member_indices,
        *annotation_indices,
    }
    for index, block in enumerate(blocks):
        if index in ignored_indices or _block_angle(block) != relation.parent.angle:
            continue
        bbox = _block_local_bbox(block, page_size, relation.parent.angle)
        if bbox is None:
            continue
        if (
            min(bbox[2], corridor[2]) > max(bbox[0], corridor[0])
            and min(bbox[3], corridor[3]) > max(bbox[1], corridor[1])
        ):
            return True
    return False


def _component_relation_is_materially_better(
    relation: _AnnotationRelation,
    single_relations: list[_AnnotationRelation],
) -> bool:
    """仅当图片并集明显提高注释覆盖时允许组件替代单图父块。"""

    if len(relation.parent.member_indices) < 2:
        return True
    member_indices = set(relation.parent.member_indices)
    comparable = [
        single
        for single in single_relations
        if single.direction == relation.direction
        and single.parent.member_indices[0] in member_indices
    ]
    if not comparable:
        return True
    best_single_coverage = max(single.annotation_coverage for single in comparable)
    return relation.annotation_coverage >= best_single_coverage + _COMPONENT_COVERAGE_GAIN


def _choose_annotation_relation(
    annotation_index: int,
    kind: _AnnotationKind,
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
    parents: list[_VisualParent],
    annotation_indices: set[int],
) -> _AnnotationRelation | None:
    """先过滤空间、组件收益和阻挡关系，再按距离、覆盖和居中选父块。"""

    annotation = blocks[annotation_index]
    angle = _block_angle(annotation)
    annotation_bbox = _block_local_bbox(annotation, page_size, angle)
    if annotation_bbox is None:
        return None
    line_height = _block_median_line_height(annotation, page_size)
    relations = [
        relation
        for parent in parents
        if parent.angle == angle
        and (
            relation := _best_parent_relation(
                parent,
                annotation_bbox,
                line_height,
                kind,
            )
        )
        is not None
    ]
    single_relations = [
        relation for relation in relations if len(relation.parent.member_indices) == 1
    ]
    relations = [
        relation
        for relation in relations
        if _component_relation_is_materially_better(relation, single_relations)
        and not _relation_has_intervening_block(
            relation,
            annotation_index,
            annotation_bbox,
            line_height,
            blocks,
            page_size,
            annotation_indices,
        )
    ]
    if not relations:
        return None
    return min(
        relations,
        key=lambda relation: (
            relation.normalized_gap,
            -relation.annotation_coverage,
            -relation.projection_overlap,
            relation.center_offset,
            len(relation.parent.member_indices),
            relation.parent.member_indices,
        ),
    )


def _caption_blocks_use_distinct_regular_lanes(
    anchor: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    """确认两个标题块分别属于互不重叠的普通栏带。"""

    if (
        anchor.get("_lane_is_span") is not False
        or candidate.get("_lane_is_span") is not False
    ):
        return False
    anchor_interval = _coerce_lane_interval(anchor.get("_lane_interval"))
    candidate_interval = _coerce_lane_interval(candidate.get("_lane_interval"))
    if anchor_interval is None or candidate_interval is None:
        return False
    return (
        anchor_interval[1] <= candidate_interval[0]
        or candidate_interval[1] <= anchor_interval[0]
    )


def _caption_blocks_have_compatible_typography(
    anchor: dict[str, Any],
    candidate: dict[str, Any],
    page_size: tuple[float, float],
) -> bool:
    """用行高和字体交集确认跨栏标题块来自同一排版层级。"""

    anchor_height = _block_median_line_height(anchor, page_size)
    candidate_height = _block_median_line_height(candidate, page_size)
    if anchor_height <= 0 or candidate_height <= 0:
        return False
    height_ratio = candidate_height / anchor_height
    if not (
        _CROSS_LANE_CAPTION_MIN_LINE_HEIGHT_RATIO
        <= height_ratio
        <= _CROSS_LANE_CAPTION_MAX_LINE_HEIGHT_RATIO
    ):
        return False

    anchor_fonts = anchor.get("_font_signatures")
    candidate_fonts = candidate.get("_font_signatures")
    if not (
        isinstance(anchor_fonts, (set, frozenset))
        and isinstance(candidate_fonts, (set, frozenset))
        and anchor_fonts
        and candidate_fonts
    ):
        return False
    return not anchor_fonts.isdisjoint(candidate_fonts)


def _cross_lane_caption_companion_relation(
    anchor_index: int,
    candidate_index: int,
    anchor_relation: _AnnotationRelation,
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
    parents: list[_VisualParent],
    annotation_indices: set[int],
) -> _AnnotationRelation | None:
    """仅凭空间、栏带和排版信息确认一个跨栏标题同伴。"""

    if anchor_relation.direction not in {"above", "below"}:
        return None
    anchor = blocks[anchor_index]
    candidate = blocks[candidate_index]
    if (
        candidate.get("type") != "text"
        or _block_angle(candidate) != anchor_relation.parent.angle
    ):
        return None
    if not _caption_blocks_use_distinct_regular_lanes(anchor, candidate):
        return None
    if not _caption_blocks_have_compatible_typography(
        anchor,
        candidate,
        page_size,
    ):
        return None

    angle = anchor_relation.parent.angle
    anchor_first_row = _block_first_local_row_bbox(anchor, page_size, angle)
    candidate_first_row = _block_first_local_row_bbox(
        candidate,
        page_size,
        angle,
    )
    if anchor_first_row is None or candidate_first_row is None:
        return None
    pair_height = statistics.median(
        (
            _block_median_line_height(anchor, page_size),
            _block_median_line_height(candidate, page_size),
        )
    )
    if (
        abs(anchor_first_row[1] - candidate_first_row[1])
        > _CROSS_LANE_CAPTION_ROW_TOP_TOLERANCE * pair_height
    ):
        return None

    relation = _choose_annotation_relation(
        candidate_index,
        "caption",
        blocks,
        page_size,
        parents,
        annotation_indices | {candidate_index},
    )
    if (
        relation is None
        or relation.parent != anchor_relation.parent
        or relation.direction != anchor_relation.direction
    ):
        return None
    return relation


def _expand_cross_lane_caption_assignments(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
    parents: list[_VisualParent],
    candidates: dict[int, _AnnotationKind],
    assignments: dict[int, _AnnotationRelation],
) -> dict[int, _AnnotationRelation]:
    """从原始已绑定标题出发，保守补标唯一的跨栏空间同伴。"""

    annotation_indices = set(candidates)
    proposals: dict[int, list[_AnnotationRelation]] = {}
    for anchor_index, anchor_relation in assignments.items():
        if candidates.get(anchor_index) != "caption":
            continue
        matches = [
            (index, relation)
            for index in range(len(blocks))
            if index not in annotation_indices
            and (
                relation := _cross_lane_caption_companion_relation(
                    anchor_index,
                    index,
                    anchor_relation,
                    blocks,
                    page_size,
                    parents,
                    annotation_indices,
                )
            )
            is not None
        ]
        if len(matches) != 1:
            continue
        index, relation = matches[0]
        proposals.setdefault(index, []).append(relation)

    output: dict[int, _AnnotationRelation] = {}
    for index, relations in proposals.items():
        # 同伴同时被多个锚点认领时归属不唯一，保守地维持正文分类。
        if len(relations) == 1:
            output[index] = relations[0]
    return output


def _merge_parent_assignment_groups(
    assignments: dict[int, _AnnotationRelation],
) -> list[tuple[set[int], dict[int, _AnnotationRelation]]]:
    """合并共享视觉成员的父候选，防止组件与单图重复展开同一主体。"""

    groups: list[tuple[set[int], dict[int, _AnnotationRelation]]] = []
    for annotation_index, relation in assignments.items():
        parent_indices = set(relation.parent.member_indices)
        overlapping = [
            index
            for index, (members, _relations) in enumerate(groups)
            if members & parent_indices
        ]
        merged_members = set(parent_indices)
        merged_relations = {annotation_index: relation}
        for group_index in reversed(overlapping):
            members, relations = groups.pop(group_index)
            merged_members.update(members)
            merged_relations.update(relations)
        groups.append((merged_members, merged_relations))
    return groups


def _sort_visual_body_members(
    member_indices: set[int],
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """在共同局部方向内用 XYCut++ 排列区域中的视觉主体成员。"""

    if not member_indices:
        return []
    angle = _block_angle(blocks[min(member_indices)])
    proxies: list[dict[str, Any]] = []
    for index in sorted(member_indices):
        local_bbox = _block_local_bbox(blocks[index], page_size, angle)
        if local_bbox is None:
            continue
        proxies.append({"bbox": local_bbox, "_block": blocks[index]})
    return [proxy["_block"] for proxy in sort_entries(proxies)]


def _annotation_local_sort_key(
    index: int,
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> tuple[float, float, int]:
    """按注释自身方向的局部上、左坐标提供稳定排序键。"""

    block = blocks[index]
    local_bbox = _block_local_bbox(block, page_size)
    if local_bbox is None:
        return (float("inf"), float("inf"), index)
    return (local_bbox[1], local_bbox[0], index)


def _sort_annotation_indices_by_visual_rows(
    indices: list[int],
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
    relations: dict[int, _AnnotationRelation],
) -> list[int]:
    """先聚合首行对齐的标题视觉行，再按行内左到右稳定排序。"""

    positioned: list[tuple[int, BBox, float]] = []
    fallback: list[int] = []
    for index in indices:
        block = blocks[index]
        angle = _block_angle(block)
        first_row = _block_first_local_row_bbox(block, page_size, angle)
        if first_row is None:
            fallback.append(index)
            continue
        positioned.append(
            (
                index,
                first_row,
                _block_median_line_height(block, page_size),
            )
        )
    positioned.sort(key=lambda item: (item[1][1], item[1][0], item[0]))

    rows: list[list[tuple[int, BBox, float]]] = []
    for item in positioned:
        if rows:
            current_row = rows[-1]
            item_relation = relations.get(item[0])
            shares_cross_lane_row = (
                item_relation is not None
                and item_relation.direction in {"above", "below"}
                and all(
                    (
                        member_relation := relations.get(member[0])
                    ) is not None
                    and member_relation.direction == item_relation.direction
                    and _caption_blocks_use_distinct_regular_lanes(
                        blocks[member[0]],
                        blocks[item[0]],
                    )
                    for member in current_row
                )
            )
            reference_top = statistics.median(
                member[1][1]
                for member in current_row
            )
            reference_height = statistics.median(
                member[2]
                for member in current_row
            )
            pair_height = statistics.median((reference_height, item[2]))
            if (
                shares_cross_lane_row
                and abs(item[1][1] - reference_top)
                <= _CROSS_LANE_CAPTION_ROW_TOP_TOLERANCE * pair_height
            ):
                current_row.append(item)
                continue
        rows.append([item])

    output: list[int] = []
    for row in rows:
        row.sort(key=lambda item: (item[1][0], item[1][1], item[0]))
        output.extend(item[0] for item in row)
    output.extend(
        sorted(
            fallback,
            key=lambda index: _annotation_local_sort_key(
                index,
                blocks,
                page_size,
            ),
        )
    )
    return output


def _build_visual_annotation_regions(
    assignments: dict[int, _AnnotationRelation],
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[list[dict[str, Any]]]:
    """按前置标题、主体、后置标题、脚注顺序展开每个虚拟小区域。"""

    regions: list[list[dict[str, Any]]] = []
    for parent_indices, relations in _merge_parent_assignment_groups(assignments):
        leading = [
            index
            for index, relation in relations.items()
            if blocks[index].get("type") == "caption"
            and relation.direction in {"above", "left"}
        ]
        trailing = [
            index
            for index, relation in relations.items()
            if blocks[index].get("type") == "caption"
            and relation.direction in {"below", "right"}
        ]
        footnotes = [
            index
            for index in relations
            if blocks[index].get("type") == "footnote"
        ]
        regions.append(
            [
                *(
                    blocks[index]
                    for index in _sort_annotation_indices_by_visual_rows(
                        leading,
                        blocks,
                        page_size,
                        relations,
                    )
                ),
                *_sort_visual_body_members(parent_indices, blocks, page_size),
                *(
                    blocks[index]
                    for index in _sort_annotation_indices_by_visual_rows(
                        trailing,
                        blocks,
                        page_size,
                        relations,
                    )
                ),
                *(
                    blocks[index]
                    for index in sorted(
                        footnotes,
                        key=lambda value: _annotation_local_sort_key(
                            value,
                            blocks,
                            page_size,
                        ),
                    )
                ),
            ]
        )
    return regions


def _classify_and_bind_visual_annotations(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
    *,
    merge_text_block_group: _TextBlockGroupMerger | None = None,
) -> list[list[dict[str, Any]]]:
    """重分类强规则独立注释，并返回供全局 XYCut++ 使用的有序视觉区域。"""

    candidates = _collect_annotation_candidates(blocks)
    if not candidates:
        return []
    line_heights = [
        _block_median_line_height(blocks[index], page_size)
        for index in candidates
    ]
    component_line_height = statistics.median(line_heights) if line_heights else 8.0
    parents = _build_visual_parents(blocks, page_size, component_line_height)
    if not parents:
        return []
    annotation_indices = set(candidates)
    assignments = {
        index: relation
        for index, kind in candidates.items()
        if (
            relation := _choose_annotation_relation(
                index,
                kind,
                blocks,
                page_size,
                parents,
                annotation_indices,
            )
        )
        is not None
    }
    cross_lane_assignments = _expand_cross_lane_caption_assignments(
        blocks,
        page_size,
        parents,
        candidates,
        assignments,
    )
    candidates.update(dict.fromkeys(cross_lane_assignments, "caption"))
    assignments.update(cross_lane_assignments)
    consumed_indices = _merge_table_footnote_continuations(
        blocks,
        candidates,
        assignments,
        page_size,
        merge_text_block_group,
    )
    for index, relation in assignments.items():
        blocks[index]["type"] = candidates[index]
        blocks[index]["_visual_annotation_direction"] = relation.direction
    regions = _build_visual_annotation_regions(assignments, blocks, page_size)
    if consumed_indices:
        blocks[:] = [
            block
            for index, block in enumerate(blocks)
            if index not in consumed_indices
        ]
    return regions
