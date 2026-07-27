# Copyright (c) Opendatalab. All rights reserved.

"""分类页眉、页脚、页码、侧栏和页脚注。"""

from __future__ import annotations

import re
import statistics
import unicodedata
from difflib import SequenceMatcher
from typing import Literal


from mineru.types import BBox

from .models import (
    _AxisLine,
    _LineItem,
    _LocalAxisLine,
    _MarginalCandidate,
    _PreparedPage,
    _TextLane,
)
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_intersects,
    _clip_bbox,
    _coerce_bbox,
    _expand_bbox,
    _horizontal_bbox_gap,
    _rotate_bbox_to_upright,
    _transform_axis_lines,
)
from .line_layout import (
    _effective_text_row_gap,
    _infer_text_lanes,
    _line_effective_height,
)
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s*)?[\-\u2013\u2014\u00b7\u2022]*\s*(?:\u7b2c\s*)?"
    r"(?P<value>\d{1,4}|[ivxlcdm]+|[\u3007\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u4e24]+)"
    r"(?:\s*(?:/|of|\u5171)\s*(?:\d{1,4}|[ivxlcdm]+|[\u3007\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u4e24]+))?"
    r"\s*(?:\u9875)?\s*[\-\u2013\u2014\u00b7\u2022]*\s*$",
    re.IGNORECASE,
)


def _classify_page_auxiliary_text(prepared: _PreparedPage) -> None:
    """在容器认领后仅按空间关系标注侧栏文字和页脚注。"""

    _classify_aside_text(prepared.remaining_lines, prepared.page_size)
    prepared.page_footnote_groups = _classify_page_footnotes(
        prepared.remaining_lines,
        prepared.table_bboxes,
        prepared.drawing_lines,
        prepared.page_size,
    )


def _classify_aside_text(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> None:
    """在横排正文占绝对多数时，以边缘带和物理尺寸识别垂直侧栏。"""

    available = [line for line in lines if line.semantic_type is None]
    upright_lines = [line for line in available if line.angle == 0]
    if len(upright_lines) < 4:
        return

    support_by_angle = _geometric_text_support_by_angle(available, page_size)
    total_support = sum(support_by_angle.values())
    if total_support <= 0 or support_by_angle.get(0, 0.0) / total_support < 0.8:
        return

    page_width, page_height = page_size
    if page_width <= 0 or page_height <= 0:
        return
    # 侧栏必须完整位于 12% 边缘带，且兼具不超过 8% 的窄宽和至少 15% 的物理高度。
    aside_source_indices = {
        line.source_index
        for line in available
        if line.angle in {90, 270}
        and line.bbox[2] - line.bbox[0] <= 0.08 * page_width
        and line.bbox[3] - line.bbox[1] >= 0.15 * page_height
        and (
            line.bbox[2] <= 0.12 * page_width
            or line.bbox[0] >= 0.88 * page_width
        )
    }
    for line in available:
        if line.source_index in aside_source_indices:
            line.semantic_type = "aside_text"


def _geometric_text_support_by_angle(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> dict[int, float]:
    """按局部行宽乘有效行高累计各文字方向的纯几何支持度。"""

    support_by_angle: dict[int, float] = {}
    for line in lines:
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
        local_width = max(0.1, local_bbox[2] - local_bbox[0])
        support_by_angle[line.angle] = support_by_angle.get(line.angle, 0.0) + (
            local_width * _line_effective_height(line, local_bbox)
        )
    return support_by_angle


def _classify_page_footnotes(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    drawing_lines: list[_AxisLine],
    page_size: tuple[float, float],
) -> list[set[int]]:
    """识别主方向页脚注，并按触发分隔线返回来源编号分组。"""

    available = [line for line in lines if line.semantic_type is None]
    if not available or not drawing_lines:
        return []
    support_by_angle = _geometric_text_support_by_angle(available, page_size)
    if not support_by_angle:
        return []
    dominant_angle = max(
        sorted(support_by_angle),
        key=lambda angle: support_by_angle[angle],
    )
    line_geometry = [
        (line, _rotate_bbox_to_upright(line.bbox, page_size, dominant_angle))
        for line in available
        if line.angle == dominant_angle
    ]
    if not line_geometry:
        return []

    local_page_size = (
        (page_size[1], page_size[0])
        if dominant_angle in {90, 270}
        else page_size
    )
    local_page_width, local_page_height = local_page_size
    if local_page_width <= 0 or local_page_height <= 0:
        return []
    effective_heights = [
        _line_effective_height(line, bbox)
        for line, bbox in line_geometry
    ]
    median_height = statistics.median(effective_heights) if effective_heights else 1.0
    lanes = _infer_text_lanes(line_geometry, local_page_width, median_height)
    local_axis_lines = _transform_axis_lines(
        drawing_lines,
        page_size,
        dominant_angle,
    )

    candidate_groups: list[set[int]] = []
    for axis_line in local_axis_lines:
        if axis_line.orientation != "horizontal":
            continue
        # 分隔线中心必须进入页面下方 30%，上方的正文分隔线不参与脚注判定。
        rule_center_y = _bbox_center_y(axis_line.bbox)
        if rule_center_y < 0.7 * local_page_height:
            continue
        # 表格边界会产生断裂横线；除框内线段外，也排除与其同高且近邻的框外线段。
        if _rule_belongs_to_confirmed_table(
            axis_line,
            local_axis_lines,
            table_bboxes,
            local_page_width,
        ):
            continue
        rule_source_indices: set[int] = set()
        for lane in lanes:
            rule_source_indices.update(
                _footnote_lane_members(
                    lane,
                    axis_line.bbox,
                    local_page_size,
                )
            )
        if rule_source_indices:
            candidate_groups.append(rule_source_indices)

    page_footnote_groups = _merge_overlapping_source_groups(candidate_groups)
    footnote_source_indices = set().union(*page_footnote_groups) if page_footnote_groups else set()
    for line in available:
        if line.source_index in footnote_source_indices:
            line.semantic_type = "page_footnote"
    return page_footnote_groups


def _rule_belongs_to_confirmed_table(
    candidate: _LocalAxisLine,
    local_axis_lines: list[_LocalAxisLine],
    table_bboxes: list[BBox],
    local_page_width: float,
) -> bool:
    """把表格框内横线及其同高近邻断裂段一并排除，避免框外残段触发脚注。"""

    if not table_bboxes:
        return False
    maximum_segment_gap = 0.04 * local_page_width
    for table_line in local_axis_lines:
        if table_line.orientation != "horizontal":
            continue
        table_margin = max(0.5, table_line.width)
        if not any(
            _bbox_intersects(
                _expand_bbox(table_line.original_bbox, table_margin),
                table_bbox,
            )
            for table_bbox in table_bboxes
        ):
            continue
        center_tolerance = max(1.0, candidate.width, table_line.width)
        if abs(_bbox_center_y(candidate.bbox) - _bbox_center_y(table_line.bbox)) > center_tolerance:
            continue
        if _horizontal_bbox_gap(candidate.bbox, table_line.bbox) <= maximum_segment_gap:
            return True
    return False


def _merge_overlapping_source_groups(groups: list[set[int]]) -> list[set[int]]:
    """合并共享来源行的分隔线候选组，消除重复绘图线造成的重复分组。"""

    merged: list[set[int]] = []
    for group in groups:
        combined = set(group)
        index = 0
        while index < len(merged):
            if combined & merged[index]:
                combined.update(merged.pop(index))
                index = 0
                continue
            index += 1
        merged.append(combined)
    return sorted(merged, key=lambda group: min(group))


def _footnote_lane_members(
    lane: _TextLane,
    rule_bbox: BBox,
    local_page_size: tuple[float, float],
) -> set[int]:
    """验证横线与单个栏带的对齐关系，并返回其下连续脚注行的来源编号。"""

    lane_lines = [item for item in lane.lines if item[0].semantic_type is None]
    if not lane_lines:
        return set()
    lane_lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
    local_page_width, local_page_height = local_page_size
    lane_width = max(0.1, lane.right - lane.left)
    lane_heights = [_line_effective_height(line, bbox) for line, bbox in lane_lines]
    median_height = statistics.median(lane_heights) if lane_heights else 1.0
    rule_width = max(0.0, rule_bbox[2] - rule_bbox[0])
    # 同时限制绝对短线、相对长线和左缘偏移，排除图标、公式线及跨栏正文分隔线。
    if rule_width < max(4.0 * median_height, 0.04 * local_page_width):
        return set()
    if rule_width > 0.65 * lane_width:
        return set()
    if abs(rule_bbox[0] - lane.left) > max(2.0 * median_height, 0.04 * lane_width):
        return set()

    # 首行采用较宽的 3.5% 页高窗口；命中后仅按紧凑的连续净空向下扩展。
    first_gap_limit = max(3.0 * median_height, 0.035 * local_page_height)
    first_index: int | None = None
    for index, (_line, bbox) in enumerate(lane_lines):
        rule_gap = bbox[1] - rule_bbox[3]
        if rule_gap < -0.5 * median_height:
            continue
        if rule_gap <= first_gap_limit:
            first_index = index
        break
    if first_index is None:
        return set()

    continuation_gap_limit = max(1.25 * median_height, 0.01 * local_page_height)
    members = [lane_lines[first_index]]
    for current in lane_lines[first_index + 1 :]:
        if _effective_text_row_gap(members[-1], current) > continuation_gap_limit:
            break
        members.append(current)
    return {line.source_index for line, _bbox in members}


def _classify_repeated_page_marginals(pages: list[_PreparedPage]) -> None:
    """仅用相邻或同奇偶页的重复证据标注页码、页眉和页脚。"""

    if len(pages) < 2:
        return
    candidates = [
        candidate
        for page_index, page in enumerate(pages)
        for line in page.remaining_lines
        if (candidate := _build_marginal_candidate(page_index, line, page.page_size)) is not None
    ]

    for left_index, left in enumerate(candidates):
        left_value = _parse_page_number_value(left.line.text)
        if left_value is None:
            continue
        for right in candidates[left_index + 1 :]:
            page_delta = right.page_index - left.page_index
            if page_delta > 2:
                break
            right_value = _parse_page_number_value(right.line.text)
            if (
                page_delta > 0
                and right_value is not None
                and right_value - left_value == page_delta
                and _page_number_candidates_match(left, right)
            ):
                left.line.semantic_type = "page_number"
                right.line.semantic_type = "page_number"

    for left_index, left in enumerate(candidates):
        if left.line.semantic_type == "page_number":
            continue
        for right in candidates[left_index + 1 :]:
            page_delta = right.page_index - left.page_index
            if page_delta > 2:
                break
            if (
                page_delta > 0
                and left.region != "side"
                and right.region != "side"
                and right.line.semantic_type != "page_number"
                and _marginal_geometry_matches(left, right)
                and _marginal_text_matches(left.line.text, right.line.text)
            ):
                left.line.semantic_type = left.region
                right.line.semantic_type = right.region


def _classify_repeated_visual_headers(pages: list[_PreparedPage]) -> None:
    """仅按页首位置与跨页重复几何，把整体图片重标为视觉页眉。"""

    candidates: list[tuple[int, dict[str, object], BBox, int]] = []
    for page_index, page in enumerate(pages):
        # 首页常使用独立封面版式，不参与正文页视觉页眉聚类。
        if page_index == 0:
            continue
        page_width, page_height = page.page_size
        if page_width <= 0 or page_height <= 0:
            continue
        for block in page.fixed_blocks:
            if block.get("type") != "image":
                continue
            content = block.get("content")
            # 当前 model_list 不保留空文本 header；这里只限制输出可表示性，
            # 跨页匹配本身完全不比较 content。
            if not isinstance(content, str) or not content.strip():
                continue
            bbox = _clip_bbox(_coerce_bbox(block.get("bbox")), page.page_size)
            if bbox is None or bbox[3] > 0.12 * page_height:
                continue
            normalized_bbox = (
                bbox[0] / page_width,
                bbox[1] / page_height,
                bbox[2] / page_width,
                bbox[3] / page_height,
            )
            angle = int(block.get("angle", 0) or 0) % 360
            candidates.append((page_index, block, normalized_bbox, angle))

    if len(candidates) < 3:
        return

    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        """查找视觉页眉候选所属几何簇的根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        """合并跨页距离和归一化几何均匹配的两个候选。"""

        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, (
        first_page,
        _first_block,
        first_bbox,
        first_angle,
    ) in enumerate(candidates):
        for second_index in range(first_index + 1, len(candidates)):
            second_page, _second_block, second_bbox, second_angle = candidates[
                second_index
            ]
            page_delta = second_page - first_page
            if page_delta > 2:
                break
            if (
                page_delta > 0
                and first_angle == second_angle
                and _visual_header_geometry_matches(first_bbox, second_bbox)
            ):
                union(first_index, second_index)

    clusters: dict[int, list[int]] = {}
    for candidate_index in range(len(candidates)):
        clusters.setdefault(find(candidate_index), []).append(candidate_index)
    for member_indices in clusters.values():
        page_indices = {candidates[index][0] for index in member_indices}
        if len(page_indices) < 3:
            continue
        for index in member_indices:
            candidates[index][1]["type"] = "header"


def _visual_header_geometry_matches(first: BBox, second: BBox) -> bool:
    """比较两个归一化页首图片的 IoU 与宽高尺度。"""

    first_width = first[2] - first[0]
    first_height = first[3] - first[1]
    second_width = second[2] - second[0]
    second_height = second[3] - second[1]
    if min(first_width, first_height, second_width, second_height) <= 0:
        return False
    if max(first_width, second_width) / min(first_width, second_width) > 1.1:
        return False
    if max(first_height, second_height) / min(first_height, second_height) > 1.1:
        return False

    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    union_area = first_width * first_height + second_width * second_height - intersection
    return union_area > 0 and intersection / union_area >= 0.9


def _build_marginal_candidate(
    page_index: int,
    line: _LineItem,
    page_size: tuple[float, float],
) -> _MarginalCandidate | None:
    """把页面上下百分之十五内的常规小行转换成跨页比较候选。"""

    if line.semantic_type is not None:
        return None
    local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
    local_page_size = (page_size[1], page_size[0]) if line.angle in {90, 270} else page_size
    local_page_width, local_page_height = local_page_size
    if local_page_width <= 0 or local_page_height <= 0:
        return None
    normalized_center_y = _bbox_center_y(local_bbox) / local_page_height
    normalized_center_x = _bbox_center_x(local_bbox) / local_page_width
    if normalized_center_y <= 0.15:
        region: Literal["header", "footer", "side"] = "header"
    elif normalized_center_y >= 0.85:
        region = "footer"
    elif (
        normalized_center_y <= 0.18
        or normalized_center_y >= 0.82
        or (
            (normalized_center_x <= 0.15 or normalized_center_x >= 0.85)
            and (normalized_center_y <= 0.3 or normalized_center_y >= 0.7)
        )
    ):
        # 仅页码递增逻辑会消费 side；稳定文本不会被侧栏位置猜成页眉页脚。
        region = "side"
    else:
        return None
    if _line_effective_height(line, local_bbox) > 0.06 * local_page_height:
        return None
    return _MarginalCandidate(
        page_index=page_index,
        line=line,
        local_bbox=local_bbox,
        local_page_size=local_page_size,
        region=region,
    )


def _page_number_candidates_match(
    first: _MarginalCandidate,
    second: _MarginalCandidate,
) -> bool:
    """校验连续页码的同边缘几何，横竖版切换时允许边缘位置随版面改变。"""

    if _marginal_geometry_matches(first, second):
        return True
    first_landscape = first.local_page_size[0] > first.local_page_size[1]
    second_landscape = second.local_page_size[0] > second.local_page_size[1]
    if first_landscape == second_landscape or first.line.angle != second.line.angle:
        return False
    first_height = _line_effective_height(first.line, first.local_bbox) / first.local_page_size[1]
    second_height = _line_effective_height(second.line, second.local_bbox) / second.local_page_size[1]
    return min(first_height, second_height) > 0 and max(first_height, second_height) / min(
        first_height,
        second_height,
    ) <= 1.5


def _marginal_geometry_matches(
    first: _MarginalCandidate,
    second: _MarginalCandidate,
) -> bool:
    """比较边缘候选的方向、纵向带、字号以及同侧或镜像横向位置。"""

    if first.region != second.region or first.line.angle != second.line.angle:
        return False
    first_width, first_height = first.local_page_size
    second_width, second_height = second.local_page_size
    first_y = _bbox_center_y(first.local_bbox) / first_height
    second_y = _bbox_center_y(second.local_bbox) / second_height
    if abs(first_y - second_y) > 0.025:
        return False
    first_line_height = _line_effective_height(first.line, first.local_bbox) / first_height
    second_line_height = _line_effective_height(second.line, second.local_bbox) / second_height
    if min(first_line_height, second_line_height) <= 0 or max(first_line_height, second_line_height) / min(
        first_line_height,
        second_line_height,
    ) > 1.35:
        return False
    if (
        first.line.font_signature is not None
        and second.line.font_signature is not None
        and first.line.font_coverage >= 0.75
        and second.line.font_coverage >= 0.75
        and first.line.font_signature != second.line.font_signature
    ):
        return False

    first_normalized_bbox = (
        first.local_bbox[0] / first_width,
        first.local_bbox[1] / first_height,
        first.local_bbox[2] / first_width,
        first.local_bbox[3] / first_height,
    )
    second_normalized_bbox = (
        second.local_bbox[0] / second_width,
        second.local_bbox[1] / second_height,
        second.local_bbox[2] / second_width,
        second.local_bbox[3] / second_height,
    )
    same_side = (
        _bbox_axis_overlap_ratio(first_normalized_bbox, second_normalized_bbox, axis="x") >= 0.4
        or abs(_bbox_center_x(first_normalized_bbox) - _bbox_center_x(second_normalized_bbox)) <= 0.08
    )
    mirrored = abs(
        _bbox_center_x(first_normalized_bbox) + _bbox_center_x(second_normalized_bbox) - 1.0
    ) <= 0.12
    return same_side or mirrored


def _parse_page_number_value(text: str) -> int | None:
    """解析整行阿拉伯、罗马或中文页码；混有稳定正文的行不作为纯页码。"""

    normalized = unicodedata.normalize("NFKC", str(text or ""))
    match = _PAGE_NUMBER_RE.fullmatch(normalized)
    if match is None:
        return None
    value = match.group("value")
    if value.isdecimal():
        return int(value)
    if re.fullmatch(r"[ivxlcdm]+", value, re.IGNORECASE):
        return _roman_number_to_int(value)
    return _chinese_page_number_to_int(value)


def _roman_number_to_int(value: str) -> int | None:
    """把页码中的规范罗马数字转换成整数，非法组合返回空。"""

    roman_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    normalized = value.upper()
    total = 0
    previous = 0
    for char in reversed(normalized):
        current = roman_values.get(char)
        if current is None:
            return None
        total += -current if current < previous else current
        previous = max(previous, current)
    if total <= 0 or total > 4999:
        return None
    return total


def _chinese_page_number_to_int(value: str) -> int | None:
    """把常见百位以内中文页码转换成整数，供跨页递增校验使用。"""

    digits = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if all(char in digits for char in value):
        try:
            return int("".join(str(digits[char]) for char in value))
        except ValueError:
            return None
    total = 0
    current_digit = 0
    for char in value:
        if char in digits:
            current_digit = digits[char]
        elif char == "十":
            total += (current_digit or 1) * 10
            current_digit = 0
        elif char == "百":
            total += (current_digit or 1) * 100
            current_digit = 0
        else:
            return None
    return total + current_digit if total + current_digit > 0 else None


def _marginal_text_matches(first_text: str, second_text: str) -> bool:
    """在屏蔽变化数字后比较边缘稳定文本，短文本只接受完全一致。"""

    first = _normalize_marginal_text(first_text)
    second = _normalize_marginal_text(second_text)
    if not first or not second:
        return False
    if first == second:
        return True
    if min(len(first), len(second)) < 8:
        return False
    return SequenceMatcher(a=first, b=second, autojunk=False).ratio() >= 0.92


def _normalize_marginal_text(text: str) -> str:
    """统一边缘重复文本的宽窄字符、大小写、空白和可变数字。"""

    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    normalized = re.sub(r"\d+", "#", normalized)
    return re.sub(r"\s+", "", normalized).strip()
