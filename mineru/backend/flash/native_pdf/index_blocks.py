# Copyright (c) Opendatalab. All rights reserved.

"""识别并构造 Flash 原生 PDF 的目录正文块。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import statistics


from mineru.types import BBox, BlockType

from .models import _LineItem
from .geometry import (
    _bbox_center_x,
    _bbox_center_y,
    _bbox_overlap_in_smaller,
    _bbox_union_many,
    _rotate_bbox_to_upright,
)
from .line_layout import _line_effective_height


_INDEX_PAGE_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9０-９])(?:[0-9０-９]+|[ivxlcdmIVXLCDM]+)"
    r"[\s.。．、,，;；:：)）\]】]*$"
)
_INDEX_MIN_ROWS = 5
_INDEX_MIN_PAGE_NUMBER_RATIO = 0.7


@dataclass(slots=True)
class _IndexRow:
    """保存目录候选视觉行的成员、局部几何和合并文本。"""

    members: list[_LineItem]
    local_member_bboxes: list[BBox]
    local_bbox: BBox
    content: str
    ends_in_page_number: bool


def _extract_index_blocks(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    container_bboxes: list[BBox],
) -> tuple[list[dict[str, object]], list[_LineItem]]:
    """用页码行尾和稳定版式识别目录区域，并返回独立 index 块。"""

    claimed_line_ids: set[int] = set()
    blocks: list[dict[str, object]] = []
    for angle in sorted({line.angle for line in lines if line.semantic_type is None}):
        rows = _build_index_rows(lines, page_size, angle)
        if len(rows) < _INDEX_MIN_ROWS:
            continue
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        local_containers = [_rotate_bbox_to_upright(bbox, page_size, angle) for bbox in container_bboxes]
        eligible_rows = [
            row
            for row in rows
            if not any(_bbox_overlap_in_smaller(row.local_bbox, container_bbox) >= 0.35 for container_bbox in local_containers)
        ]
        for band in _split_index_bands(eligible_rows):
            candidate = _trim_index_band_edges(band)
            if not _index_band_has_stable_layout(candidate, local_page_width):
                continue
            heading_row = _find_index_heading_row(
                eligible_rows,
                candidate,
                local_page_width,
            )
            if heading_row is not None:
                for line in heading_row.members:
                    line.semantic_type = "paragraph_title"
            for row in candidate:
                for line in row.members:
                    line.semantic_type = BlockType.INDEX
                    claimed_line_ids.add(id(line))
            blocks.append(
                {
                    "type": BlockType.INDEX,
                    "bbox": _bbox_union_many([line.bbox for row in candidate for line in row.members]),
                    "angle": angle,
                    "content": "\n".join(row.content for row in candidate),
                }
            )

    remaining_lines = [line for line in lines if id(line) not in claimed_line_ids]
    return blocks, remaining_lines


def _build_index_rows(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
) -> list[_IndexRow]:
    """按 visual_row_id 复原当前方向的完整视觉行，保留左右分裂成员。"""

    row_groups: dict[tuple[str, int], list[_LineItem]] = {}
    for line in lines:
        if line.angle != angle or line.semantic_type is not None:
            continue
        key = ("visual", line.visual_row_id) if line.visual_row_id is not None else ("source", line.source_index)
        row_groups.setdefault(key, []).append(line)

    rows: list[_IndexRow] = []
    for members in row_groups.values():
        ordered = sorted(
            members,
            key=lambda line: (
                _rotate_bbox_to_upright(line.bbox, page_size, angle)[0],
                line.run_index,
                line.source_index,
            ),
        )
        local_member_bboxes = [_rotate_bbox_to_upright(line.bbox, page_size, angle) for line in ordered]
        local_bbox = _bbox_union_many(local_member_bboxes)
        content = " ".join(part for line in ordered if (part := line.text.strip()))
        if not content:
            continue
        rows.append(
            _IndexRow(
                members=ordered,
                local_member_bboxes=local_member_bboxes,
                local_bbox=local_bbox,
                content=content,
                ends_in_page_number=_index_row_ends_in_page_number(content),
            )
        )
    rows.sort(
        key=lambda row: (
            row.local_bbox[1],
            row.local_bbox[0],
            min(line.source_index for line in row.members),
        )
    )
    return rows


def _index_row_ends_in_page_number(content: str) -> bool:
    """识别行尾的半角、全角阿拉伯页码或罗马页码。"""

    return _INDEX_PAGE_NUMBER_RE.search(content.rstrip()) is not None


def _split_index_bands(rows: list[_IndexRow]) -> list[list[_IndexRow]]:
    """按显著纵向断层拆分候选带，同时容纳目录章节之间的加大行距。"""

    if not rows:
        return []
    median_height = statistics.median(max(_line_effective_height(line, row.local_bbox) for line in row.members) for row in rows)
    maximum_pitch = 3.25 * max(0.1, median_height)
    bands: list[list[_IndexRow]] = [[rows[0]]]
    for row in rows[1:]:
        previous = bands[-1][-1]
        if _bbox_center_y(row.local_bbox) - _bbox_center_y(previous.local_bbox) > maximum_pitch:
            bands.append([row])
        else:
            bands[-1].append(row)
    return bands


def _trim_index_band_edges(rows: list[_IndexRow]) -> list[_IndexRow]:
    """移除候选带两端不带页码的标题或邻接正文，内部少量续行继续保留。"""

    start = 0
    end = len(rows)
    while start < end and not rows[start].ends_in_page_number:
        start += 1
    while end > start and not rows[end - 1].ends_in_page_number:
        end -= 1
    return rows[start:end]


def _index_band_has_stable_layout(
    rows: list[_IndexRow],
    local_page_width: float,
) -> bool:
    """联合页码比例、右边界、行宽、缩进和行距确认目录候选。"""

    if len(rows) < _INDEX_MIN_ROWS or local_page_width <= 0:
        return False
    page_number_rows = [row for row in rows if row.ends_in_page_number]
    required_page_number_rows = max(
        4,
        math.ceil(_INDEX_MIN_PAGE_NUMBER_RATIO * len(rows)),
    )
    if len(page_number_rows) < required_page_number_rows:
        return False

    row_heights = [max(0.1, row.local_bbox[3] - row.local_bbox[1]) for row in rows]
    median_height = statistics.median(row_heights)
    median_right = statistics.median(row.local_bbox[2] for row in page_number_rows)
    right_tolerance = max(1.5 * median_height, 0.02 * local_page_width)
    aligned_right_ratio = sum(abs(row.local_bbox[2] - median_right) <= right_tolerance for row in page_number_rows) / len(
        page_number_rows
    )
    if median_right < 0.75 * local_page_width or aligned_right_ratio < 0.7:
        return False

    wide_row_ratio = sum(row.local_bbox[2] - row.local_bbox[0] >= 0.55 * local_page_width for row in rows) / len(rows)
    right_sidecar_count = sum(_index_row_has_right_sidecar(row, local_page_width, median_height) for row in rows)
    if wide_row_ratio < 0.7 and right_sidecar_count < 2:
        return False

    left_body_ratio = sum(row.local_bbox[0] <= 0.3 * local_page_width for row in rows) / len(rows)
    if left_body_ratio < 0.7:
        return False

    pitches = [
        _bbox_center_y(current.local_bbox) - _bbox_center_y(previous.local_bbox) for previous, current in zip(rows, rows[1:])
    ]
    if not pitches or min(pitches) <= 0:
        return False
    median_pitch = statistics.median(pitches)
    regular_pitch_ratio = sum(0.55 * median_pitch <= pitch <= 1.75 * median_pitch for pitch in pitches) / len(pitches)
    return regular_pitch_ratio >= 0.7


def _find_index_heading_row(
    all_rows: list[_IndexRow],
    candidate: list[_IndexRow],
    local_page_width: float,
) -> _IndexRow | None:
    """用候选带上方的居中、短行和垂直邻接关系保留目录标题。"""

    if not candidate:
        return None
    first_position = next(
        (index for index, row in enumerate(all_rows) if row is candidate[0]),
        None,
    )
    if first_position is None or first_position == 0:
        return None
    heading = all_rows[first_position - 1]
    if heading.ends_in_page_number:
        return None
    candidate_heights = [max(0.1, row.local_bbox[3] - row.local_bbox[1]) for row in candidate]
    median_height = statistics.median(candidate_heights)
    heading_width = heading.local_bbox[2] - heading.local_bbox[0]
    centered = abs(_bbox_center_x(heading.local_bbox) - 0.5 * local_page_width) <= 0.15 * local_page_width
    vertical_pitch = _bbox_center_y(candidate[0].local_bbox) - _bbox_center_y(heading.local_bbox)
    if (
        heading_width > 0.4 * local_page_width
        or not centered
        or not 1.25 * median_height <= vertical_pitch <= 5.0 * median_height
    ):
        return None
    return heading


def _index_row_has_right_sidecar(
    row: _IndexRow,
    local_page_width: float,
    median_height: float,
) -> bool:
    """检查视觉行末尾是否存在靠近页面右侧的窄页码片段。"""

    if len(row.members) < 2:
        return False
    last_bbox = row.local_member_bboxes[-1]
    return last_bbox[0] >= 0.75 * local_page_width and last_bbox[2] - last_bbox[0] <= max(
        0.12 * local_page_width, 4.0 * median_height
    )
