# Copyright (c) Opendatalab. All rights reserved.
from dataclasses import dataclass
from typing import Any, Final, Sequence


"""
 * XY-Cut++ algorithm for reading order detection based on arXiv:2504.10258.
 * <p>
 * An enhanced XY-Cut implementation that handles:
 * <ul>
 *   <li>Cross-layout elements (headers, footers spanning multiple columns)</li>
 *   <li>Adaptive axis selection based on density ratios</li>
 *   <li>L-shaped region handling</li>
 * </ul>
 * <p>
 * This is a simplified geometric implementation without semantic type priorities.
 * <p>
 * Algorithm overview:
 * <ol>
 *   <li>Pre-mask: Identify cross-layout elements (width > beta * maxWidth, overlaps >= 2)</li>
 *   <li>Compute density ratio to determine split direction preference</li>
 *   <li>Recursive segmentation with adaptive XY/YX-Cut</li>
 *   <li>Merge cross-layout elements at appropriate positions</li>
 * </ol>
"""


DEFAULT_BETA: Final = 2.0
DEFAULT_DENSITY_THRESHOLD: Final = 0.9
OVERLAP_THRESHOLD: Final = 0.1
MIN_OVERLAP_COUNT: Final = 2
MIN_GAP_THRESHOLD: Final = 5.0
NARROW_ELEMENT_WIDTH_RATIO: Final = 0.1


@dataclass
class _CutInfo:
    position: float
    gap: float


@dataclass
class _SortableEntry:
    index: int
    payload: dict[str, Any]
    bbox: tuple[float, float, float, float]

    @property
    def left(self) -> float:
        return self.bbox[0]

    @property
    def top(self) -> float:
        return self.bbox[1]

    @property
    def right(self) -> float:
        return self.bbox[2]

    @property
    def bottom(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0


def sort_entries(
    entries: Sequence[dict[str, Any]],
    *,
    beta: float = DEFAULT_BETA,
    density_threshold: float = DEFAULT_DENSITY_THRESHOLD,
) -> list[dict[str, Any]]:
    if len(entries) <= 1:
        return list(entries)

    valid_entries = _build_sortable_entries(entries)
    if len(valid_entries) <= 1:
        return [entry.payload for entry in valid_entries]

    cross_layout_entries = _identify_cross_layout_elements(valid_entries, beta)
    cross_layout_ids = {entry.index for entry in cross_layout_entries}
    remaining_entries = [
        entry for entry in valid_entries if entry.index not in cross_layout_ids
    ]

    if not remaining_entries:
        return [entry.payload for entry in _sort_by_y_then_x(valid_entries)]

    density_ratio = _compute_density_ratio(remaining_entries)
    prefer_horizontal_first = density_ratio > density_threshold
    sorted_main = _recursive_segment(remaining_entries, prefer_horizontal_first)
    merged_entries = _merge_cross_layout_elements(sorted_main, cross_layout_entries)
    return [entry.payload for entry in merged_entries]


def _build_sortable_entries(
    entries: Sequence[dict[str, Any]],
) -> list[_SortableEntry]:
    sortable_entries: list[_SortableEntry] = []
    for index, entry in enumerate(entries):
        bbox = _normalize_bbox(entry.get("bbox"))
        if bbox is None:
            continue
        sortable_entries.append(
            _SortableEntry(
                index=index,
                payload=entry,
                bbox=bbox,
            )
        )
    return sortable_entries


def _normalize_bbox(
    bbox: Any,
) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None

    if x1 <= x0 or y1 <= y0:
        return None

    return (x0, y0, x1, y1)


def _identify_cross_layout_elements(
    entries: Sequence[_SortableEntry], beta: float
) -> list[_SortableEntry]:
    if len(entries) < 3:
        return []

    max_width = max(entry.width for entry in entries)
    threshold = beta * max_width

    cross_layout_entries = []
    for entry in entries:
        if entry.width < threshold:
            continue
        if _has_minimum_overlaps(entry, entries, MIN_OVERLAP_COUNT):
            cross_layout_entries.append(entry)
    return cross_layout_entries


def _has_minimum_overlaps(
    entry: _SortableEntry,
    entries: Sequence[_SortableEntry],
    min_count: int,
) -> bool:
    overlap_count = 0
    for other in entries:
        if other.index == entry.index:
            continue
        overlap_ratio = _calculate_horizontal_overlap_ratio(entry, other)
        if overlap_ratio < OVERLAP_THRESHOLD:
            continue
        overlap_count += 1
        if overlap_count >= min_count:
            return True
    return False


def _calculate_horizontal_overlap_ratio(
    entry1: _SortableEntry,
    entry2: _SortableEntry,
) -> float:
    overlap_left = max(entry1.left, entry2.left)
    overlap_right = min(entry1.right, entry2.right)
    overlap_width = max(0.0, overlap_right - overlap_left)
    if overlap_width <= 0:
        return 0.0

    smaller_width = min(entry1.width, entry2.width)
    if smaller_width <= 0:
        return 0.0

    return overlap_width / smaller_width


def _compute_density_ratio(entries: Sequence[_SortableEntry]) -> float:
    if not entries:
        return 1.0

    region = _calculate_bounding_region(entries)
    if region is None:
        return 1.0

    region_area = (region[2] - region[0]) * (region[3] - region[1])
    if region_area <= 0:
        return 1.0

    content_area = sum(entry.area for entry in entries)
    return min(1.0, content_area / region_area)


def _calculate_bounding_region(
    entries: Sequence[_SortableEntry],
) -> tuple[float, float, float, float] | None:
    if not entries:
        return None

    left = min(entry.left for entry in entries)
    top = min(entry.top for entry in entries)
    right = max(entry.right for entry in entries)
    bottom = max(entry.bottom for entry in entries)
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _recursive_segment(
    entries: Sequence[_SortableEntry],
    prefer_horizontal_first: bool,
) -> list[_SortableEntry]:
    if len(entries) <= 1:
        return list(entries)

    horizontal_cut = _find_best_horizontal_cut_with_projection(entries)
    vertical_cut = _find_best_vertical_cut_with_projection(entries)

    has_valid_horizontal_cut = horizontal_cut.gap >= MIN_GAP_THRESHOLD
    has_valid_vertical_cut = vertical_cut.gap >= MIN_GAP_THRESHOLD

    if has_valid_horizontal_cut and has_valid_vertical_cut:
        use_horizontal_cut = horizontal_cut.gap > vertical_cut.gap
    elif has_valid_horizontal_cut:
        use_horizontal_cut = True
    elif has_valid_vertical_cut:
        use_horizontal_cut = False
    else:
        return _sort_by_y_then_x(entries)

    if use_horizontal_cut:
        groups = _split_by_horizontal_cut(entries, horizontal_cut.position)
    else:
        groups = _split_by_vertical_cut(entries, vertical_cut.position)

    if len(groups) <= 1:
        return _sort_by_y_then_x(entries)

    result: list[_SortableEntry] = []
    for group in groups:
        result.extend(_recursive_segment(group, prefer_horizontal_first))
    return result


def _find_best_vertical_cut_with_projection(
    entries: Sequence[_SortableEntry],
) -> _CutInfo:
    if len(entries) < 2:
        return _CutInfo(0.0, 0.0)

    edge_cut = _find_vertical_cut_by_edges(entries)
    if edge_cut.gap >= MIN_GAP_THRESHOLD:
        return edge_cut

    if len(entries) < 3:
        return edge_cut

    region = _calculate_bounding_region(entries)
    if region is None:
        return edge_cut

    region_width = region[2] - region[0]
    narrow_threshold = region_width * NARROW_ELEMENT_WIDTH_RATIO
    filtered_entries = [
        entry for entry in entries if entry.width >= narrow_threshold
    ]
    if len(filtered_entries) < 2 or len(filtered_entries) == len(entries):
        return edge_cut

    filtered_cut = _find_vertical_cut_by_edges(filtered_entries)
    if (
        filtered_cut.gap > edge_cut.gap
        and filtered_cut.gap >= MIN_GAP_THRESHOLD
    ):
        return filtered_cut
    return edge_cut


def _find_vertical_cut_by_edges(entries: Sequence[_SortableEntry]) -> _CutInfo:
    sorted_entries = sorted(entries, key=lambda entry: (entry.left, entry.right))
    largest_gap = 0.0
    cut_position = 0.0
    prev_right: float | None = None

    for entry in sorted_entries:
        if prev_right is not None and entry.left > prev_right:
            gap = entry.left - prev_right
            if gap > largest_gap:
                largest_gap = gap
                cut_position = (prev_right + entry.left) / 2.0
        prev_right = entry.right if prev_right is None else max(prev_right, entry.right)

    return _CutInfo(cut_position, largest_gap)


def _find_best_horizontal_cut_with_projection(
    entries: Sequence[_SortableEntry],
) -> _CutInfo:
    if len(entries) < 2:
        return _CutInfo(0.0, 0.0)

    sorted_entries = sorted(entries, key=lambda entry: (entry.top, entry.bottom))
    largest_gap = 0.0
    cut_position = 0.0
    prev_bottom: float | None = None

    for entry in sorted_entries:
        if prev_bottom is not None and entry.top > prev_bottom:
            gap = entry.top - prev_bottom
            if gap > largest_gap:
                largest_gap = gap
                cut_position = (prev_bottom + entry.top) / 2.0
        prev_bottom = entry.bottom if prev_bottom is None else max(prev_bottom, entry.bottom)

    return _CutInfo(cut_position, largest_gap)


def _split_by_horizontal_cut(
    entries: Sequence[_SortableEntry],
    cut_y: float,
) -> list[list[_SortableEntry]]:
    above = [entry for entry in entries if entry.center_y < cut_y]
    below = [entry for entry in entries if entry.center_y >= cut_y]

    groups = []
    if above:
        groups.append(above)
    if below:
        groups.append(below)
    return groups


def _split_by_vertical_cut(
    entries: Sequence[_SortableEntry],
    cut_x: float,
) -> list[list[_SortableEntry]]:
    left = [entry for entry in entries if entry.center_x < cut_x]
    right = [entry for entry in entries if entry.center_x >= cut_x]

    groups = []
    if left:
        groups.append(left)
    if right:
        groups.append(right)
    return groups


def _merge_cross_layout_elements(
    sorted_main: Sequence[_SortableEntry],
    cross_layout_entries: Sequence[_SortableEntry],
) -> list[_SortableEntry]:
    if not cross_layout_entries:
        return list(sorted_main)

    if not sorted_main:
        return _sort_by_y_then_x(cross_layout_entries)

    sorted_cross_layout = _sort_by_y_then_x(cross_layout_entries)

    result: list[_SortableEntry] = []
    main_index = 0
    cross_index = 0

    while main_index < len(sorted_main) or cross_index < len(sorted_cross_layout):
        if cross_index >= len(sorted_cross_layout):
            result.append(sorted_main[main_index])
            main_index += 1
            continue

        if main_index >= len(sorted_main):
            result.append(sorted_cross_layout[cross_index])
            cross_index += 1
            continue

        main_entry = sorted_main[main_index]
        cross_entry = sorted_cross_layout[cross_index]
        if cross_entry.top <= main_entry.top:
            result.append(cross_entry)
            cross_index += 1
        else:
            result.append(main_entry)
            main_index += 1

    return result


def _sort_by_y_then_x(
    entries: Sequence[_SortableEntry],
) -> list[_SortableEntry]:
    return sorted(entries, key=lambda entry: (entry.top, entry.left))
