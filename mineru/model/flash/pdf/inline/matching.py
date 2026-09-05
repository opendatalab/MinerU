# Copyright (c) Opendatalab. All rights reserved.
"""对齐原生行证据与输出块文本，保留来源和偏移。"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any, Sequence

from loguru import logger

from .....types import BBox
from .....utils.text import is_hyphen_at_line_end

if TYPE_CHECKING:
    from ..models import _LineItem
    from ..native_text import _NativeVisualResplit
from .common import _bbox_overlap_ratio, _canonical_styles, _coerce_bbox, _normalize_match_fragment, _ordered_line_chars
from .types import (
    _PDF_GEOMETRIC_TEXT_STYLES,
    _PDF_TEXT_STYLE_TARGET_BLOCK_TYPES,
    PDF_NATURAL_TEXT_STYLE_BLOCK_TYPES,
    PDFTextEvidenceLine,
    PDFTextLinkLine,
    PDFTextLinkRange,
    PDFTextScriptLine,
    PDFTextScriptRange,
    PDFTextStyle,
    PDFTextStyleLine,
    PDFTextStyleRange,
    _LineProjectionMatch,
    _MatchedLinkRange,
    _ProjectedChar,
    _RawLinkInterval,
)


def _resplit_evidence_segments(
    evidence_text: str,
    resplit: _NativeVisualResplit,
) -> list[tuple[_LineItem, int, int, str]] | None:
    """按原字符身份把紧凑 evidence 文本映射到每个重切成员区间。"""

    source_spans: list[tuple[int, int]] = []
    spans_by_object_id: dict[int, tuple[int, int]] = {}
    spans_by_char_idx: dict[int, list[tuple[int, int]]] = {}
    source_parts: list[str] = []
    cursor = 0
    for char in _ordered_line_chars(resplit.source):
        fragment = _normalize_match_fragment(char.get("char"))
        if not fragment:
            continue
        span = (cursor, cursor + len(fragment))
        source_spans.append(span)
        spans_by_object_id[id(char)] = span
        char_idx = char.get("char_idx")
        if isinstance(char_idx, int) and not isinstance(char_idx, bool):
            spans_by_char_idx.setdefault(char_idx, []).append(span)
        source_parts.append(fragment)
        cursor = span[1]
    if "".join(source_parts) != evidence_text:
        return None

    used_spans: list[tuple[int, int]] = []
    segments: list[tuple[_LineItem, int, int, str]] = []
    for member in sorted(
        resplit.members,
        key=lambda item: (item.run_index, item.source_index),
    ):
        member_spans: list[tuple[int, int]] = []
        member_parts: list[str] = []
        for char in _ordered_line_chars(member):
            fragment = _normalize_match_fragment(char.get("char"))
            if not fragment:
                continue
            span = spans_by_object_id.get(id(char))
            if span is None:
                char_idx = char.get("char_idx")
                candidates = (
                    spans_by_char_idx.get(char_idx, []) if isinstance(char_idx, int) and not isinstance(char_idx, bool) else []
                )
                span = candidates[0] if len(candidates) == 1 else None
            if span is None or evidence_text[span[0] : span[1]] != fragment:
                return None
            member_spans.append(span)
            member_parts.append(fragment)
        if not member_spans:
            return None
        member_start = min(start for start, _end in member_spans)
        member_end = max(end for _start, end in member_spans)
        member_text = "".join(member_parts)
        if (
            sum(end - start for start, end in member_spans) != member_end - member_start
            or evidence_text[member_start:member_end] != member_text
        ):
            return None
        used_spans.extend(member_spans)
        segments.append(
            (member, member_start, member_end, member_text),
        )
    if sorted(used_spans) != source_spans:
        return None
    return segments


def _partition_resplit_text_evidence(
    style_lines: list[PDFTextStyleLine],
    link_lines: list[PDFTextLinkLine],
    resplits: dict[int, _NativeVisualResplit],
) -> tuple[list[PDFTextStyleLine], list[PDFTextLinkLine]]:
    """只替换被重切粗行的样式与链接 evidence，其它行保持原对象和顺序。"""

    if not resplits:
        return style_lines, link_lines

    partitioned_styles: list[PDFTextStyleLine] = []
    for line in style_lines:
        resplit = resplits.get(line.source_index)
        if resplit is None:
            partitioned_styles.append(line)
            continue
        segments = _resplit_evidence_segments(line.text, resplit)
        if segments is None:
            logger.warning(
                "Keep coarse PDF style evidence after an unsafe resplit mapping: "
                f"source_index={line.source_index}, text={line.text!r}"
            )
            partitioned_styles.append(line)
            continue
        for member, member_start, member_end, member_text in segments:
            style_ranges = tuple(
                PDFTextStyleRange(
                    start=max(member_start, style_range.start) - member_start,
                    end=min(member_end, style_range.end) - member_start,
                    styles=style_range.styles,
                )
                for style_range in line.style_ranges
                if max(member_start, style_range.start) < min(member_end, style_range.end)
            )
            partitioned_styles.append(
                PDFTextStyleLine(
                    bbox=member.bbox,
                    text=member_text,
                    style_ranges=style_ranges,
                    source_index=member.source_index,
                )
            )

    partitioned_links: list[PDFTextLinkLine] = []
    for line in link_lines:
        resplit = resplits.get(line.source_index)
        if resplit is None:
            partitioned_links.append(line)
            continue
        segments = _resplit_evidence_segments(line.text, resplit)
        if segments is None:
            logger.warning(
                "Keep coarse PDF link evidence after an unsafe resplit mapping: "
                f"source_index={line.source_index}, text={line.text!r}"
            )
            partitioned_links.append(line)
            continue
        for member, member_start, member_end, member_text in segments:
            link_ranges = tuple(
                PDFTextLinkRange(
                    start=max(member_start, link_range.start) - member_start,
                    end=min(member_end, link_range.end) - member_start,
                    target=link_range.target,
                )
                for link_range in line.link_ranges
                if max(member_start, link_range.start) < min(member_end, link_range.end)
            )
            if not link_ranges:
                continue
            partitioned_links.append(
                PDFTextLinkLine(
                    bbox=member.bbox,
                    text=member_text,
                    link_ranges=link_ranges,
                    source_index=member.source_index,
                )
            )
    return partitioned_styles, partitioned_links


def _realign_repaired_text_evidence(
    style_lines: list[PDFTextStyleLine],
    link_lines: list[PDFTextLinkLine],
    line_bboxes: dict[int, BBox],
    resplits: dict[int, _NativeVisualResplit],
) -> tuple[list[PDFTextStyleLine], list[PDFTextLinkLine]]:
    """同步未重切修复行的 evidence 框，再按字符身份切分发生重切的样式与链接。"""

    aligned_styles = style_lines
    for index, line in enumerate(style_lines):
        bbox = line_bboxes.get(line.source_index)
        if line.source_index in resplits or bbox is None or bbox == line.bbox:
            continue
        if aligned_styles is style_lines:
            aligned_styles = list(style_lines)
        aligned_styles[index] = PDFTextStyleLine(
            bbox=bbox,
            text=line.text,
            style_ranges=line.style_ranges,
            source_index=line.source_index,
        )

    aligned_links = link_lines
    for index, line in enumerate(link_lines):
        bbox = line_bboxes.get(line.source_index)
        if line.source_index in resplits or bbox is None or bbox == line.bbox:
            continue
        if aligned_links is link_lines:
            aligned_links = list(link_lines)
        aligned_links[index] = PDFTextLinkLine(
            bbox=bbox,
            text=line.text,
            link_ranges=line.link_ranges,
            source_index=line.source_index,
        )

    return _partition_resplit_text_evidence(
        aligned_styles,
        aligned_links,
        resplits,
    )


def _block_bbox_to_page_bbox(value: Any, page_size: tuple[float, float]) -> BBox | None:
    """把 model-list 的归一化 bbox 转回页面 point，同时兼容已是绝对坐标的内部输入。"""

    bbox = _coerce_bbox(value)
    if bbox is None:
        return None
    if all(0.0 <= coordinate <= 1.0 for coordinate in bbox):
        return (
            bbox[0] * page_size[0],
            bbox[1] * page_size[1],
            bbox[2] * page_size[0],
            bbox[3] * page_size[1],
        )
    return bbox


def _line_block_score(line_bbox: BBox, block_bbox: BBox) -> tuple[float, float, float]:
    """计算文本行归属 block 的中心包含、重叠率与紧致度评分。"""

    center_x = (line_bbox[0] + line_bbox[2]) / 2
    center_y = (line_bbox[1] + line_bbox[3]) / 2
    center_inside = float(block_bbox[0] <= center_x <= block_bbox[2] and block_bbox[1] <= center_y <= block_bbox[3])
    overlap_ratio = _bbox_overlap_ratio(line_bbox, block_bbox)
    block_area = (block_bbox[2] - block_bbox[0]) * (block_bbox[3] - block_bbox[1])
    return center_inside, overlap_ratio, -block_area


def _assign_lines_to_blocks(
    blocks: list[dict[str, Any]],
    lines: Sequence[PDFTextEvidenceLine],
    page_size: tuple[float, float],
) -> dict[int, list[PDFTextEvidenceLine]]:
    """把每个视觉文本行唯一分配给最匹配的自然语言 block。"""

    target_bboxes = {
        block_index: block_bbox
        for block_index, block in enumerate(blocks)
        if block.get("type") in PDF_NATURAL_TEXT_STYLE_BLOCK_TYPES
        and isinstance(block.get("content"), str)
        and (block_bbox := _block_bbox_to_page_bbox(block.get("bbox"), page_size)) is not None
    }
    assignments: dict[int, list[PDFTextEvidenceLine]] = {}
    for line in lines:
        matches = [
            (block_index, _line_block_score(line.bbox, block_bbox))
            for block_index, block_bbox in target_bboxes.items()
            if (
                block_bbox[0] <= (line.bbox[0] + line.bbox[2]) / 2 <= block_bbox[2]
                and block_bbox[1] <= (line.bbox[1] + line.bbox[3]) / 2 <= block_bbox[3]
            )
            or _bbox_overlap_ratio(line.bbox, block_bbox) >= 0.5
        ]
        if not matches:
            continue
        block_index, _score = max(matches, key=lambda item: (*item[1], -item[0]))
        assignments.setdefault(block_index, []).append(line)
    for block_lines in assignments.values():
        block_lines.sort(
            key=lambda line: (
                line.source_index,
                line.bbox[1],
                line.bbox[0],
            )
        )
    return assignments


def _assign_script_lines_to_blocks(
    blocks: list[dict[str, Any]],
    lines: Sequence[PDFTextScriptLine],
    page_size: tuple[float, float],
) -> dict[int, list[PDFTextScriptLine]]:
    """保留整行主归属，并为无法投影的脚本区间补充 tight bbox 备用归属。"""

    target_bboxes = {
        block_index: block_bbox
        for block_index, block in enumerate(blocks)
        if block.get("type") in PDF_NATURAL_TEXT_STYLE_BLOCK_TYPES
        and isinstance(block.get("content"), str)
        and (block_bbox := _block_bbox_to_page_bbox(block.get("bbox"), page_size)) is not None
    }
    target_projected = {
        block_index: _project_content_chars(str(blocks[block_index]["content"])) for block_index in target_bboxes
    }
    primary_assignments = _assign_lines_to_blocks(blocks, lines, page_size)
    assignments: dict[int, list[PDFTextScriptLine]] = {
        block_index: [line for line in block_lines if isinstance(line, PDFTextScriptLine)]
        for block_index, block_lines in primary_assignments.items()
    }
    primary_block_by_line = {id(line): block_index for block_index, block_lines in assignments.items() for line in block_lines}
    fallback_ranges: dict[tuple[int, int], list[PDFTextScriptRange]] = {}
    for line_index, line in enumerate(lines):
        for script_range in line.script_ranges:
            evidence_line = PDFTextStyleLine(
                bbox=line.bbox,
                text=line.text,
                style_ranges=(
                    PDFTextStyleRange(
                        script_range.start,
                        script_range.end,
                        (script_range.style,),
                    ),
                ),
                source_index=line.source_index,
            )
            primary_block_index = primary_block_by_line.get(id(line))
            if primary_block_index is not None and _match_script_line_ranges(
                target_projected[primary_block_index],
                evidence_line,
            ):
                continue
            matches = [
                (
                    block_index,
                    _line_block_score(script_range.bbox, block_bbox),
                )
                for block_index, block_bbox in target_bboxes.items()
                if block_index != primary_block_index
                and _match_script_line_ranges(
                    target_projected[block_index],
                    evidence_line,
                )
                if (
                    block_bbox[0] <= (script_range.bbox[0] + script_range.bbox[2]) / 2 <= block_bbox[2]
                    and block_bbox[1] <= (script_range.bbox[1] + script_range.bbox[3]) / 2 <= block_bbox[3]
                )
                or _bbox_overlap_ratio(script_range.bbox, block_bbox) >= 0.5
            ]
            if not matches:
                continue
            block_index, _score = max(matches, key=lambda item: (*item[1], -item[0]))
            fallback_ranges.setdefault((block_index, line_index), []).append(script_range)
    for (block_index, line_index), script_ranges in fallback_ranges.items():
        line = lines[line_index]
        assignments.setdefault(block_index, []).append(
            PDFTextScriptLine(
                bbox=line.bbox,
                text=line.text,
                script_ranges=tuple(script_ranges),
                source_index=line.source_index,
                angle=line.angle,
            )
        )
    for block_lines in assignments.values():
        block_lines.sort(key=lambda line: (line.source_index, line.bbox[1], line.bbox[0]))
    return assignments


def _filter_line_styles_for_block(
    lines: Sequence[PDFTextStyleLine],
    block_type: Any,
) -> list[PDFTextStyleLine]:
    """按目标 block 类型过滤样式区间，同时保留无样式物理行用于顺序对齐。"""

    output: list[PDFTextStyleLine] = []
    for line in lines:
        filtered_ranges: list[PDFTextStyleRange] = []
        for style_range in line.style_ranges:
            styles = _canonical_styles(
                style
                for style in style_range.styles
                if block_type
                in _PDF_TEXT_STYLE_TARGET_BLOCK_TYPES.get(
                    style,
                    frozenset(),
                )
            )
            if styles:
                filtered_ranges.append(
                    PDFTextStyleRange(
                        style_range.start,
                        style_range.end,
                        styles,
                    )
                )
        output.append(
            PDFTextStyleLine(
                bbox=line.bbox,
                text=line.text,
                style_ranges=tuple(filtered_ranges),
                source_index=line.source_index,
            )
        )
    return output


def _project_content_chars(content: str) -> list[_ProjectedChar]:
    """把原始文字投影为忽略空白和圆括号公式的可比较字符。"""

    projected: list[_ProjectedChar] = []
    pending_formula_gap = False
    cursor = 0
    while cursor < len(content):
        if content.startswith(r"\(", cursor):
            formula_end = content.find(r"\)", cursor + 2)
            if formula_end >= 0:
                cursor = formula_end + 2
                pending_formula_gap = True
                continue
        raw_char = content[cursor]
        fragment = _normalize_match_fragment(raw_char)
        for fragment_index, value in enumerate(fragment):
            projected.append(
                _ProjectedChar(
                    value=value,
                    raw_start=cursor,
                    raw_end=cursor + 1,
                    existing_styles=frozenset(),
                    formula_gap_before=pending_formula_gap and fragment_index == 0,
                    inside_hyperlink=False,
                )
            )
        if fragment:
            pending_formula_gap = False
        cursor += 1
    return projected


def _all_occurrences(content: str, target: str, start: int) -> list[int]:
    """返回 target 在 content 指定位置后的全部精确匹配起点。"""

    output: list[int] = []
    cursor = start
    while target and (match := content.find(target, cursor)) >= 0:
        output.append(match)
        cursor = match + 1
    return output


def _resolve_fallback_occurrence(
    content: str,
    line: PDFTextStyleLine,
    style_range: PDFTextStyleRange,
    start: int,
) -> int | None:
    """在整行无法对齐时，用唯一样式片段及两侧精确上下文选择位置。"""

    target = line.text[style_range.start : style_range.end]
    occurrences = _all_occurrences(content, target, start)
    if not occurrences:
        return None
    left_context = line.text[max(0, style_range.start - 12) : style_range.start]
    right_context = line.text[style_range.end : style_range.end + 12]
    scored = [
        (
            int(bool(left_context) and content[max(0, position - len(left_context)) : position] == left_context)
            + int(
                bool(right_context)
                and content[position + len(target) : position + len(target) + len(right_context)] == right_context
            ),
            position,
        )
        for position in occurrences
    ]
    has_geometric_style = bool(_PDF_GEOMETRIC_TEXT_STYLES.intersection(style_range.styles))
    if len(occurrences) == 1 and (has_geometric_style or len(target) >= 3):
        return occurrences[0]
    best_score = max(score for score, _position in scored)
    best_positions = [position for score, position in scored if score == best_score]
    if has_geometric_style:
        return best_positions[0] if best_score > 0 and len(best_positions) == 1 else None
    required_context_score = int(bool(left_context)) + int(bool(right_context))
    return (
        best_positions[0]
        if required_context_score > 0 and best_score == required_context_score and len(best_positions) == 1
        else None
    )


def _match_line_across_formula_gaps(
    line_text: str,
    projected: Sequence[_ProjectedChar],
    start: int,
) -> _LineProjectionMatch | None:
    """用精确字符序列跨过公式空洞，将一个物理行对齐到 block 文本。"""

    if not line_text or start >= len(projected) or not any(token.formula_gap_before for token in projected[start:]):
        return None
    for projected_start in range(max(0, start), len(projected)):
        first_token = projected[projected_start]
        if not first_token.formula_gap_before and first_token.value != line_text[0]:
            continue
        states: dict[int, tuple[int | None, ...]] = {0: ()}
        for projected_index in range(projected_start, len(projected)):
            token = projected[projected_index]
            next_states: dict[int, tuple[int | None, ...]] = {}
            for source_index, mapping in states.items():
                if source_index >= len(line_text):
                    continue
                if token.formula_gap_before:
                    candidate_source_indices = range(
                        source_index + 1,
                        len(line_text),
                    )
                elif line_text[source_index] == token.value:
                    candidate_source_indices = (source_index,)
                else:
                    continue
                for matched_source_index in candidate_source_indices:
                    if line_text[matched_source_index] != token.value:
                        continue
                    next_source_index = matched_source_index + 1
                    next_mapping = (
                        *mapping,
                        *([None] * (matched_source_index - source_index)),
                        projected_index,
                    )
                    next_states.setdefault(next_source_index, next_mapping)
            complete = next_states.get(len(line_text))
            if complete is not None:
                return _LineProjectionMatch(
                    start=projected_start,
                    end=projected_index + 1,
                    source_to_projected=complete,
                )
            if not next_states:
                break
            states = next_states
    return None


def _ranges_from_line_projection(
    line: PDFTextStyleLine,
    match: _LineProjectionMatch,
) -> list[PDFTextStyleRange]:
    """把物理行样式区间投影为公式字符被跳过后的 block 文本区间。"""

    output: list[PDFTextStyleRange] = []
    for style_range in line.style_ranges:
        current_start: int | None = None
        previous_index: int | None = None
        for projected_index in match.source_to_projected[style_range.start : style_range.end]:
            if projected_index is None:
                if current_start is not None and previous_index is not None:
                    output.append(
                        PDFTextStyleRange(
                            current_start,
                            previous_index + 1,
                            style_range.styles,
                        )
                    )
                current_start = None
                previous_index = None
                continue
            if current_start is not None and previous_index is not None and projected_index != previous_index + 1:
                output.append(
                    PDFTextStyleRange(
                        current_start,
                        previous_index + 1,
                        style_range.styles,
                    )
                )
                current_start = None
            if current_start is None:
                current_start = projected_index
            previous_index = projected_index
        if current_start is not None and previous_index is not None:
            output.append(
                PDFTextStyleRange(
                    current_start,
                    previous_index + 1,
                    style_range.styles,
                )
            )
    return output


def _lines_form_dehyphenated_continuation(
    line: PDFTextEvidenceLine,
    next_line: PDFTextEvidenceLine | None,
) -> bool:
    """判断相邻物理行是否符合正文回填使用的英文断词规则。"""

    return bool(
        next_line is not None
        and next_line.source_index == line.source_index + 1
        and line.text
        and next_line.text
        and is_hyphen_at_line_end(line.text)
        and next_line.text[0].islower()
    )


def _match_line_without_terminal_hyphen(
    projected_text: str,
    line: PDFTextEvidenceLine,
    next_line: PDFTextEvidenceLine | None,
    start: int,
) -> _LineProjectionMatch | None:
    """将 block 已删除的行末断词符映射为空洞，歧义时拒绝匹配。"""

    if not _lines_form_dehyphenated_continuation(line, next_line):
        return None
    candidate = line.text[:-1]
    next_first_char = next_line.text[0] if next_line is not None else ""
    occurrences = [
        position
        for position in _all_occurrences(projected_text, candidate, start)
        if (position + len(candidate) < len(projected_text) and projected_text[position + len(candidate)] == next_first_char)
    ]
    if len(occurrences) != 1:
        logger.debug(f"Skip ambiguous PDF text dehyphenation mapping: line={line.text!r}, occurrences={len(occurrences)}")
        return None
    position = occurrences[0]
    return _LineProjectionMatch(
        start=position,
        end=position + len(candidate),
        source_to_projected=(
            *range(position, position + len(candidate)),
            None,
        ),
    )


def _match_style_ranges(
    projected: Sequence[_ProjectedChar],
    lines: Sequence[PDFTextStyleLine],
) -> list[PDFTextStyleRange]:
    """按物理行顺序把字体与装饰线证据确定性对齐到 block 文本。"""

    projected_text = "".join(token.value for token in projected)
    output: list[PDFTextStyleRange] = []
    cursor = 0
    for line_index, line in enumerate(lines):
        next_line = lines[line_index + 1] if line_index + 1 < len(lines) else None
        line_start = projected_text.find(line.text, cursor)
        if line_start >= 0:
            output.extend(
                PDFTextStyleRange(
                    line_start + style_range.start,
                    line_start + style_range.end,
                    style_range.styles,
                )
                for style_range in line.style_ranges
            )
            cursor = line_start + len(line.text)
            continue
        formula_match = _match_line_across_formula_gaps(
            line.text,
            projected,
            cursor,
        )
        if formula_match is not None:
            output.extend(_ranges_from_line_projection(line, formula_match))
            cursor = formula_match.end
            continue
        dehyphenated_match = _match_line_without_terminal_hyphen(
            projected_text,
            line,
            next_line,
            cursor,
        )
        if dehyphenated_match is not None:
            output.extend(_ranges_from_line_projection(line, dehyphenated_match))
            cursor = dehyphenated_match.end
            continue
        skipped_ranges: list[PDFTextStyleRange] = []
        for style_range in line.style_ranges:
            position = _resolve_fallback_occurrence(
                projected_text,
                line,
                style_range,
                cursor,
            )
            if position is None:
                skipped_ranges.append(style_range)
                continue
            output.append(
                PDFTextStyleRange(
                    position,
                    position + style_range.end - style_range.start,
                    style_range.styles,
                )
            )
            cursor = position + style_range.end - style_range.start
        if skipped_ranges:
            skipped_samples = [
                (
                    line.text[style_range.start : style_range.end],
                    style_range.styles,
                )
                for style_range in skipped_ranges[:3]
            ]
            logger.debug(
                "Skip ambiguous PDF text style mapping: "
                f"line={line.text!r}, skipped={len(skipped_ranges)}, "
                f"samples={skipped_samples!r}"
            )
    return _merge_style_ranges(output)


def _match_script_line_ranges(
    projected: Sequence[_ProjectedChar],
    line: PDFTextStyleLine,
) -> list[PDFTextStyleRange]:
    """独立投影单条脚本行，避免其它视觉行推进 cursor 后吞掉短脚本。"""

    projected_text = "".join(token.value for token in projected)
    exact_occurrences = _all_occurrences(projected_text, line.text, 0)
    if len(exact_occurrences) == 1:
        line_start = exact_occurrences[0]
        return [
            PDFTextStyleRange(
                line_start + style_range.start,
                line_start + style_range.end,
                style_range.styles,
            )
            for style_range in line.style_ranges
        ]
    formula_match = _match_line_across_formula_gaps(
        line.text,
        projected,
        0,
    )
    if formula_match is not None:
        return _ranges_from_line_projection(line, formula_match)
    output: list[PDFTextStyleRange] = []
    for style_range in line.style_ranges:
        position = _resolve_fallback_occurrence(
            projected_text,
            line,
            style_range,
            0,
        )
        if position is None:
            continue
        output.append(
            PDFTextStyleRange(
                position,
                position + style_range.end - style_range.start,
                style_range.styles,
            )
        )
    return _merge_style_ranges(output)


def _merge_style_ranges(ranges: Sequence[PDFTextStyleRange]) -> list[PDFTextStyleRange]:
    """把重叠样式取并集，并合并相邻且样式集合相同的区间。"""

    events: dict[int, dict[PDFTextStyle, int]] = {}
    for style_range in ranges:
        styles = _canonical_styles(style_range.styles)
        if style_range.start >= style_range.end or not styles:
            continue
        for position, delta in (
            (style_range.start, 1),
            (style_range.end, -1),
        ):
            position_events = events.setdefault(position, {})
            for style in styles:
                position_events[style] = position_events.get(style, 0) + delta

    active_counts: dict[PDFTextStyle, int] = {}
    merged: list[PDFTextStyleRange] = []
    previous_position: int | None = None
    for position in sorted(events):
        active_styles = _canonical_styles(style for style, count in active_counts.items() if count > 0)
        if previous_position is not None and previous_position < position and active_styles:
            if merged and merged[-1].end == previous_position and merged[-1].styles == active_styles:
                merged[-1] = PDFTextStyleRange(
                    merged[-1].start,
                    position,
                    active_styles,
                )
            else:
                merged.append(
                    PDFTextStyleRange(
                        previous_position,
                        position,
                        active_styles,
                    )
                )
        for style, delta in events[position].items():
            active_counts[style] = active_counts.get(style, 0) + delta
        previous_position = position
    return merged


def _resolve_link_fallback_occurrence(
    content: str,
    line: PDFTextLinkLine,
    link_range: PDFTextLinkRange,
    start: int,
) -> int | None:
    """整行无法对齐时，用唯一标签或两侧精确上下文定位链接片段。"""

    target_text = line.text[link_range.start : link_range.end]
    occurrences = _all_occurrences(content, target_text, start)
    if not occurrences:
        return None
    if len(occurrences) == 1:
        return occurrences[0]

    left_context = line.text[max(0, link_range.start - 12) : link_range.start]
    right_context = line.text[link_range.end : link_range.end + 12]
    required_context_score = int(bool(left_context)) + int(bool(right_context))
    if required_context_score == 0:
        return None
    scored = [
        (
            int(bool(left_context) and content[max(0, position - len(left_context)) : position] == left_context)
            + int(
                bool(right_context)
                and content[position + len(target_text) : position + len(target_text) + len(right_context)] == right_context
            ),
            position,
        )
        for position in occurrences
    ]
    best_score = max(score for score, _position in scored)
    best_positions = [position for score, position in scored if score == best_score]
    return best_positions[0] if best_score == required_context_score and len(best_positions) == 1 else None


def _project_link_range_from_line_match(
    link_range: PDFTextLinkRange,
    match: _LineProjectionMatch,
    source_index: int,
) -> list[_MatchedLinkRange]:
    """按行字符投影映射链接区间，并在缺失字符处安全分段。"""

    output: list[_MatchedLinkRange] = []
    current_start: int | None = None
    previous_index: int | None = None
    for projected_index in match.source_to_projected[link_range.start : link_range.end]:
        if projected_index is None:
            if current_start is not None and previous_index is not None:
                output.append(
                    _MatchedLinkRange(
                        current_start,
                        previous_index + 1,
                        link_range.target,
                        source_index,
                    )
                )
            current_start = None
            previous_index = None
            continue
        if current_start is not None and previous_index is not None and projected_index != previous_index + 1:
            output.append(
                _MatchedLinkRange(
                    current_start,
                    previous_index + 1,
                    link_range.target,
                    source_index,
                )
            )
            current_start = None
        if current_start is None:
            current_start = projected_index
        previous_index = projected_index
    if current_start is not None and previous_index is not None:
        output.append(
            _MatchedLinkRange(
                current_start,
                previous_index + 1,
                link_range.target,
                source_index,
            )
        )
    return output


def _link_lines_form_dehyphenated_continuation(
    line: PDFTextLinkLine,
    next_line: PDFTextLinkLine | None,
) -> bool:
    """判断相邻同 href 链接行是否符合文本回填的英文断词规则。"""

    if not _lines_form_dehyphenated_continuation(line, next_line):
        return False
    tail_targets = {link_range.target for link_range in line.link_ranges if link_range.start < link_range.end == len(line.text)}
    head_targets = {link_range.target for link_range in next_line.link_ranges if link_range.start == 0 < link_range.end}
    return bool(tail_targets.intersection(head_targets))


def _match_link_line_without_terminal_hyphen(
    projected_text: str,
    line: PDFTextLinkLine,
    next_line: PDFTextLinkLine | None,
    start: int,
) -> _LineProjectionMatch | None:
    """在严格跨行条件下将已被 block 回填删除的行末断词符投影为空洞。"""

    if not _link_lines_form_dehyphenated_continuation(line, next_line):
        return None
    return _match_line_without_terminal_hyphen(
        projected_text,
        line,
        next_line,
        start,
    )


def _merge_matched_link_ranges(
    ranges: Sequence[_MatchedLinkRange],
) -> list[_MatchedLinkRange]:
    """删除不同目标重叠区，并合并同一物理行内的同目标相邻区间。"""

    valid_ranges = [link_range for link_range in ranges if link_range.start < link_range.end and link_range.target]
    if not valid_ranges:
        return []
    boundaries = sorted({position for link_range in valid_ranges for position in (link_range.start, link_range.end)})
    merged: list[_MatchedLinkRange] = []
    for start, end in zip(boundaries, boundaries[1:]):
        active = [link_range for link_range in valid_ranges if link_range.start < end and link_range.end > start]
        targets = {link_range.target for link_range in active}
        if len(targets) != 1:
            continue
        target = next(iter(targets))
        source_index = min(link_range.source_index for link_range in active if link_range.target == target)
        if merged and merged[-1].end == start and merged[-1].target == target and merged[-1].source_index == source_index:
            merged[-1] = _MatchedLinkRange(
                merged[-1].start,
                end,
                target,
                source_index,
            )
        else:
            merged.append(
                _MatchedLinkRange(
                    start,
                    end,
                    target,
                    source_index,
                )
            )
    return merged


def _match_link_ranges(
    projected: Sequence[_ProjectedChar],
    lines: Sequence[PDFTextLinkLine],
) -> list[_MatchedLinkRange]:
    """按物理行顺序把 Link 几何证据确定性对齐到 block 文本。"""

    projected_text = "".join(token.value for token in projected)
    output: list[_MatchedLinkRange] = []
    cursor = 0
    for line_index, line in enumerate(lines):
        next_line = lines[line_index + 1] if line_index + 1 < len(lines) else None
        line_start = projected_text.find(line.text, cursor)
        if line_start >= 0:
            output.extend(
                _MatchedLinkRange(
                    line_start + link_range.start,
                    line_start + link_range.end,
                    link_range.target,
                    line.source_index,
                )
                for link_range in line.link_ranges
            )
            cursor = line_start + len(line.text)
            continue
        formula_match = _match_line_across_formula_gaps(
            line.text,
            projected,
            cursor,
        )
        if formula_match is not None:
            for link_range in line.link_ranges:
                output.extend(
                    _project_link_range_from_line_match(
                        link_range,
                        formula_match,
                        line.source_index,
                    )
                )
            cursor = formula_match.end
            continue

        dehyphenated_match = _match_link_line_without_terminal_hyphen(
            projected_text,
            line,
            next_line,
            cursor,
        )
        if dehyphenated_match is not None:
            for link_range in line.link_ranges:
                output.extend(
                    _project_link_range_from_line_match(
                        link_range,
                        dehyphenated_match,
                        line.source_index,
                    )
                )
            cursor = dehyphenated_match.end
            continue

        skipped_ranges: list[PDFTextLinkRange] = []
        for link_range in line.link_ranges:
            position = _resolve_link_fallback_occurrence(
                projected_text,
                line,
                link_range,
                cursor,
            )
            if position is None:
                skipped_ranges.append(link_range)
                continue
            output.append(
                _MatchedLinkRange(
                    position,
                    position + link_range.end - link_range.start,
                    link_range.target,
                    line.source_index,
                )
            )
            cursor = position + link_range.end - link_range.start
        if skipped_ranges:
            logger.debug(
                "Skip ambiguous PDF hyperlink mapping: "
                f"line={line.text!r}, skipped={len(skipped_ranges)}, "
                f"samples={[(line.text[item.start : item.end], item.target) for item in skipped_ranges[:3]]!r}"
            )
    return _merge_matched_link_ranges(output)


def _append_raw_link_interval(
    intervals: list[_RawLinkInterval],
    start: int | None,
    end: int,
    target: str,
    source_index: int,
) -> None:
    """向结果追加一个合法原字符串链接区间。"""

    if start is not None and start < end and target:
        intervals.append(
            _RawLinkInterval(
                start,
                end,
                target,
                source_index,
            )
        )


def _raw_link_intervals(
    content: str,
    projected: Sequence[_ProjectedChar],
    ranges: Sequence[_MatchedLinkRange],
) -> list[_RawLinkInterval]:
    """把链接区间转换为不跨公式或已有 hyperlink 的原字符串区间。"""

    intervals: list[_RawLinkInterval] = []
    for link_range in ranges:
        current_start: int | None = None
        current_end = 0
        for token in projected[link_range.start : link_range.end]:
            if token.inside_hyperlink or (token.formula_gap_before and current_start is not None):
                _append_raw_link_interval(
                    intervals,
                    current_start,
                    current_end,
                    link_range.target,
                    link_range.source_index,
                )
                current_start = None
            if token.inside_hyperlink:
                continue
            if current_start is None:
                current_start = token.raw_start
                current_end = token.raw_end
                continue
            gap = content[current_end : token.raw_start]
            if token.raw_start <= current_end or not gap or gap.isspace():
                current_end = max(current_end, token.raw_end)
            else:
                _append_raw_link_interval(
                    intervals,
                    current_start,
                    current_end,
                    link_range.target,
                    link_range.source_index,
                )
                current_start = token.raw_start
                current_end = token.raw_end
        _append_raw_link_interval(
            intervals,
            current_start,
            current_end,
            link_range.target,
            link_range.source_index,
        )
    return intervals


def _raw_link_gap_is_boundary_only(gap: str) -> bool:
    """判断两个跨行链接片段之间是否只包含空白或非正文边界符号。"""

    if not gap:
        return True
    if r"\(" in gap or r"\)" in gap:
        return False
    return not any(char.isalnum() for char in html.unescape(gap))


def _merge_raw_link_intervals(
    content: str,
    intervals: Sequence[_RawLinkInterval],
) -> list[_RawLinkInterval]:
    """合并相邻物理行中同 href 的首尾链接片段，不跨越正文或公式。"""

    merged: list[_RawLinkInterval] = []
    for interval in sorted(
        intervals,
        key=lambda item: (item.start, item.end, item.source_index, item.target),
    ):
        if interval.start >= interval.end or not interval.target:
            continue
        if (
            merged
            and merged[-1].target == interval.target
            and interval.source_index == merged[-1].source_index + 1
            and interval.start >= merged[-1].end
            and _raw_link_gap_is_boundary_only(content[merged[-1].end : interval.start])
        ):
            merged[-1] = _RawLinkInterval(
                merged[-1].start,
                interval.end,
                interval.target,
                interval.source_index,
            )
        else:
            merged.append(interval)
    return merged


__all__ = [
    "_resplit_evidence_segments",
    "_partition_resplit_text_evidence",
    "_realign_repaired_text_evidence",
    "_block_bbox_to_page_bbox",
    "_line_block_score",
    "_assign_lines_to_blocks",
    "_assign_script_lines_to_blocks",
    "_filter_line_styles_for_block",
    "_project_content_chars",
    "_all_occurrences",
    "_resolve_fallback_occurrence",
    "_match_line_across_formula_gaps",
    "_ranges_from_line_projection",
    "_lines_form_dehyphenated_continuation",
    "_match_line_without_terminal_hyphen",
    "_match_style_ranges",
    "_match_script_line_ranges",
    "_merge_style_ranges",
    "_resolve_link_fallback_occurrence",
    "_project_link_range_from_line_match",
    "_link_lines_form_dehyphenated_continuation",
    "_match_link_line_without_terminal_hyphen",
    "_merge_matched_link_ranges",
    "_match_link_ranges",
    "_append_raw_link_interval",
    "_raw_link_intervals",
    "_raw_link_gap_is_boundary_only",
    "_merge_raw_link_intervals",
]
