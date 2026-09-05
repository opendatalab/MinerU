# Copyright (c) Opendatalab. All rights reserved.
"""按公式区域和字符几何识别上下标证据。"""

from __future__ import annotations

import statistics
import unicodedata
from typing import Any, Literal, Sequence

from .....types import BBox
from ..geometry import _rotate_bbox_to_upright
from ..script_geometry import ScriptRole, classify_char_script_roles
from .common import _coerce_bbox, _normalize_match_fragment, _ordered_line_chars
from .types import (
    _PDF_SCRIPT_AUTHOR_MARKS,
    _PDF_SCRIPT_CITATION_BRACKETS,
    _PDF_SCRIPT_COMPACT_JOINERS,
    _PDF_SCRIPT_MATH_BASE_CHARS,
    _PDF_SCRIPT_SIGN_CHARS,
    _PDF_SCRIPT_SPACED_OPERATORS,
    _PDF_SCRIPT_TOKEN_CONNECTORS,
    _PDF_SCRIPT_TRAILING_MARKS,
    PDFTextScriptLine,
    PDFTextScriptRange,
)


def _rotate_origin_to_upright(
    origin: tuple[float, float],
    page_size: tuple[float, float],
    angle: int,
) -> tuple[float, float]:
    """把页面字符 origin 旋到当前 Flash 行的局部正向坐标。"""
    x, y = origin
    page_width, page_height = page_size
    if angle == 270:
        return page_height - y, x
    if angle == 90:
        return y, page_width - x
    if angle == 180:
        return page_width - x, page_height - y
    return origin


def _bbox_center_inside_region(bbox: BBox, region: BBox) -> bool:
    """判断字符 tight bbox 中心是否落入公式区域。"""
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3]


def _script_region_memberships(
    chars: list[dict[str, Any]],
    tight_bboxes: dict[int, BBox],
    regions: list[BBox],
) -> list[int | None]:
    """按页面 tight 中心把字符分配到公式区域，区域外返回 None。"""
    memberships: list[int | None] = []
    for char in chars:
        char_idx = char.get("char_idx")
        tight_bbox = tight_bboxes.get(char_idx) if isinstance(char_idx, int) else None
        region_index = None
        if tight_bbox is not None:
            region_index = next(
                (index for index, region in enumerate(regions) if _bbox_center_inside_region(tight_bbox, region)),
                None,
            )
        memberships.append(region_index)
    return memberships


def _script_char_text(char: dict[str, Any]) -> str:
    """返回单字符脚本判定使用的稳定文本。"""
    return str(char.get("char", ""))


def _is_cjk_text(text: str) -> bool:
    """判断单字符是否属于 CJK、日文假名或韩文书写系统。"""
    if len(text) != 1:
        return False
    codepoint = ord(text)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    )


def _is_math_identifier_char(text: str) -> bool:
    """识别可与拉丁 base/index 共同组成数学 token 的字母数字字符。"""
    if len(text) != 1 or _is_cjk_text(text):
        return False
    if text.isascii():
        return text.isalnum()
    category = unicodedata.category(text)
    unicode_name = unicodedata.name(text, "")
    return (
        text in _PDF_SCRIPT_MATH_BASE_CHARS
        or "GREEK" in unicode_name
        or "MATHEMATICAL" in unicode_name
        or category in {"Lu", "Ll", "Lm"}
    )


def _is_math_script_token_char(text: str) -> bool:
    """判断字符是否属于可按 source order 重新锚定的数学 token。"""
    return _is_math_identifier_char(text) or text in _PDF_SCRIPT_TOKEN_CONNECTORS


def _iter_math_script_tokens(chars: list[dict[str, Any]]) -> list[list[int]]:
    """按连续数学 identifier 和连接符切分局部 token，并在 CJK 边界断开。"""
    tokens: list[list[int]] = []
    current: list[int] = []
    for index, char in enumerate(chars):
        if _is_math_script_token_char(_script_char_text(char)):
            current.append(index)
            continue
        if current:
            tokens.append(current)
            current = []
    if current:
        tokens.append(current)
    return tokens


def _citation_script_indices(chars: list[dict[str, Any]], roles: list[ScriptRole]) -> set[int]:
    """识别方括号引用区间，避免保守 token 规则删除数字引用。"""
    protected: set[int] = set()
    for start, char in enumerate(chars):
        closing = _PDF_SCRIPT_CITATION_BRACKETS.get(_script_char_text(char))
        if closing is None:
            continue
        for end in range(start + 1, min(len(chars), start + 16)):
            if _script_char_text(chars[end]) != closing:
                continue
            if any(roles[index] != "body" and _script_char_text(chars[index]).isalnum() for index in range(start + 1, end)):
                protected.update(range(start, end + 1))
            break
    return protected


def _token_origin(
    char: dict[str, Any],
    origins: dict[int, tuple[float, float]],
) -> float | None:
    """读取 token 字符的局部正向 origin y。"""
    char_idx = char.get("char_idx")
    origin = origins.get(char_idx) if isinstance(char_idx, int) else None
    return float(origin[1]) if origin is not None else None


def _token_tight_height(
    char: dict[str, Any],
    tight_bboxes: dict[int, BBox],
) -> float:
    """读取 token 字符的局部正向 tight 高度。"""
    char_idx = char.get("char_idx")
    bbox = tight_bboxes.get(char_idx) if isinstance(char_idx, int) else None
    return max(0.0, bbox[3] - bbox[1]) if bbox is not None else 0.0


def _has_adjacent_math_base(
    chars: list[dict[str, Any]],
    index: int,
    roles: list[ScriptRole],
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
) -> bool:
    """判断孤立索引左侧是否存在紧邻且位移明确的非 CJK 数学 base。"""
    if index <= 0 or roles[index - 1] != "body":
        return False
    base_text = _script_char_text(chars[index - 1])
    if not _is_math_identifier_char(base_text):
        return False
    base_origin = _token_origin(chars[index - 1], origins)
    script_origin = _token_origin(chars[index], origins)
    base_height = _token_tight_height(chars[index - 1], tight_bboxes)
    base_idx = chars[index - 1].get("char_idx")
    script_idx = chars[index].get("char_idx")
    base_bbox = tight_bboxes.get(base_idx) if isinstance(base_idx, int) else None
    script_bbox = tight_bboxes.get(script_idx) if isinstance(script_idx, int) else None
    if base_origin is None or script_origin is None or base_bbox is None or script_bbox is None:
        return False
    return (
        _bbox_axis_overlap(base_bbox, script_bbox, axis="y") > 0
        or _horizontal_gap_between_bboxes(base_bbox, script_bbox) <= max(2.0, 0.5 * base_height)
    ) and abs(script_origin - base_origin) >= max(0.35, 0.08 * base_height)


def _horizontal_gap_between_bboxes(first: BBox, second: BBox) -> float:
    """返回两个 tight bbox 的水平间隙。"""
    return max(0.0, first[0] - second[2], second[0] - first[2])


def _token_split_position(
    chars: list[dict[str, Any]],
    token: list[int],
    roles: list[ScriptRole],
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
) -> int | None:
    """用最左 origin 簇和显式连接符确定 base 与索引的分界。"""
    alnum_positions = [index for index in token if _is_math_identifier_char(_script_char_text(chars[index]))]
    if len(alnum_positions) < 2:
        return None
    first = alnum_positions[0]
    first_origin = _token_origin(chars[first], origins)
    first_height = _token_tight_height(chars[first], tight_bboxes)
    origin_tolerance = max(0.35, 0.06 * first_height)
    leading_connectors = [
        index for index in token if index < first and _script_char_text(chars[index]) in _PDF_SCRIPT_TOKEN_CONNECTORS
    ]
    if leading_connectors:
        return alnum_positions[1]
    for position in alnum_positions[1:]:
        if any(_script_char_text(chars[index]) in _PDF_SCRIPT_TOKEN_CONNECTORS for index in range(first + 1, position)):
            return position
        origin = _token_origin(chars[position], origins)
        if first_origin is not None and origin is not None and abs(origin - first_origin) > origin_tolerance:
            return position
    scripted_positions = [position for position in alnum_positions[1:] if roles[position] != "body"]
    if roles[first] == "body" and scripted_positions:
        return scripted_positions[0]
    return None


def _script_geometry_is_aligned(
    chars: list[dict[str, Any]],
    first: int,
    second: int,
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
) -> bool:
    """判断两个字符是否处在同一 displaced baseline 上。"""
    first_origin = _token_origin(chars[first], origins)
    second_origin = _token_origin(chars[second], origins)
    first_height = _token_tight_height(chars[first], tight_bboxes)
    second_height = _token_tight_height(chars[second], tight_bboxes)
    first_idx = chars[first].get("char_idx")
    second_idx = chars[second].get("char_idx")
    first_bbox = tight_bboxes.get(first_idx) if isinstance(first_idx, int) else None
    second_bbox = tight_bboxes.get(second_idx) if isinstance(second_idx, int) else None
    if first_origin is None or second_origin is None or first_bbox is None or second_bbox is None:
        return False
    scale = max(first_height, second_height, 1.0)
    first_center = (first_bbox[1] + first_bbox[3]) / 2
    second_center = (second_bbox[1] + second_bbox[3]) / 2
    return abs(first_origin - second_origin) <= max(0.35, 0.06 * scale) and abs(first_center - second_center) <= max(
        0.75,
        0.3 * scale,
    )


def _nearest_nonspace_index(
    chars: list[dict[str, Any]],
    start: int,
    step: Literal[-1, 1],
) -> int | None:
    """从指定位置向前或向后查找最近的非空白字符。"""
    index = start + step
    while 0 <= index < len(chars):
        if not _script_char_text(chars[index]).isspace():
            return index
        index += step
    return None


def _close_spaced_script_operators(
    chars: list[dict[str, Any]],
    roles: list[ScriptRole],
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
) -> None:
    """跨少量 PDF 空格闭合同基线的 `1 - x` 一类角标 run。"""
    for seed, role in enumerate(list(roles)):
        if role == "body" or not _is_math_identifier_char(_script_char_text(chars[seed])):
            continue
        operator_index = _nearest_nonspace_index(chars, seed, 1)
        if operator_index is None or operator_index - seed > 3:
            continue
        if _script_char_text(chars[operator_index]) not in _PDF_SCRIPT_SPACED_OPERATORS:
            continue
        target = _nearest_nonspace_index(chars, operator_index, 1)
        if target is None or target - operator_index > 3:
            continue
        if not _is_math_identifier_char(_script_char_text(chars[target])):
            continue
        if not _script_geometry_is_aligned(chars, seed, operator_index, tight_bboxes, origins):
            continue
        if not _script_geometry_is_aligned(chars, seed, target, tight_bboxes, origins):
            continue
        roles[operator_index] = role
        roles[target] = role


def _close_compact_aligned_script_suffixes(
    chars: list[dict[str, Any]],
    raw_roles: list[ScriptRole],
    refined_roles: list[ScriptRole],
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
) -> None:
    """把已有可信角标 run 后同基线的紧凑连字符后缀整体闭合。"""
    for joiner_index in range(1, len(chars) - 1):
        if _script_char_text(chars[joiner_index]) not in _PDF_SCRIPT_COMPACT_JOINERS:
            continue
        left_seed = joiner_index - 1
        role = refined_roles[left_seed]
        if role == "body" or not _is_math_identifier_char(_script_char_text(chars[left_seed])):
            continue

        left_start = left_seed
        while (
            left_start > 0
            and refined_roles[left_start - 1] == role
            and _is_math_identifier_char(_script_char_text(chars[left_start - 1]))
        ):
            left_start -= 1
        if left_seed - left_start + 1 < 2:
            continue
        anchor_index = left_start - 1
        if (
            anchor_index < 0
            or refined_roles[anchor_index] != "body"
            or not _is_math_identifier_char(_script_char_text(chars[anchor_index]))
        ):
            continue

        suffix_start = joiner_index + 1
        suffix_end = suffix_start
        while suffix_end < len(chars) and _is_math_identifier_char(_script_char_text(chars[suffix_end])):
            suffix_end += 1
        if suffix_end - suffix_start < 2:
            continue
        restored_indices = range(joiner_index, suffix_end)
        if any(raw_roles[index] != role for index in restored_indices):
            continue
        if not _script_geometry_is_aligned(chars, left_seed, joiner_index, tight_bboxes, origins):
            continue
        if any(
            not _script_geometry_is_aligned(chars, left_seed, index, tight_bboxes, origins)
            for index in range(suffix_start, suffix_end)
        ):
            continue
        refined_roles[joiner_index:suffix_end] = [role] * (suffix_end - joiner_index)


def _protected_subscript_indices(
    chars: list[dict[str, Any]],
    roles: list[ScriptRole],
    tokens: list[list[int]],
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
) -> set[int]:
    """找出拥有内部 base 或与其同基线连通的下标字符。"""
    protected: set[int] = set()
    for token in tokens:
        for position, index in enumerate(token):
            if roles[index] != "sub" or not _is_math_identifier_char(_script_char_text(chars[index])):
                continue
            if any(
                earlier < index and roles[earlier] == "body" and _is_math_identifier_char(_script_char_text(chars[earlier]))
                for earlier in token[:position]
            ) or _has_adjacent_math_base(chars, index, roles, tight_bboxes, origins):
                protected.add(index)
    changed = True
    while changed:
        changed = False
        for index, role in enumerate(roles):
            if role != "sub" or index in protected or not _is_math_identifier_char(_script_char_text(chars[index])):
                continue
            for seed in tuple(protected):
                start, end = sorted((seed, index))
                if end - start > 5 or not _script_geometry_is_aligned(chars, seed, index, tight_bboxes, origins):
                    continue
                if all(
                    _script_char_text(chars[bridge]).isspace()
                    or _script_char_text(chars[bridge]) in _PDF_SCRIPT_TOKEN_CONNECTORS
                    or _script_char_text(chars[bridge]) in _PDF_SCRIPT_SIGN_CHARS
                    or _script_char_text(chars[bridge]) == "."
                    for bridge in range(start + 1, end)
                ):
                    protected.add(index)
                    changed = True
                    break
    return protected


def _refine_math_script_tokens(
    chars: list[dict[str, Any]],
    roles: list[ScriptRole],
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
    *,
    formula_region: bool,
) -> list[ScriptRole]:
    """以最左稳定簇保护 base，并对弱单字符和复杂未分段 token 保守拒识。"""
    refined = list(roles)
    citation_indices = _citation_script_indices(chars, refined)
    complex_unsegmented_token = False
    tokens = _iter_math_script_tokens(chars)
    token_alnum_positions = {
        tuple(token): [index for index in token if _is_math_identifier_char(_script_char_text(chars[index]))]
        for token in tokens
    }
    token_splits = {
        tuple(token): _token_split_position(
            chars,
            token,
            refined,
            tight_bboxes,
            origins,
        )
        for token in tokens
    }
    token_families: dict[str, list[tuple[int, ...]]] = {}
    for token in tokens:
        key = tuple(token)
        alnum_positions = token_alnum_positions[key]
        if len(alnum_positions) >= 2:
            token_families.setdefault(_script_char_text(chars[alnum_positions[0]]), []).append(key)
    trusted_family_bases = {
        base
        for base, members in token_families.items()
        if len(members) >= 3
        or any(token_splits[member] is not None for member in members)
        or any(any(_script_char_text(chars[index]) in _PDF_SCRIPT_TOKEN_CONNECTORS for index in member) for member in members)
    }
    for token in tokens:
        if any(index in citation_indices for index in token):
            continue
        token_key = tuple(token)
        alnum_positions = token_alnum_positions[token_key]
        if not alnum_positions or not any(refined[index] != "body" for index in token):
            continue
        token_roles = {refined[index] for index in token if refined[index] != "body"}
        if token_roles == {"sup", "sub"}:
            complex_unsegmented_token = True
        if len(alnum_positions) == 1:
            continue
        first_position = alnum_positions[0]
        suffix_positions = alnum_positions[1:]
        if (
            refined[first_position] == "sup"
            and all(refined[index] == "body" for index in suffix_positions)
            and len(suffix_positions) >= 2
            and all(_script_char_text(chars[index]).isalpha() for index in suffix_positions)
        ):
            continue
        split_position = token_splits[token_key]
        if split_position is None and _script_char_text(chars[alnum_positions[0]]) in trusted_family_bases:
            split_position = alnum_positions[1]
        if split_position is None:
            if all(refined[index] != "body" for index in alnum_positions):
                for index in token:
                    refined[index] = "body"
            continue
        base_positions = [index for index in alnum_positions if index < split_position]
        base_origins = [origin for index in base_positions if (origin := _token_origin(chars[index], origins)) is not None]
        base_heights = [_token_tight_height(chars[index], tight_bboxes) for index in base_positions]
        base_origin = statistics.median(base_origins) if base_origins else None
        base_height = statistics.median([height for height in base_heights if height > 0]) if any(base_heights) else 0.0
        for index in token:
            if index < split_position or _script_char_text(chars[index]) in _PDF_SCRIPT_TOKEN_CONNECTORS:
                refined[index] = "body"
                continue
            text = _script_char_text(chars[index])
            if not _is_math_identifier_char(text) or refined[index] != "body":
                continue
            origin = _token_origin(chars[index], origins)
            if base_origin is None or origin is None:
                continue
            shift = origin - base_origin
            if abs(shift) >= max(0.35, 0.08 * base_height):
                refined[index] = "sub" if shift > 0 else "sup"
    _close_spaced_script_operators(
        chars,
        refined,
        tight_bboxes,
        origins,
    )
    if complex_unsegmented_token and not formula_region:
        for index in range(len(refined)):
            if index not in citation_indices:
                refined[index] = "body"
    scripted_alnum = [
        index for index, role in enumerate(refined) if role != "body" and _script_char_text(chars[index]).isalnum()
    ]
    if not formula_region and len(scripted_alnum) >= 2 and any(_script_char_text(char) in {"∑", "∫"} for char in chars):
        for index in range(len(refined)):
            if index not in citation_indices:
                refined[index] = "body"
    has_compact_multiply = any(
        _script_char_text(char) == "×"
        and 0 < index < len(chars) - 1
        and not _script_char_text(chars[index - 1]).isspace()
        and not _script_char_text(chars[index + 1]).isspace()
        for index, char in enumerate(chars)
    )
    if not formula_region and len(scripted_alnum) >= 2 and has_compact_multiply:
        for index in range(len(refined)):
            if index not in citation_indices:
                refined[index] = "body"
    if not formula_region:
        for operator_index, char in enumerate(chars):
            operator = _script_char_text(char)
            nearby = [candidate for candidate in scripted_alnum if abs(candidate - operator_index) <= 5]
            if (
                operator in {"/", "⁄"}
                and any(candidate < operator_index for candidate in nearby)
                and any(candidate > operator_index for candidate in nearby)
            ):
                for candidate in nearby:
                    if candidate not in citation_indices:
                        refined[candidate] = "body"
    protected_subscripts = _protected_subscript_indices(
        chars,
        refined,
        tokens,
        tight_bboxes,
        origins,
    )
    for index, role in enumerate(list(refined)):
        if (
            role == "sub"
            and index not in citation_indices
            and index not in protected_subscripts
            and _is_math_identifier_char(_script_char_text(chars[index]))
        ):
            refined[index] = "body"
    for index, role in enumerate(list(refined)):
        if role == "body" or index in citation_indices:
            continue
        text = _script_char_text(chars[index])
        if text.isalnum() or text in _PDF_SCRIPT_AUTHOR_MARKS:
            continue
        if text in {",", "，"} and all(
            0 <= neighbor < len(refined)
            and refined[neighbor] == role
            and (_script_char_text(chars[neighbor]).isalnum() or _script_char_text(chars[neighbor]) in _PDF_SCRIPT_AUTHOR_MARKS)
            for neighbor in (index - 1, index + 1)
        ):
            continue
        if text == "." and all(
            0 <= neighbor < len(refined) and refined[neighbor] == role and _script_char_text(chars[neighbor]).isdigit()
            for neighbor in (index - 1, index + 1)
        ):
            continue
        if text in _PDF_SCRIPT_SIGN_CHARS:
            sign_neighbors = [
                neighbor
                for step in (-1, 1)
                if (neighbor := _nearest_nonspace_index(chars, index, step)) is not None
                and abs(neighbor - index) <= 3
                and refined[neighbor] == role
            ]
            if sign_neighbors and any(_script_char_text(chars[neighbor]).isdigit() for neighbor in sign_neighbors):
                continue
        if text in _PDF_SCRIPT_TRAILING_MARKS:
            previous = _nearest_nonspace_index(chars, index, -1)
            body_prefix = previous - 1 if previous is not None else -1
            if (
                role == "sup"
                and previous is not None
                and index - previous == 1
                and body_prefix >= 0
                and refined[body_prefix] == "body"
                and _script_char_text(chars[body_prefix]).isalpha()
                and refined[previous] == role
                and roles[index] == role
                and _script_char_text(chars[previous]).isalpha()
                and _script_geometry_is_aligned(chars, previous, index, tight_bboxes, origins)
            ):
                continue
        refined[index] = "body"
    if not formula_region:
        _close_compact_aligned_script_suffixes(
            chars,
            roles,
            refined,
            tight_bboxes,
            origins,
        )
    return refined


def _bbox_axis_overlap(first: BBox, second: BBox, *, axis: Literal["x", "y"]) -> float:
    """返回两个 bbox 在指定轴上的绝对重叠长度。"""
    start, end = (0, 2) if axis == "x" else (1, 3)
    return max(0.0, min(first[end], second[end]) - max(first[start], second[start]))


def _fraction_member_indices(
    page_size: tuple[float, float],
    all_chars: list[dict[str, Any]],
    tight_bboxes: dict[int, BBox],
    drawing_lines: Sequence[Any],
    angle: int,
) -> set[int]:
    """按页面方向一次识别分数线两侧的上下叠字，供复杂分式整块拒识。"""
    if not all_chars or not drawing_lines:
        return set()
    local_chars: list[tuple[int, str, BBox]] = []
    local_heights = []
    for char in all_chars:
        char_idx = char.get("char_idx")
        text = _script_char_text(char)
        bbox = tight_bboxes.get(char_idx) if isinstance(char_idx, int) else None
        if not isinstance(char_idx, int) or bbox is None or not text.isprintable() or text.isspace():
            continue
        local_bbox = _rotate_bbox_to_upright(bbox, page_size, angle)
        local_chars.append((char_idx, text, local_bbox))
        local_heights.append(local_bbox[3] - local_bbox[1])
    scale = statistics.median([height for height in local_heights if height > 0]) if local_heights else 8.0
    members: set[int] = set()
    for drawing in drawing_lines:
        raw_bbox = _coerce_bbox(getattr(drawing, "bbox", drawing))
        if raw_bbox is None:
            continue
        local_rule = _rotate_bbox_to_upright(raw_bbox, page_size, angle)
        width = local_rule[2] - local_rule[0]
        height = local_rule[3] - local_rule[1]
        if width < max(2.0, 0.45 * scale) or width > 12.0 * scale or height > max(1.25, 0.25 * scale):
            continue
        rule_y = (local_rule[1] + local_rule[3]) / 2
        aligned = [
            (char_idx, bbox)
            for char_idx, text, bbox in local_chars
            if text.isalnum()
            and abs((bbox[1] + bbox[3]) / 2 - rule_y) <= 2.25 * scale
            and (
                _bbox_axis_overlap(bbox, local_rule, axis="x") > 0
                or local_rule[0] - 0.25 * scale <= (bbox[0] + bbox[2]) / 2 <= local_rule[2] + 0.25 * scale
            )
        ]
        above = [
            (char_idx, bbox)
            for char_idx, bbox in aligned
            if bbox[3] <= rule_y + 0.2 * scale and rule_y - bbox[3] <= 1.75 * scale
        ]
        below = [
            (char_idx, bbox)
            for char_idx, bbox in aligned
            if bbox[1] >= rule_y - 0.2 * scale and bbox[1] - rule_y <= 1.75 * scale
        ]
        # 超过局部公式尺度的长横线更像脚注/段落分隔线，不用于分式成员抑制。
        if width > 8.0 * scale:
            continue
        if above and below:
            members.update(char_idx for char_idx, _bbox in above)
            members.update(char_idx for char_idx, _bbox in below)
    return members


def _strong_structural_script_roles(
    chars: list[dict[str, Any]],
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
) -> dict[int, ScriptRole]:
    """提取可在恢复公式区域中保留的引用和邻接 base 强脚本证据。"""

    roles = classify_char_script_roles(
        chars,
        tight_bboxes=tight_bboxes,
        origins=origins,
    )
    strong_roles: dict[int, ScriptRole] = {}
    for index in _citation_script_indices(chars, roles):
        if roles[index] != "body":
            strong_roles[index] = roles[index]
    for index, role in enumerate(roles):
        if role != "sup" or not _is_math_identifier_char(_script_char_text(chars[index])):
            continue
        base_height = _token_tight_height(chars[index - 1], tight_bboxes) if index > 0 else 0.0
        script_height = _token_tight_height(chars[index], tight_bboxes)
        if base_height <= 0 or script_height > 0.8 * base_height:
            continue
        next_index = _nearest_nonspace_index(chars, index, 1)
        if next_index is not None and _is_math_script_token_char(_script_char_text(chars[next_index])):
            continue
        if _has_adjacent_math_base(
            chars,
            index,
            roles,
            tight_bboxes,
            origins,
        ):
            strong_roles[index] = role
    return strong_roles


def _classify_script_runs(
    chars: list[dict[str, Any]],
    local_tight_bboxes: dict[int, BBox],
    local_origins: dict[int, tuple[float, float]],
    memberships: list[int | None],
) -> tuple[list[str], list[int], list[bool]]:
    """按公式区域边界分段分类，并要求公式段内部存在稳定 body。"""
    roles: list[ScriptRole] = ["body"] * len(chars)
    body_counts = [0] * len(chars)
    formula_flags = [False] * len(chars)
    start = 0
    while start < len(chars):
        membership = memberships[start]
        end = start + 1
        while end < len(chars) and memberships[end] == membership:
            end += 1
        run_chars = chars[start:end]
        run_indices = {int(char["char_idx"]) for char in run_chars if isinstance(char.get("char_idx"), int)}
        run_roles = classify_char_script_roles(
            run_chars,
            tight_bboxes={index: local_tight_bboxes[index] for index in run_indices if index in local_tight_bboxes},
            origins={index: local_origins[index] for index in run_indices if index in local_origins},
        )
        run_roles = _refine_math_script_tokens(
            run_chars,
            run_roles,
            local_tight_bboxes,
            local_origins,
            formula_region=membership is not None,
        )
        visible = [
            index
            for index, char in enumerate(run_chars)
            if str(char.get("char", "")).isprintable() and not str(char.get("char", "")).isspace()
        ]
        body_count = sum(run_roles[index] == "body" and str(run_chars[index].get("char", "")).isalnum() for index in visible)
        marked_count = sum(run_roles[index] != "body" for index in visible)
        body_tight_heights = [
            local_tight_bboxes[int(run_chars[index]["char_idx"])][3] - local_tight_bboxes[int(run_chars[index]["char_idx"])][1]
            for index in visible
            if run_roles[index] == "body"
            and isinstance(run_chars[index].get("char_idx"), int)
            and int(run_chars[index]["char_idx"]) in local_tight_bboxes
        ]
        script_tight_heights = [
            local_tight_bboxes[int(run_chars[index]["char_idx"])][3] - local_tight_bboxes[int(run_chars[index]["char_idx"])][1]
            for index in visible
            if run_roles[index] != "body"
            and isinstance(run_chars[index].get("char_idx"), int)
            and int(run_chars[index]["char_idx"]) in local_tight_bboxes
        ]
        stable_formula_body = (
            body_count > 0
            and bool(body_tight_heights)
            and (not script_tight_heights or max(body_tight_heights) >= 1.1 * max(script_tight_heights))
        )
        if membership is not None and (not stable_formula_body or marked_count >= len(visible)):
            run_roles = ["body"] * len(run_chars)
        for offset, role in enumerate(run_roles, start=start):
            roles[offset] = role
            body_counts[offset] = body_count
            formula_flags[offset] = membership is not None
        start = end
    return roles, body_counts, formula_flags


def _script_line_char_roles(
    line: Any,
    page_size: tuple[float, float],
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
    fraction_members: set[int],
) -> tuple[list[dict[str, Any]], list[ScriptRole], list[int], list[bool]]:
    """按正文同款公式分段返回原字符及其上下标角色。"""

    chars = _ordered_line_chars(line)
    if not chars:
        return [], [], [], []
    angle = int(getattr(line, "angle", 0) or 0) % 360
    local_chars: list[dict[str, Any]] = []
    local_tight_bboxes: dict[int, BBox] = {}
    local_origins: dict[int, tuple[float, float]] = {}
    for char in chars:
        local_char = dict(char)
        bbox = _coerce_bbox(char.get("bbox"))
        if bbox is not None:
            local_char["bbox"] = _rotate_bbox_to_upright(bbox, page_size, angle)
        local_chars.append(local_char)
        char_idx = char.get("char_idx")
        if not isinstance(char_idx, int):
            continue
        tight_bbox = tight_bboxes.get(char_idx)
        if tight_bbox is not None:
            local_tight_bboxes[char_idx] = _rotate_bbox_to_upright(
                tight_bbox,
                page_size,
                angle,
            )
        origin = origins.get(char_idx)
        if origin is not None:
            local_origins[char_idx] = _rotate_origin_to_upright(
                origin,
                page_size,
                angle,
            )
    regions = [bbox for value in getattr(line, "inline_math_regions", []) if (bbox := _coerce_bbox(value)) is not None]
    memberships = _script_region_memberships(chars, tight_bboxes, regions)
    roles, body_counts, formula_flags = _classify_script_runs(
        local_chars,
        local_tight_bboxes,
        local_origins,
        memberships,
    )
    if bool(getattr(line, "compact_formula_cluster", False)) or (
        bool(getattr(line, "restored_inline_cluster", False)) and bool(regions)
    ):
        strong_structural_roles = _strong_structural_script_roles(
            local_chars,
            local_tight_bboxes,
            local_origins,
        )
        roles = [strong_structural_roles.get(index, "body") for index in range(len(roles))]
    for index, char in enumerate(chars):
        char_idx = char.get("char_idx")
        if isinstance(char_idx, int) and char_idx in fraction_members:
            roles[index] = "body"
    return chars, roles, body_counts, formula_flags


def _script_line_payload(
    line: Any,
    page_size: tuple[float, float],
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
    fraction_members: set[int],
) -> PDFTextScriptLine | None:
    """把 Flash 行转换为公式分段后的紧凑上下标 sidecar。"""

    chars, roles, body_counts, formula_flags = _script_line_char_roles(
        line,
        page_size,
        tight_bboxes,
        origins,
        fraction_members,
    )
    if not chars:
        return None
    angle = int(getattr(line, "angle", 0) or 0) % 360
    compact_parts: list[str] = []
    compact_roles: list[str] = []
    compact_bboxes: list[BBox | None] = []
    compact_body_counts: list[int] = []
    compact_formula_flags: list[bool] = []
    for index, char in enumerate(chars):
        fragment = _normalize_match_fragment(char.get("char"))
        if not fragment:
            continue
        compact_parts.append(fragment)
        char_idx = char.get("char_idx")
        page_tight_bbox = tight_bboxes.get(char_idx) if isinstance(char_idx, int) else None
        compact_roles.extend([roles[index]] * len(fragment))
        compact_bboxes.extend([page_tight_bbox] * len(fragment))
        compact_body_counts.extend([body_counts[index]] * len(fragment))
        compact_formula_flags.extend([formula_flags[index]] * len(fragment))
    text = "".join(compact_parts)
    if not text:
        return None
    ranges: list[PDFTextScriptRange] = []
    start = 0
    while start < len(compact_roles):
        role = compact_roles[start]
        end = start + 1
        while end < len(compact_roles) and compact_roles[end] == role:
            end += 1
        if role in {"sup", "sub"}:
            range_bboxes = [bbox for bbox in compact_bboxes[start:end] if bbox is not None]
            if range_bboxes:
                page_bbox = (
                    min(bbox[0] for bbox in range_bboxes),
                    min(bbox[1] for bbox in range_bboxes),
                    max(bbox[2] for bbox in range_bboxes),
                    max(bbox[3] for bbox in range_bboxes),
                )
                ranges.append(
                    PDFTextScriptRange(
                        start=start,
                        end=end,
                        style="superscript" if role == "sup" else "subscript",
                        bbox=page_bbox,
                        stable_body_count=max(compact_body_counts[start:end], default=0),
                        formula_region=any(compact_formula_flags[start:end]),
                    )
                )
        start = end
    return PDFTextScriptLine(
        bbox=getattr(line, "bbox"),
        text=text,
        script_ranges=tuple(ranges),
        source_index=int(getattr(line, "source_index", 0) or 0),
        angle=angle,
    )


def detect_pdf_text_script_lines(
    lines: list[Any],
    page_size: tuple[float, float],
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
    *,
    all_chars: list[dict[str, Any]] | None = None,
    drawing_lines: Sequence[Any] | None = None,
) -> list[PDFTextScriptLine]:
    """检测 Flash 剩余自然文本行中的上下标候选。"""
    resolved_chars = all_chars or []
    resolved_drawings = drawing_lines or []
    fraction_members_by_angle = {
        angle: _fraction_member_indices(
            page_size,
            resolved_chars,
            tight_bboxes,
            resolved_drawings,
            angle,
        )
        for angle in {int(getattr(line, "angle", 0) or 0) % 360 for line in lines}
    }
    return [
        payload
        for line in lines
        if (
            payload := _script_line_payload(
                line,
                page_size,
                tight_bboxes,
                origins,
                fraction_members_by_angle[int(getattr(line, "angle", 0) or 0) % 360],
            )
        )
        is not None
    ]


__all__ = [
    "_rotate_origin_to_upright",
    "_bbox_center_inside_region",
    "_script_region_memberships",
    "_script_char_text",
    "_is_cjk_text",
    "_is_math_identifier_char",
    "_is_math_script_token_char",
    "_iter_math_script_tokens",
    "_citation_script_indices",
    "_token_origin",
    "_token_tight_height",
    "_has_adjacent_math_base",
    "_horizontal_gap_between_bboxes",
    "_token_split_position",
    "_script_geometry_is_aligned",
    "_nearest_nonspace_index",
    "_close_spaced_script_operators",
    "_close_compact_aligned_script_suffixes",
    "_protected_subscript_indices",
    "_refine_math_script_tokens",
    "_bbox_axis_overlap",
    "_fraction_member_indices",
    "_strong_structural_script_roles",
    "_classify_script_runs",
    "_script_line_char_roles",
    "_script_line_payload",
    "detect_pdf_text_script_lines",
]
