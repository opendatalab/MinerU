# Copyright (c) Opendatalab. All rights reserved.
import math
from typing import Any, List

import numpy as np
import pypdfium2 as pdfium
from pdftext.pdf.chars import deduplicate_chars, get_chars
from pdftext.pdf.pages import assign_scripts, get_blocks, get_lines, get_spans
from pdftext.schema import Bbox

from mineru.utils.pdfium_guard import close_pdfium_child, pdfium_guard

try:
    from pdftext.pdf.chars import PageChars
except ImportError:
    PageChars = None

NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE = 1.0
OFFSET_DUPLICATE_CHAR_BBOX_TOLERANCE = 2.5
OFFSET_DUPLICATE_TRANSLATION_TOLERANCE = 0.1
OFFSET_DUPLICATE_MIN_BBOX_OVERLAP_RATIO = 0.45


def get_page(
    page: pdfium.PdfPage,
    quote_loosebox: bool = True,
    superscript_height_threshold: float = 0.7,
    line_distance_threshold: float = 0.1,
) -> dict:
    page_chars = get_page_chars(page, quote_loosebox=quote_loosebox)
    lines = get_lines_from_chars(
        page_chars["chars"],
        superscript_height_threshold=superscript_height_threshold,
        line_distance_threshold=line_distance_threshold,
    )
    blocks = get_blocks(lines)

    return {
        "bbox": page_chars["bbox"],
        "width": page_chars["width"],
        "height": page_chars["height"],
        "rotation": page_chars["rotation"],
        "blocks": blocks,
    }


def _get_char_bbox_coords(char: dict[str, Any]) -> tuple[float, ...]:
    """统一提取字符 bbox 坐标，兼容 pdftext Bbox 对象和普通 list。"""
    bbox = char.get("bbox")
    bbox_coords = getattr(bbox, "bbox", bbox)
    return tuple(float(coord) for coord in bbox_coords)


def _get_visible_char_signature(
    char: dict[str, Any],
) -> tuple[str, tuple[Any, Any, Any, Any], float]:
    """生成可见字符去重签名，不把 bbox 放入签名以便单独做近重合判断。"""
    font = char.get("font") or {}
    font_key = (
        font.get("name"),
        font.get("flags"),
        font.get("size"),
        font.get("weight"),
    )
    rotation_key = round(float(char.get("rotation") or 0.0), 3)
    return char.get("char", ""), font_key, rotation_key


def _is_near_identical_bbox(
    bbox_a: tuple[float, ...],
    bbox_b: tuple[float, ...],
) -> bool:
    """判断两个字符 bbox 是否属于同一视觉位置的一点内抖动。"""
    return all(
        abs(coord_a - coord_b) <= NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE
        for coord_a, coord_b in zip(bbox_a, bbox_b)
    )


def _calculate_bbox_overlap_in_smaller_area(
    bbox_a: tuple[float, ...],
    bbox_b: tuple[float, ...],
) -> float:
    """计算两个字符框交集占较小字符框面积的比例。"""
    intersection_width = max(
        0.0,
        min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0]),
    )
    intersection_height = max(
        0.0,
        min(bbox_a[3], bbox_b[3]) - max(bbox_a[1], bbox_b[1]),
    )
    bbox_a_area = max(0.0, bbox_a[2] - bbox_a[0]) * max(
        0.0,
        bbox_a[3] - bbox_a[1],
    )
    bbox_b_area = max(0.0, bbox_b[2] - bbox_b[0]) * max(
        0.0,
        bbox_b[3] - bbox_b[1],
    )
    smaller_area = min(bbox_a_area, bbox_b_area)
    if smaller_area == 0:
        return 0.0
    return intersection_width * intersection_height / smaller_area


def _is_adjacent_offset_duplicate_char(
    previous_char: dict[str, Any],
    current_char: dict[str, Any],
) -> bool:
    """识别相邻字符中由对角平移阴影产生的第二个重复字符。"""
    if _get_visible_char_signature(previous_char) != _get_visible_char_signature(
        current_char
    ):
        return False

    previous_bbox = _get_char_bbox_coords(previous_char)
    current_bbox = _get_char_bbox_coords(current_char)
    x_start_offset = current_bbox[0] - previous_bbox[0]
    y_start_offset = current_bbox[1] - previous_bbox[1]
    x_end_offset = current_bbox[2] - previous_bbox[2]
    y_end_offset = current_bbox[3] - previous_bbox[3]

    # 阴影层应是同一字符框的刚性平移，避免把大小不同的相邻同字误判为重复。
    if (
        abs(x_start_offset - x_end_offset)
        > OFFSET_DUPLICATE_TRANSLATION_TOLERANCE
        or abs(y_start_offset - y_end_offset)
        > OFFSET_DUPLICATE_TRANSLATION_TOLERANCE
    ):
        return False

    if not (
        NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE
        < abs(x_start_offset)
        <= OFFSET_DUPLICATE_CHAR_BBOX_TOLERANCE
        and NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE
        < abs(y_start_offset)
        <= OFFSET_DUPLICATE_CHAR_BBOX_TOLERANCE
    ):
        return False

    return (
        _calculate_bbox_overlap_in_smaller_area(previous_bbox, current_bbox)
        >= OFFSET_DUPLICATE_MIN_BBOX_OVERLAP_RATIO
    )


def _get_near_identical_bbox_bucket_key(
    bbox_coords: tuple[float, ...],
) -> tuple[int, int]:
    """按字符 bbox 左上角生成空间桶 key，缩小近重合判断的候选范围。"""
    return (
        math.floor(bbox_coords[0] / NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE),
        math.floor(bbox_coords[1] / NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE),
    )


def _iter_neighbor_bbox_bucket_keys(
    bucket_key: tuple[int, int],
):
    """遍历当前桶及周围 8 个邻近桶，覆盖 bbox 容差范围内的候选字符。"""
    bucket_x, bucket_y = bucket_key
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            yield bucket_x + offset_x, bucket_y + offset_y


def _is_pdftext_page_chars(chars: Any) -> bool:
    """判断对象是否为 pdftext 0.7 引入的 PageChars 列式字符容器。"""
    return PageChars is not None and isinstance(chars, PageChars)


def _materialize_page_chars(chars) -> list[dict[str, Any]]:
    """将 pdftext 0.7 的 PageChars 物化为 MinerU 既有 char dict 列表。"""
    boxes = chars.boxes.tolist()
    rotations = chars.rotations.tolist()
    font_ids = chars.font_ids.tolist()
    char_indices = chars.char_indices.tolist()

    return [
        {
            "bbox": Bbox([float(coord) for coord in boxes[index]]),
            "char": chars.text[index],
            "rotation": float(rotations[index]),
            "font": chars.fonts[int(font_ids[index])],
            "char_idx": int(char_indices[index]),
        }
        for index in range(len(chars))
    ]


def _ensure_legacy_chars(chars) -> list[dict[str, Any]]:
    """统一输出旧版 char dict 列表，隔离 pdftext 0.7 的返回结构变化。"""
    if _is_pdftext_page_chars(chars):
        return _materialize_page_chars(chars)
    return chars


def _get_single_char_text(char: dict[str, Any]) -> str:
    """提取单个 PDF 字符文本，异常空值用替换符保证 PageChars 长度一致。"""
    text = char.get("char", "")
    if len(text) == 1:
        return text
    return text[:1] or "\uFFFD"


def _get_char_font_id(
    char: dict[str, Any],
    fonts: list[dict[str, Any]],
    font_cache: dict[tuple[Any, Any, Any, Any], int],
) -> int:
    """为旧版字符 font 生成 PageChars 需要的页内 font id。"""
    font = char.get("font") or {}
    font_key = (
        font.get("name"),
        font.get("flags"),
        font.get("size"),
        font.get("weight"),
    )
    font_id = font_cache.get(font_key)
    if font_id is None:
        font_id = len(fonts)
        font_cache[font_key] = font_id
        fonts.append(
            {
                "name": font.get("name"),
                "flags": font.get("flags"),
                "size": font.get("size"),
                "weight": font.get("weight"),
            }
        )
    return font_id


def _get_char_index(char: dict[str, Any], fallback_idx: int) -> int:
    """提取旧版字符索引，缺失或为空时回退到当前列表位置。"""
    char_idx = char.get("char_idx")
    if char_idx is None:
        char_idx = fallback_idx
    return int(char_idx)


def _legacy_chars_to_page_chars(chars):
    """将旧版 char dict 列表打包回 pdftext 0.7 get_spans 所需的 PageChars。"""
    if PageChars is None or _is_pdftext_page_chars(chars):
        return chars

    fonts = []
    font_cache = {}
    text_parts = []
    codes = []
    rotations = []
    boxes = []
    font_ids = []
    char_indices = []

    for fallback_idx, char in enumerate(chars):
        char_text = _get_single_char_text(char)
        text_parts.append(char_text)
        codes.append(ord(char_text))
        rotations.append(float(char.get("rotation") or 0.0))
        boxes.append(_get_char_bbox_coords(char))
        font_ids.append(_get_char_font_id(char, fonts, font_cache))
        char_indices.append(_get_char_index(char, fallback_idx))

    return PageChars(
        "".join(text_parts),
        np.array(codes, dtype=np.uint32),
        np.array(rotations, dtype=np.float64),
        np.array(boxes, dtype=np.float64).reshape((len(boxes), 4)),
        np.array(font_ids, dtype=np.int32),
        fonts,
        np.array(char_indices, dtype=np.int64),
    )


def _deduplicate_near_identical_chars(
    chars: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """清理 PDFium 文本层边界处同字符、同位置的重复可见字符。"""
    seen_visible_char_bboxes = {}
    deduplicated_chars = []

    for char in chars:
        text = char.get("char", "")
        if not text or text.isspace():
            deduplicated_chars.append(char)
            continue

        visible_char_key = _get_visible_char_signature(char)
        bbox_coords = _get_char_bbox_coords(char)
        if deduplicated_chars and _is_adjacent_offset_duplicate_char(
            deduplicated_chars[-1],
            char,
        ):
            continue
        bbox_bucket_key = _get_near_identical_bbox_bucket_key(bbox_coords)
        visible_char_bbox_buckets = seen_visible_char_bboxes.setdefault(
            visible_char_key,
            {},
        )
        if any(
            _is_near_identical_bbox(bbox_coords, seen_bbox)
            for neighbor_bucket_key in _iter_neighbor_bbox_bucket_keys(
                bbox_bucket_key
            )
            for seen_bbox in visible_char_bbox_buckets.get(neighbor_bucket_key, [])
        ):
            continue
        visible_char_bbox_buckets.setdefault(bbox_bucket_key, []).append(bbox_coords)
        deduplicated_chars.append(char)

    return deduplicated_chars


def get_page_chars(
    page: pdfium.PdfPage,
    textpage=None,
    quote_loosebox: bool = True,
    page_char_count: int | None = None,
) -> dict:
    """轻量读取页面字符坐标，供只需要 char 级信息的路径复用。"""
    owns_textpage = textpage is None
    try:
        with pdfium_guard():
            if textpage is None:
                textpage = page.get_textpage()
            page_bbox: List[float] = page.get_bbox()
            page_width = math.ceil(abs(page_bbox[2] - page_bbox[0]))
            page_height = math.ceil(abs(page_bbox[1] - page_bbox[3]))

            page_rotation = 0
            try:
                page_rotation = page.get_rotation()
            except Exception:
                pass

            if page_char_count is None:
                page_char_count = textpage.count_chars()

            chars = deduplicate_chars(
                get_chars(textpage, page_bbox, page_rotation, quote_loosebox)
            )
            chars = _ensure_legacy_chars(chars)
            chars = _deduplicate_near_identical_chars(chars)
    finally:
        if owns_textpage:
            close_pdfium_child(textpage)

    return {
        "bbox": page_bbox,
        "width": page_width,
        "height": page_height,
        "rotation": page_rotation,
        "char_count": page_char_count,
        "chars": chars,
    }


def get_lines_from_chars(
    chars,
    superscript_height_threshold: float = 0.7,
    line_distance_threshold: float = 0.1,
):
    """从已提取的字符构建 pdftext lines，避免重复读取 PDFium textpage。"""
    chars = _legacy_chars_to_page_chars(chars)
    spans = get_spans(
        chars,
        superscript_height_threshold=superscript_height_threshold,
        line_distance_threshold=line_distance_threshold,
    )
    lines = get_lines(spans)
    assign_scripts(
        lines,
        height_threshold=superscript_height_threshold,
        line_distance_threshold=line_distance_threshold,
    )
    return lines
