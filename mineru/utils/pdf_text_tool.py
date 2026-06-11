# Copyright (c) Opendatalab. All rights reserved.
import math
from typing import Any, List

import pypdfium2 as pdfium
from pdftext.pdf.chars import deduplicate_chars, get_chars
from pdftext.pdf.pages import assign_scripts, get_blocks, get_lines, get_spans

from mineru.utils.pdfium_guard import close_pdfium_child, pdfium_guard

NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE = 1.0


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
