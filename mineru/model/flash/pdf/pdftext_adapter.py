# Copyright (c) Opendatalab. All rights reserved.
"""集中隔离 pdftext 0.6 列表与 0.7 PageChars 容器之间的适配。"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from pdftext.pdf.chars import deduplicate_chars
from pdftext.pdf.pages import assign_scripts, get_lines, get_spans
from pdftext.schema import Bbox, Char, Line

try:
    from pdftext.pdf.chars import PageChars
except ImportError:
    PageChars = None


def _is_pdftext_page_chars(chars: Any) -> bool:
    """判断对象是否为 pdftext 0.7 引入的 PageChars 列式字符容器。"""
    return PageChars is not None and isinstance(chars, PageChars)


def _deduplicate_pdftext_chars(chars: Any) -> Any:
    """按当前 pdftext 返回类型调用官方去重，兼容测试或旧版本的 list 字符。"""
    if _is_pdftext_page_chars(chars) or PageChars is None:
        return deduplicate_chars(chars)
    return chars


def _materialize_page_chars(chars: Any) -> list[Char]:
    """将 pdftext 0.7 的 PageChars 物化为 MinerU 既有 char dict 列表。"""
    boxes = chars.boxes.tolist()
    rotations = chars.rotations.tolist()
    font_ids = chars.font_ids.tolist()
    char_indices = chars.char_indices.tolist()

    return [
        cast(
            Char,
            {
                "bbox": Bbox([float(coord) for coord in boxes[index]]),
                "char": chars.text[index],
                "rotation": float(rotations[index]),
                "font": chars.fonts[int(font_ids[index])],
                "char_idx": int(char_indices[index]),
            },
        )
        for index in range(len(chars))
    ]


def _ensure_legacy_chars(chars: Any) -> list[Char]:
    """统一输出旧版 char dict 列表，隔离 pdftext 0.7 的返回结构变化。"""
    if _is_pdftext_page_chars(chars):
        return _materialize_page_chars(chars)
    return cast(list[Char], chars)


def _char_bbox_values(bbox: object) -> tuple[float, float, float, float] | None:
    """将 tuple/list 或 pdftext Bbox 对象统一转换为四元组坐标。"""
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return tuple(float(value) for value in bbox)  # type: ignore[return-value]

    attrs = ("x_start", "y_start", "x_end", "y_end")
    if all(hasattr(bbox, attr) for attr in attrs):
        return tuple(float(getattr(bbox, attr)) for attr in attrs)  # type: ignore[return-value]
    return None


def _get_single_char_text(char: Char) -> str:
    """提取单个 PDF 字符文本，异常空值用替换符保证 PageChars 长度一致。"""
    text = str(char.get("char", ""))
    if len(text) == 1:
        return text
    return text[:1] or "\ufffd"


def _get_char_font_id(
    char: Char,
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


def _get_char_index(char: Char, fallback_idx: int) -> int:
    """提取旧版字符索引，缺失或为空时回退到当前列表位置。"""
    char_idx = char.get("char_idx")
    if char_idx is None:
        char_idx = fallback_idx
    return int(char_idx)


def _legacy_chars_to_page_chars(chars: Any) -> Any:
    """将旧版 char dict 列表打包回 pdftext 0.7 get_spans 所需的 PageChars。"""
    if PageChars is None or _is_pdftext_page_chars(chars):
        return chars

    fonts: list[dict[str, Any]] = []
    font_cache: dict[tuple[Any, Any, Any, Any], int] = {}
    text_parts: list[str] = []
    codes: list[int] = []
    rotations: list[float] = []
    boxes: list[tuple[float, float, float, float]] = []
    font_ids: list[int] = []
    char_indices: list[int] = []

    for fallback_idx, char in enumerate(cast(list[Char], chars)):
        char_text = _get_single_char_text(char)
        bbox_values = _char_bbox_values(char.get("bbox"))
        if bbox_values is None:
            bbox_values = (0.0, 0.0, 0.0, 0.0)
        text_parts.append(char_text)
        codes.append(ord(char_text))
        rotations.append(float(char.get("rotation") or 0.0))
        boxes.append(bbox_values)
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


def get_lines_from_chars(
    chars: list[Char],
    superscript_height_threshold: float = 0.7,
    line_distance_threshold: float = 0.1,
) -> list[Line]:
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


__all__ = ["get_lines_from_chars"]
