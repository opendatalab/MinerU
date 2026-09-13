# Copyright (c) Opendatalab. All rights reserved.
from typing import Any

from pdftext.schema import Bbox

from mineru.utils.pdf_text_tool import _deduplicate_near_identical_chars


def _char(
    text: str,
    bbox: tuple[float, float, float, float],
    char_idx: int,
    font_name: str = "TestFont",
    font_size: float = 12.0,
) -> dict[str, Any]:
    return {
        "char": text,
        "bbox": Bbox([float(coord) for coord in bbox]),
        "rotation": 0,
        "font": {"name": font_name, "flags": 0, "size": font_size, "weight": 400},
        "char_idx": char_idx,
    }


def _text(chars: list[dict[str, Any]]) -> str:
    return "".join(char["char"] for char in chars)


def test_deduplicate_keeps_ff_ligature_expansion() -> None:
    """ToUnicode 展开的 ff 连字共享同一字符框且索引连续，不应被去重。"""
    ff_bbox = (113.32, 189.17, 127.18, 213.11)
    chars = [
        _char("d", (100.0, 189.17, 106.0, 213.11), 0),
        _char("i", (106.0, 189.17, 109.0, 213.11), 1),
        _char("f", ff_bbox, 2),
        _char("f", ff_bbox, 3),
        _char("e", (127.18, 189.17, 133.0, 213.11), 4),
        _char("r", (133.0, 189.17, 139.0, 213.11), 5),
    ]

    assert _text(_deduplicate_near_identical_chars(chars)) == "differ"


def test_deduplicate_keeps_ffi_ligature_expansion() -> None:
    """ffi 连字展开出的 f、f、i 共享同一字符框，第二个 f 不应被去重。"""
    ffi_bbox = (320.15, 265.63, 329.91, 277.58)
    chars = [
        _char("e", (314.95, 265.63, 320.15, 277.58), 0),
        _char("f", ffi_bbox, 1),
        _char("f", ffi_bbox, 2),
        _char("i", ffi_bbox, 3),
        _char("c", (329.91, 265.63, 335.11, 277.58), 4),
    ]

    assert _text(_deduplicate_near_identical_chars(chars)) == "effic"


def test_deduplicate_removes_non_adjacent_shadow_duplicate() -> None:
    """真实阴影重复字来自另一个文本对象，字符索引不连续，仍应被去重。"""
    char_bbox = (10.0, 20.0, 16.0, 32.0)
    chars = [
        _char("x", char_bbox, 0),
        _char("y", (16.0, 20.0, 22.0, 32.0), 1),
        # 阴影层整体来自第二个文本对象：x 的重复与原 x 索引不连续
        _char("x", char_bbox, 2),
        _char("y", (16.0, 20.0, 22.0, 32.0), 3),
    ]

    assert _text(_deduplicate_near_identical_chars(chars)) == "xy"


def test_deduplicate_removes_non_ligature_adjacent_duplicate() -> None:
    """索引连续但拼不出已知连字的同框重复字符（如 oo）仍应被去重。"""
    char_bbox = (10.0, 20.0, 16.0, 32.0)
    chars = [
        _char("c", (4.0, 20.0, 10.0, 32.0), 0),
        _char("o", char_bbox, 1),
        _char("o", char_bbox, 2),
        _char("l", (16.0, 20.0, 22.0, 32.0), 3),
    ]

    assert _text(_deduplicate_near_identical_chars(chars)) == "col"


def test_deduplicate_keeps_separate_ff_with_distinct_bboxes() -> None:
    """未连字的两个 f 字符框不同，本就不触发去重，行为保持不变。"""
    chars = [
        _char("f", (10.0, 20.0, 14.0, 32.0), 0),
        _char("f", (14.0, 20.0, 18.0, 32.0), 1),
    ]

    assert _text(_deduplicate_near_identical_chars(chars)) == "ff"
