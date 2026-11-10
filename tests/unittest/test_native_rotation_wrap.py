"""原生文字方向在整圈边界处保持对称，并保留现有 0.1 度判定。

Keep native text rotation symmetric at the turn boundary without widening its tolerance.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from docvortex.document.pdf import Bbox, Char, PDFDocument

from mineru.backend.analysis.pdf.text import content, native
from mineru.backend.analysis.pdf.text.models import _AnalyzeSpan
from mineru.types import ContentType


_PDF_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "native_rotation_wrap.pdf"
_SHORT_PDF_FIXTURE = _PDF_FIXTURE.with_name("native_rotation_short_labels.pdf")
_PDF_TEXT = "ROTATION PROBE 123"


@pytest.mark.parametrize("degrees", [0.0, 90.0, 180.0, 270.0])
def test_standard_rotations_remain_supported(degrees: float) -> None:
    assert native._is_supported_rotation(math.radians(degrees))


@pytest.mark.parametrize("center", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (-0.101, False),
        (-0.100001, False),
        (-0.099999, True),
        (-0.05, True),
        (0.05, True),
        (0.099999, True),
        (0.100001, False),
        (0.101, False),
    ],
)
def test_tolerance_on_both_sides_of_each_direction(center: float, offset: float, expected: bool) -> None:
    assert native._is_supported_rotation(math.radians((center + offset) % 360)) is expected


@pytest.mark.parametrize("degrees", [-0.1, 0.1, 359.9])
def test_horizontal_exact_boundary_remains_excluded(degrees: float) -> None:
    assert not native._is_supported_rotation(math.radians(degrees))


@pytest.mark.parametrize("degrees", [89.9, 90.1, 179.9, 180.1, 269.9, 270.1])
@pytest.mark.parametrize("toward", [-math.inf, 0.0, math.inf])
def test_other_boundary_float_decisions_remain_unchanged(degrees: float, toward: float) -> None:
    """保留旧减法的浮点边界结果，不以新舍入规则扩大或缩小容差。"""
    rotation = math.radians(degrees)
    if toward:
        rotation = math.nextafter(rotation, toward)
    expected = any(abs(math.degrees(rotation) - angle) < 0.1 for angle in (0, 90, 180, 270))
    assert native._is_supported_rotation(rotation) is expected


@pytest.mark.parametrize("toward", [-math.inf, 0.0, math.inf])
def test_immediately_adjacent_horizontal_boundary_floats(toward: float) -> None:
    rotation = math.radians(0.1)
    if toward:
        rotation = math.nextafter(rotation, toward)
    assert native._is_supported_rotation(rotation) is (math.degrees(rotation) < 0.1)


@pytest.mark.parametrize("degrees", [-360.0, -90.0, -0.05, 359.95, 360.0, 360.05, 450.0, 720.0])
def test_equivalent_signed_or_full_turn_angles(degrees: float) -> None:
    assert native._is_supported_rotation(math.radians(degrees))


@pytest.mark.parametrize("degrees", [-45.0, 20.0, 45.0, 135.0, 225.0, 315.0])
def test_oblique_directions_remain_rejected(degrees: float) -> None:
    assert not native._is_supported_rotation(math.radians(degrees))


@pytest.mark.parametrize("rotation", [math.nan, math.inf, -math.inf])
def test_nonfinite_directions_remain_rejected(rotation: float) -> None:
    assert not native._is_supported_rotation(rotation)


def test_normalized_rotation_changes_are_confined_to_zero_wrap() -> None:
    """逐 0.01 度比较一整圈：只允许原本漏掉的近 360 度方向变为可用。"""
    changed = []
    for hundredths in range(36000):
        degrees = hundredths / 100
        rotation = math.radians(degrees)
        old = any(abs(math.degrees(rotation) - angle) < 0.1 for angle in (0, 90, 180, 270))
        new = native._is_supported_rotation(rotation)
        if old != new:
            assert not old and new
            assert 359.9 < math.degrees(rotation) < 360
            changed.append(degrees)
    assert changed == [359.91, 359.92, 359.93, 359.94, 359.95, 359.96, 359.97, 359.98, 359.99]


def _char_with_rotation(degrees: float) -> Char:
    return {
        "bbox": Bbox([10.0, 10.0, 20.0, 20.0]),
        "char": "A",
        "rotation": math.radians(degrees),
        "font": {"name": "Body", "flags": 0, "size": 10.0, "weight": 400},
        "char_idx": 0,
    }


@pytest.mark.parametrize(
    ("degrees", "expected"), [(0.0, True), (0.05, True), (359.95, True), (359.899, False), (45.0, False), (315.0, False)]
)
def test_native_char_selection_uses_circular_rotation(degrees: float, expected: bool) -> None:
    char = _char_with_rotation(degrees)
    line = {"rotation": char["rotation"], "spans": [{"chars": [char]}]}
    assert native._get_chars_for_span_fill([char], line_provider=lambda: [line]) == ([char] if expected else [])


@pytest.mark.parametrize(
    ("degrees", "expected"), [(0.0, True), (0.05, True), (359.95, True), (359.899, False), (45.0, False), (315.0, False)]
)
def test_native_line_consumer_uses_same_rotation_rule(degrees: float, expected: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    char = _char_with_rotation(degrees)
    line = {"rotation": char["rotation"], "bbox": char["bbox"], "spans": [{"text": "A", "chars": [char]}]}
    monkeypatch.setattr(content, "get_lines_from_chars", lambda _chars: [line])
    page = SimpleNamespace(get_chars=lambda: [char])
    spans = content._build_pdf_text_line_spans(page)  # type: ignore[arg-type]
    assert [span.content for span in spans] == (["A"] if expected else [])


@pytest.mark.parametrize(
    ("fixture", "page_index", "text", "expected"),
    [(_PDF_FIXTURE, index, _PDF_TEXT, _PDF_TEXT if index < 6 else "") for index in range(8)]
    + [(_SHORT_PDF_FIXTURE, index, "A" if index < 3 else "AB", "A" if index < 3 else "AB") for index in range(6)],
)
def test_real_pdf_native_characters_and_lines(fixture: Path, page_index: int, text: str, expected: str) -> None:
    """短标签不足三个字符原点，不能依靠长文本基线补救通过旋转筛选。"""
    with PDFDocument(str(fixture)) as document:
        page = document[page_index]
        chars = page.get_chars()
        assert "".join(char["char"] for char in chars) == text
        assert "".join(char["char"] for char in native._get_chars_for_span_fill(chars)) == expected
        assert [span.content for span in content._build_pdf_text_line_spans(page)] == ([expected] if expected else [])


@pytest.mark.parametrize(
    ("fixture", "page_index", "expected"),
    [(_PDF_FIXTURE, index, _PDF_TEXT) for index in (1, 2)]
    + [(_SHORT_PDF_FIXTURE, index, "A" if index < 3 else "AB") for index in range(6)],
)
def test_near_horizontal_pdf_text_does_not_request_ocr(fixture: Path, page_index: int, expected: str) -> None:
    """使用真实字符几何与同源快照回填，两个近水平方向都不应请求 OCR。"""
    with PDFDocument(str(fixture)) as document:
        page = document[page_index]
        geometry = page.get_chars_with_geometry()
        visible = [char for char in geometry.chars if char["char"].strip()]
        bbox = (
            min(char["bbox"][0] for char in visible) - 2,
            min(char["bbox"][1] for char in visible) - 2,
            max(char["bbox"][2] for char in visible) + 2,
            max(char["bbox"][3] for char in visible) + 2,
        )
        image = document.render_page(page_index, scale=1).pil_image
        try:
            spans = content._fill_native_pdf_text_spans(
                page,
                [_AnalyzeSpan(ContentType.TEXT, bbox)],
                image,
                1,
                page.size,
                page_text_geometry=geometry,
                detect_scripts=False,
                _native_text=page.get_text_snapshot(),
            )
        finally:
            image.close()
        assert len(spans) == 1
        assert spans[0].image is None
        # 字间空格由其他规则处理，本测试只验证可用文字和 OCR 路由。
        assert spans[0].content.replace(" ", "") == expected.replace(" ", "")
