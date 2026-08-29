from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import pytest
from pdftext.schema import Bbox, Char

from mineru.backend.analysis.pdf.text import native
from mineru.backend.analysis.pdf.text.models import _AnalyzeSpan
from mineru.model.flash.pdf.document import PDFDocument, get_lines_from_chars
from mineru.model.flash.pdf.text_styles import PDF_NATIVE_SCRIPT_MARKUP_KEY
from mineru.types import BBox, ContentType


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEMO_PDF_DIR = _PROJECT_ROOT / "demo" / "pdfs"
_ZH_2_PDF = _DEMO_PDF_DIR / "中文论文2.pdf"
_FIXTURE_PDF_DIR = Path(__file__).resolve().parent / "pdfs"
_PDF_CONTROL_CHARS = {"\r", "\n", "\x02", "\ufffe", "\uffff"}


@dataclass(frozen=True, slots=True)
class _ScriptFixture:
    """保存测试字符及对应 tight/origin side maps。"""

    chars: list[Char]
    tight_bboxes: dict[int, BBox]
    origins: dict[int, tuple[float, float]]


def _lines_chars_containing(pdf_path: Path, page_index: int, probe: str) -> list[_ScriptFixture]:
    """从真实 PDF 指定页中返回全部包含探针的物理行字符。"""
    with PDFDocument(str(pdf_path)) as document:
        geometry = document.get_page_chars_with_geometry(page_index)
        matching_lines = []
        for line in get_lines_from_chars(geometry.chars):
            line_text = "".join(span["text"] for span in line["spans"])
            line_text = "".join(char for char in line_text if char not in _PDF_CONTROL_CHARS)
            if probe in line_text:
                matching_lines.append(
                    _ScriptFixture(
                        chars=[char for span in line["spans"] for char in span.get("chars", [])],
                        tight_bboxes=geometry.tight_bboxes,
                        origins=geometry.origins,
                    )
                )
    return matching_lines


def _line_chars_containing(pdf_path: Path, page_index: int, probe: str) -> _ScriptFixture:
    """从真实 PDF 指定页中返回唯一包含探针的物理行字符。"""
    matching_lines = _lines_chars_containing(pdf_path, page_index, probe)

    assert len(matching_lines) == 1, (pdf_path, page_index, probe, len(matching_lines))
    return matching_lines[0]


def _contiguous_chars_containing(pdf_path: Path, page_index: int, probe: str) -> _ScriptFixture:
    """忽略 PDF 控制字符后截取唯一连续探针，覆盖视觉同行但内部带换行的文本。"""
    with PDFDocument(str(pdf_path)) as document:
        geometry = document.get_page_chars_with_geometry(page_index)
        visible_chars = [char for char in geometry.chars if str(char.get("char", "")) not in _PDF_CONTROL_CHARS]

    page_text = "".join(str(char.get("char", "")) for char in visible_chars)
    start_index = page_text.find(probe)
    assert start_index >= 0, (pdf_path, page_index, probe)
    assert page_text.find(probe, start_index + 1) < 0, (pdf_path, page_index, probe)
    return _ScriptFixture(
        chars=visible_chars[start_index : start_index + len(probe)],
        tight_bboxes=geometry.tight_bboxes,
        origins=geometry.origins,
    )


def _synthetic_fixture(
    chars: list[Char],
    *,
    tight_heights: dict[int, float] | None = None,
    origin_ys: dict[int, float] | None = None,
) -> _ScriptFixture:
    """从 loose 测试字符构造可覆盖 tight 高度和 origin 的几何 fixture。"""
    tight_heights = tight_heights or {}
    origin_ys = origin_ys or {}
    tight_bboxes = {}
    origins = {}
    for char in chars:
        char_idx = int(char["char_idx"])
        bbox = tuple(float(item) for item in char["bbox"])
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        tight_height = tight_heights.get(char_idx, bbox[3] - bbox[1])
        tight_bboxes[char_idx] = (
            bbox[0],
            center_y - tight_height / 2,
            bbox[2],
            center_y + tight_height / 2,
        )
        origins[char_idx] = (center_x, origin_ys.get(char_idx, center_y))
    return _ScriptFixture(
        chars=chars,
        tight_bboxes=tight_bboxes,
        origins=origins,
    )


def _tight_only_span_bbox(fixture: _ScriptFixture) -> BBox:
    """构造真实 tight 字形可命中、异常 loose 字体框完全无法命中的窄 Span。"""
    tight_boxes = [fixture.tight_bboxes[int(char["char_idx"])] for char in fixture.chars]
    center_y = statistics.median((bbox[1] + bbox[3]) / 2 for bbox in tight_boxes)
    x0 = min(bbox[0] for bbox in tight_boxes) - 0.5
    x1 = max(bbox[2] for bbox in tight_boxes) + 0.5
    for quarter_points in range(16, 161):
        height = quarter_points / 4
        span_bbox = (x0, center_y - height / 2, x1, center_y + height / 2)
        tight_matches = [
            native.calculate_char_in_span(
                tight_bbox,
                span_bbox,
                str(char["char"]),
            )
            for char, tight_bbox in zip(fixture.chars, tight_boxes)
        ]
        loose_matches = [
            native.calculate_char_in_span(
                tuple(float(value) for value in char["bbox"]),
                span_bbox,
                str(char["char"]),
            )
            for char in fixture.chars
        ]
        if all(tight_matches) and not any(loose_matches):
            return span_bbox
    raise AssertionError("未找到可复现 loose 字体框偏移的 tight-only Span")


def _render_chars(value: list[Char] | _ScriptFixture) -> str:
    """使用生产 chars_to_content 路径重建字符内容和上下标标签。"""
    fixture = value if isinstance(value, _ScriptFixture) else _synthetic_fixture(value)
    chars = fixture.chars
    span = _AnalyzeSpan(
        type=ContentType.TEXT,
        bbox=(0.0, 0.0, 1.0, 1.0),
        metadata={"chars": chars},
    )
    native.chars_to_content(
        span,
        tight_bboxes=fixture.tight_bboxes,
        origins=fixture.origins,
    )
    return span.content


def _render_pdf_line(pdf_path: Path, page_index: int, probe: str) -> str:
    """定位真实 PDF 物理行并使用生产逻辑返回重建文本。"""
    return _render_chars(_line_chars_containing(pdf_path, page_index, probe))


def _script_char(
    index: int,
    text: str,
    *,
    height: float,
    center_y: float,
    font_name: str = "Body",
) -> Char:
    """构造最小字符几何，用于验证组件证据和边界不变量。"""
    return {
        "bbox": Bbox(
            [
                float(index),
                center_y - height / 2,
                float(index + 1),
                center_y + height / 2,
            ]
        ),
        "char": text,
        "rotation": 0.0,
        "font": {"name": font_name, "flags": 0, "size": 1.0, "weight": 400},
        "char_idx": index,
    }


def _rotated_char(index: int, text: str, rotation_degrees: float) -> Char:
    """构造带指定字符方向的最小字符，用于 span 回填角度测试。"""
    char = _script_char(index, text, height=10.0, center_y=10.0)
    char["rotation"] = math.radians(rotation_degrees)
    return char


def test_span_fill_does_not_restore_315_degree_watermark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证标准正文粗行中的 315 度水印字符不会被仿斜体分支重新放行。"""
    body = _rotated_char(0, "A", 0.0)
    watermark = _rotated_char(1, "水", 315.0)
    line = {"rotation": 0.0, "spans": [{"chars": [body, watermark]}]}
    monkeypatch.setattr(native, "get_lines_from_chars", lambda _chars: [line])

    assert native._get_chars_for_span_fill([body, watermark]) == [body]


def test_span_fill_restores_19_degree_sheared_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证标准正文粗行中的约 19 度仿斜体字符仍可参与 span 回填。"""
    body = _rotated_char(0, "A", 0.0)
    sheared = _rotated_char(1, "B", 19.2)
    line = {"rotation": 0.0, "spans": [{"chars": [body, sheared]}]}
    monkeypatch.setattr(native, "get_lines_from_chars", lambda _chars: [line])

    assert native._get_chars_for_span_fill([body, sheared]) == [body, sheared]


def _fillable_span(bbox: tuple[float, float, float, float]) -> _AnalyzeSpan:
    """构造已初始化字符容器的可回填文本 Span。"""
    return _AnalyzeSpan(
        type=ContentType.TEXT,
        bbox=bbox,
        metadata={
            "chars": [],
            "height": bbox[3] - bbox[1],
            "width": bbox[2] - bbox[0],
        },
    )


def test_span_fill_prefers_tight_bbox_over_inflated_loose_bbox() -> None:
    """验证异常 loose 中心落入下一行时仍按 tight bbox 归属正确行。"""
    first = _fillable_span((10.0, 10.0, 80.0, 20.0))
    second = _fillable_span((10.0, 22.0, 80.0, 32.0))
    chars = [
        {
            "char": text,
            "bbox": Bbox([12.0 + index * 6.0, 13.0, 18.0 + index * 6.0, 35.0]),
            "rotation": 0.0,
            "font": {},
            "char_idx": index,
        }
        for index, text in enumerate("gpySource")
    ]
    tight_bboxes = {index: (12.0 + index * 6.0, 12.0, 18.0 + index * 6.0, 19.0) for index in range(len(chars))}
    origins = {index: (15.0 + index * 6.0, 19.0) for index in range(len(chars))}

    native.fill_char_in_spans(
        [first, second],
        chars,  # type: ignore[arg-type]
        10.0,
        tight_bboxes=tight_bboxes,
        origins=origins,
    )

    assert first.content == "gpySource"
    assert second.content == ""


@pytest.mark.parametrize(
    ("page_index", "probe"),
    [
        pytest.param(0, "providing", id="page-1-providing"),
        pytest.param(10, "Source", id="page-11-source"),
        pytest.param(10, "Hypothesis", id="page-11-hypothesis"),
        pytest.param(10, "Reference", id="page-11-reference"),
        pytest.param(10, "BLEU", id="page-11-bleu"),
        pytest.param(10, "ROUGE", id="page-11-rouge"),
        pytest.param(16, "generalisation", id="page-17-generalisation"),
    ],
)
def test_zh2_tight_bbox_restores_words_rejected_by_loose_bbox(
    page_index: int,
    probe: str,
) -> None:
    """验证中文论文2真实字体框中 loose 全部失配时 tight-first 仍恢复完整英文词。"""
    fixture = _contiguous_chars_containing(_ZH_2_PDF, page_index, probe)
    span_bbox = _tight_only_span_bbox(fixture)
    loose_matches = [
        native.calculate_char_in_span(
            tuple(float(value) for value in char["bbox"]),
            span_bbox,
            str(char["char"]),
        )
        for char in fixture.chars
    ]
    span = _fillable_span(span_bbox)

    native.fill_char_in_spans(
        [span],
        fixture.chars,
        span_bbox[3] - span_bbox[1],
        tight_bboxes=fixture.tight_bboxes,
        origins=fixture.origins,
    )

    assert not any(loose_matches)
    assert span.content == probe


def test_span_fill_uses_loose_bbox_when_tight_punctuation_does_not_match() -> None:
    """验证 tight 标点未命中时继续使用原首尾标点 loose 回退。"""
    span = _fillable_span((10.0, 10.0, 30.0, 20.0))
    char = {
        "char": ".",
        "bbox": Bbox([27.0, 12.0, 31.0, 18.0]),
        "rotation": 0.0,
        "font": {},
        "char_idx": 0,
    }

    native.fill_char_in_spans(
        [span],
        [char],  # type: ignore[list-item]
        10.0,
        tight_bboxes={0: (40.0, 12.0, 41.0, 18.0)},
        origins={0: (27.0, 18.0)},
    )

    assert span.content == "."


def test_span_fill_attaches_whitespace_to_tight_owned_neighbors() -> None:
    """验证空格跟随两侧 tight 字符进入同一 Span，而不是留在旧的重叠 Span。"""
    first = _fillable_span((0.0, 0.0, 12.0, 20.0))
    closer = _fillable_span((0.0, 8.0, 12.0, 16.0))
    chars = [
        _script_char(0, "A", height=14.0, center_y=10.0),
        _script_char(1, " ", height=6.0, center_y=6.0),
        _script_char(2, "C", height=14.0, center_y=10.0),
    ]
    chars[0]["bbox"] = Bbox([0.0, 3.0, 5.0, 17.0])
    chars[1]["bbox"] = Bbox([4.8, 6.0, 4.8, 6.0])
    chars[2]["bbox"] = Bbox([5.0, 3.0, 10.0, 17.0])

    native.fill_char_in_spans(
        [first, closer],
        chars,
        8.0,
        tight_bboxes={
            0: (0.0, 10.0, 5.0, 14.0),
            1: (8.0, 10.0, 9.0, 14.0),
            2: (5.0, 10.0, 10.0, 14.0),
        },
        origins={index: (float(index), 14.0) for index in range(3)},
    )

    assert first.content == ""
    assert closer.content == "A C"


def test_span_fill_chooses_closest_tight_candidate_before_span_order() -> None:
    """验证 tight bbox 同时命中重叠 Span 时按中心距离而非列表顺序归属。"""
    farther = _fillable_span((10.0, 8.0, 30.0, 20.0))
    closer = _fillable_span((10.0, 11.0, 30.0, 19.0))
    char = {
        "char": "A",
        "bbox": Bbox([15.0, 8.0, 20.0, 24.0]),
        "rotation": 0.0,
        "font": {},
        "char_idx": 0,
    }

    native.fill_char_in_spans(
        [farther, closer],
        [char],  # type: ignore[list-item]
        10.0,
        tight_bboxes={0: (15.0, 13.0, 20.0, 17.0)},
        origins={0: (15.0, 17.0)},
    )

    assert farther.content == ""
    assert closer.content == "A"


def test_weak_offset_component_without_seed_stays_body() -> None:
    """验证只有连续级弱偏移、没有高置信证据的组件不会产生标签。"""
    chars = [
        _script_char(0, "A", height=10.0, center_y=10.0),
        _script_char(1, "x", height=9.0, center_y=10.9),
    ]

    assert _render_chars(chars) == "Ax"


def test_mixed_font_run_with_weak_dual_bbox_displacement_stays_body() -> None:
    """验证字体框高度突变但 tight/origin 中心偏移不足时仍保持正文。"""
    chars = [
        _script_char(0, "中", height=10.0, center_y=10.0, font_name="CJK"),
        _script_char(1, "S", height=20.0, center_y=12.4, font_name="Latin"),
        _script_char(2, "o", height=20.0, center_y=12.4, font_name="Latin"),
    ]
    fixture = _ScriptFixture(
        chars=chars,
        tight_bboxes={
            0: (0.0, 5.0, 1.0, 15.0),
            1: (1.0, 8.8, 2.0, 13.8),
            2: (2.0, 8.8, 3.0, 13.8),
        },
        origins={0: (0.0, 10.0), 1: (1.0, 11.2), 2: (2.0, 11.2)},
    )

    assert _render_chars(fixture) == "中So"


def test_isolated_punctuation_with_opposed_origin_stays_body() -> None:
    """验证标点的 origin 与 bbox 中心方向冲突时不会独立成为角标种子。"""
    chars = [
        _script_char(0, "A", height=10.0, center_y=10.0),
        _script_char(1, ":", height=5.0, center_y=13.0),
        _script_char(2, "B", height=10.0, center_y=10.0),
    ]
    fixture = _ScriptFixture(
        chars=chars,
        tight_bboxes={
            0: (0.0, 5.0, 1.0, 15.0),
            1: (1.0, 11.0, 2.0, 15.0),
            2: (2.0, 5.0, 3.0, 15.0),
        },
        origins={0: (0.0, 10.0), 1: (1.0, 8.0), 2: (2.0, 10.0)},
    )

    assert _render_chars(fixture) == "A:B"


def test_strong_isolated_symbol_with_three_consistent_signals_is_script() -> None:
    """验证三类几何均强位移的孤立符号仍可独立识别为角标。"""
    chars = [
        _script_char(0, "A", height=10.0, center_y=10.0),
        _script_char(1, "*", height=5.0, center_y=6.0),
        _script_char(2, "B", height=10.0, center_y=10.0),
    ]

    assert _render_chars(chars) == "A<sup>*</sup>B"


def test_strong_geometry_seed_absorbs_same_side_punctuation() -> None:
    """验证强几何种子会把同侧连续标点纳入同一个上标组件。"""
    chars = [
        _script_char(0, "A", height=10.0, center_y=10.0),
        _script_char(1, "2", height=7.0, center_y=7.0),
        _script_char(2, ",", height=9.0, center_y=8.5),
    ]

    assert _render_chars(chars) == "A<sup>2,</sup>"


def test_script_markup_records_detector_ownership() -> None:
    """验证只有脚本检测实际生成标签时才写入私有 ownership 标记。"""
    script_span = _AnalyzeSpan(
        type=ContentType.TEXT,
        bbox=(0.0, 0.0, 1.0, 1.0),
        metadata={
            "chars": [
                _script_char(0, "A", height=10.0, center_y=10.0),
                _script_char(1, "2", height=7.0, center_y=7.0),
            ]
        },
    )
    body_span = _AnalyzeSpan(
        type=ContentType.TEXT,
        bbox=(0.0, 0.0, 1.0, 1.0),
        metadata={
            "chars": [
                _script_char(0, "A", height=10.0, center_y=10.0),
                _script_char(1, "B", height=10.0, center_y=10.0),
            ]
        },
    )

    script_fixture = _synthetic_fixture(script_span.metadata["chars"])
    body_fixture = _synthetic_fixture(body_span.metadata["chars"])
    native.chars_to_content(
        script_span,
        tight_bboxes=script_fixture.tight_bboxes,
        origins=script_fixture.origins,
    )
    native.chars_to_content(
        body_span,
        tight_bboxes=body_fixture.tight_bboxes,
        origins=body_fixture.origins,
    )

    assert script_span.content == "A<sup>2</sup>"
    assert script_span.metadata[PDF_NATIVE_SCRIPT_MARKUP_KEY] is True
    assert PDF_NATIVE_SCRIPT_MARKUP_KEY not in body_span.metadata


def test_missing_extended_geometry_keeps_all_characters_body() -> None:
    """验证缺少 tight/origin 时宁可不标 style，也不恢复字体或文本猜测。"""
    span = _AnalyzeSpan(
        type=ContentType.TEXT,
        bbox=(0.0, 0.0, 1.0, 1.0),
        metadata={
            "chars": [
                _script_char(0, "R", height=10.0, center_y=10.0, font_name="Base"),
                _script_char(1, "2", height=7.0, center_y=7.0, font_name="Suffix"),
            ]
        },
    )

    native.chars_to_content(span)

    assert span.content == "R2"
    assert PDF_NATIVE_SCRIPT_MARKUP_KEY not in span.metadata


def test_overlapping_spacing_ogonek_composes_demo2_author_name() -> None:
    """验证 demo2 原生字符流中与 e 重叠的 spacing ogonek 合成为 ę。"""
    content = _render_pdf_line(_DEMO_PDF_DIR / "demo2.pdf", 0, "Kowalczuk")

    assert content.startswith("Jędrzej Kowalczuk")
    assert "\u02db" not in content


def test_non_overlapping_spacing_ogonek_remains_literal() -> None:
    """验证没有几何重叠的 spacing ogonek 不会被字符规范化误合并。"""
    chars = [
        _script_char(0, "˛", height=10.0, center_y=10.0),
        _script_char(1, "e", height=10.0, center_y=10.0),
    ]

    merged = native._merge_overlapping_spacing_diacritics(chars)

    assert [char["char"] for char in merged] == ["˛", "e"]


def test_tight_geometry_keeps_same_size_loose_base_character_body() -> None:
    """验证 loose 高度接近时仍只根据 tight/origin 标记后缀。"""
    chars = [
        _script_char(0, "9", height=20.0, center_y=12.0),
        _script_char(1, " ", height=0.0, center_y=12.0),
        _script_char(2, "R", height=11.0, center_y=10.0, font_name="Base"),
        _script_char(3, "2", height=10.5, center_y=8.5, font_name="Suffix"),
    ]

    fixture = _synthetic_fixture(
        chars,
        tight_heights={3: 7.0},
        origin_ys={2: 10.0, 3: 8.5},
    )

    assert _render_chars(fixture) == "9 R<sup>2</sup>"


@pytest.mark.parametrize(
    ("chars", "expected"),
    [
        pytest.param(
            [
                _script_char(0, "N", height=10.0, center_y=10.0, font_name="Latin"),
                _script_char(1, "P", height=10.0, center_y=10.0, font_name="Latin"),
                _script_char(2, "U", height=10.0, center_y=10.0, font_name="Latin"),
                _script_char(3, "开", height=10.0, center_y=8.5, font_name="CJK"),
                _script_char(4, "发", height=10.0, center_y=8.5, font_name="CJK"),
            ],
            "NPU开发",
            id="latin-to-cjk",
        ),
        pytest.param(
            [
                _script_char(0, "2", height=10.0, center_y=10.0, font_name="Digit"),
                _script_char(1, "4", height=10.0, center_y=10.0, font_name="Digit"),
                _script_char(2, "年", height=10.0, center_y=8.5, font_name="CJK"),
            ],
            "24年",
            id="digit-to-cjk",
        ),
    ],
)
def test_cjk_font_boundary_does_not_seed_script_component(
    chars: list[Char],
    expected: str,
) -> None:
    """验证正常的 CJK 与拉丁或数字字体切换不会被解释为上下标后缀。"""
    assert _render_chars(chars) == expected


def test_shifted_reference_geometry_marks_the_complete_component() -> None:
    """验证不读取括号语义也会把同基线引用整体标记为上标。"""
    chars = [
        _script_char(0, "依", height=10.0, center_y=10.0),
        _script_char(1, "［", height=7.0, center_y=7.0),
        _script_char(2, "1", height=10.2, center_y=7.5, font_name="Reference"),
        _script_char(3, "］", height=7.0, center_y=7.0),
        _script_char(4, "。", height=10.0, center_y=10.0),
    ]

    fixture = _synthetic_fixture(
        chars,
        tight_heights={1: 6.0, 2: 6.5, 3: 6.0},
        origin_ys={0: 10.0, 1: 7.0, 2: 7.0, 3: 7.0, 4: 10.0},
    )

    assert _render_chars(fixture) == "依<sup>［1］</sup>。"


@pytest.mark.parametrize(
    ("pdf_name", "page_index", "probe", "expected_fragments"),
    [
        pytest.param(
            "demo1.pdf",
            0,
            "Patrick N.J. Lanea,c,*",
            ("Lane<sup>a,c,*</sup>", "Best<sup>b,c,d</sup>"),
            id="author-affiliations",
        ),
        pytest.param(
            "demo3.pdf",
            1,
            "where segi",
            ("seg<sub>i</sub>",),
            id="variable-subscript",
        ),
        pytest.param(
            "demo4.pdf",
            0,
            "Innate [18F]",
            ("[<sup>18</sup>F]",),
            id="isotope-superscript",
        ),
        pytest.param(
            "mixed_elements_pages_03_06.pdf",
            0,
            "vacuum of 10–3 Pa",
            ("10<sup>–3</sup> Pa",),
            id="negative-exponent",
        ),
        pytest.param(
            "mixed_elements_pages_03_06.pdf",
            0,
            "to consist of ScFeGe2",
            ("ScFeGe<sub>2</sub>", "Sc<sub>4</sub>Fe<sub>4</sub>Ge<sub>6.6</sub>"),
            id="chemical-subscripts",
        ),
        pytest.param(
            "mixed_elements_pages_11_15.pdf",
            0,
            "H2-antagonist",
            ("H<sub>2</sub>-antagonist",),
            id="medical-subscript",
        ),
        pytest.param(
            "2407.00079v4_origi-10.pdf",
            0,
            "7: Tqueue ←",
            ("T<sub>queue</sub>",),
            id="algorithm-variable-subscript",
        ),
        pytest.param(
            "demo2.pdf",
            1,
            "where ∆g(p, q) is",
            ("∆<sub>g</sub>(p, q)", "∆<sub>c</sub>(p, q)"),
            id="delta-subscripts",
        ),
        pytest.param(
            "demo3.pdf",
            4,
            "same as BERTBASE and BERTLARGE",
            ("BERT<sub>BASE</sub>", "BERT<sub>LARGE</sub>"),
            id="model-name-subscripts",
        ),
        pytest.param(
            "demo4.pdf",
            1,
            "body mass index (BMI) were 71.75 kg and 25.3 kg/m2",
            ("kg/m<sup>2</sup>",),
            id="unit-superscript",
        ),
        pytest.param(
            "中文论文.pdf",
            0,
            "其中，季节等［1］",
            ("季节等<sup>［1］</sup>",),
            id="chinese-reference",
        ),
        pytest.param(
            "中文论文.pdf",
            0,
            "等［7］",
            ("等<sup>［7］</sup>",),
            id="chinese-reference-7",
        ),
        pytest.param(
            "中文论文.pdf",
            0,
            "Zhao等［8］",
            ("Zhao等<sup>［8］</sup>",),
            id="chinese-reference-8",
        ),
        pytest.param(
            "中文论文.pdf",
            0,
            "利用决定系数（R2",
            ("决定系数（R<sup>2</sup>",),
            id="same-size-font-suffix",
        ),
        pytest.param(
            "中文论文.pdf",
            0,
            "Voting 模型的性能略低于 Stacking 模型（R2",
            ("Stacking 模型（R<sup>2</sup>",),
            id="same-size-font-suffix-after-tall-digits",
        ),
        pytest.param(
            "中文论文.pdf",
            1,
            "扎兰屯段边坡，经烘干后测得干密度为1. 72 g⋅cm-3",
            ("g⋅cm<sup>-3</sup>",),
            id="cjk-unit-superscript",
        ),
    ],
)
def test_real_pdf_scripts_keep_confirmed_markup(
    pdf_name: str,
    page_index: int,
    probe: str,
    expected_fragments: tuple[str, ...],
) -> None:
    """验证已有输出中经页面视觉确认的真实上下标继续保留。"""
    content = _render_pdf_line(_DEMO_PDF_DIR / pdf_name, page_index, probe)

    assert all(expected_fragment in content for expected_fragment in expected_fragments)


@pytest.mark.parametrize(
    ("probe", "expected_fragment"),
    [
        pytest.param(
            "预测。Meng 等［10］利用机器学习对冻结岩土的力学",
            "Meng 等<sup>［10］</sup>",
            id="reference-10",
        ),
        pytest.param(
            "特性进行了预测。Esmaeili-Falak 等［11］通过机器学习对冻土的轴向抗压强度",
            "Esmaeili-Falak 等<sup>［11］</sup>",
            id="reference-11",
        ),
        pytest.param(
            "Li等［12］采用 ANN 模型对冻土应力应变进行预测和验证。",
            "Li等<sup>［12］</sup>",
            id="reference-12",
        ),
    ],
)
def test_chinese_mixed_font_reference_uses_local_main_font_body(
    probe: str,
    expected_fragment: str,
) -> None:
    """验证含拉丁姓名的中文 OCR 行使用主字体正文带识别数字引用。"""
    chars = _contiguous_chars_containing(_DEMO_PDF_DIR / "中文论文.pdf", 1, probe)

    assert expected_fragment in _render_chars(chars)


def test_chinese_reference_line_only_marks_the_reference() -> None:
    """验证短行中数字引用占优时，正文“依据”和句号不会反向误判为下标。"""
    chars = _contiguous_chars_containing(
        _DEMO_PDF_DIR / "中文论文.pdf",
        1,
        "依据［14-17］。",
    )

    content = _render_chars(chars)

    assert content == "依据<sup>［14-17］</sup>。"
    assert "<sub>" not in content


@pytest.mark.parametrize(
    ("pdf_name", "page_index", "probe", "expected"),
    [
        pytest.param(
            "mixed_elements_pages_03_06.pdf",
            0,
            "0.26 µm (Fig. 1a)",
            "0.26 µm (Fig. 1a)",
            id="plain-micrometer-unit",
        ),
        pytest.param(
            "demo2.pdf",
            2,
            ") to O(ω).",
            ") to O(ω).",
            id="plain-complexity-notation",
        ),
    ],
)
def test_real_pdf_plain_text_does_not_gain_script_markup(
    pdf_name: str,
    page_index: int,
    probe: str,
    expected: str,
) -> None:
    """验证字体框或 glyph 形状差异不会给普通单位和复杂度记号加角标。"""
    content = _render_pdf_line(_DEMO_PDF_DIR / pdf_name, page_index, probe)

    assert expected in content
    assert "<sup>" not in content and "<sub>" not in content


@pytest.mark.parametrize(
    ("page_index", "probe", "expected"),
    [
        pytest.param(0, "providing", "providing", id="page-1-english-abstract"),
        pytest.param(10, "Source", "Source", id="page-11-source"),
        pytest.param(10, "Hypothesis", "Hypothesis", id="page-11-hypothesis"),
        pytest.param(10, "Reference", "Reference", id="page-11-reference"),
        pytest.param(10, "如 BLEU 和 ROUGE", "BLEU 和 ROUGE", id="page-11-metrics"),
        pytest.param(16, "generalisation", "generalisation", id="page-17-reference"),
    ],
)
def test_zh2_normal_english_runs_do_not_gain_script_markup(
    page_index: int,
    probe: str,
    expected: str,
) -> None:
    """验证中文论文2中受字体框影响的普通英文 run 不会被误判成上下标。"""
    content = _render_pdf_line(_ZH_2_PDF, page_index, probe)

    assert expected in content
    assert "<sup>" not in content and "<sub>" not in content


@pytest.mark.parametrize(
    ("probe", "minimum_font_count"),
    [
        pytest.param("• 合成支持工程师 Support Engineer", 2, id="bullet-text"),
        pytest.param("• 混合字体工具 Acuity Toolkit", 2, id="mixed-font-bullet-text"),
        pytest.param("• 测试系统 Ubuntu 24.04", 2, id="version-bullet-text"),
        pytest.param(
            "mineru_toolkit_binary_1.0.0_linux_x86_64.tgz",
            1,
            id="filename-descender",
        ),
    ],
)
def test_synthetic_cjk_plain_text_does_not_gain_script_tags(
    probe: str,
    minimum_font_count: int,
) -> None:
    """验证合成项目符号、混合字体和文件名降部字符不会触发上下标误判。"""

    matching_lines = _lines_chars_containing(
        _FIXTURE_PDF_DIR / "native_cjk_layout_synthetic.pdf",
        2,
        probe,
    )
    contents = [_render_chars(chars) for chars in matching_lines]
    font_names = {str((char.get("font") or {}).get("name", "")) for fixture in matching_lines for char in fixture.chars}

    assert contents
    assert len(font_names) >= minimum_font_count
    assert all("<sup>" not in content and "<sub>" not in content for content in contents)


def test_rotated_powerpoint_page_keeps_all_plain_text_on_baseline() -> None:
    """验证 PowerPoint 旋转页的紧字形框不会把普通升部和降部误判为上下标。"""
    pdf_path = _FIXTURE_PDF_DIR / "metabolic_pathway_page_3.pdf"
    with PDFDocument(str(pdf_path)) as document:
        page = document[0]
        page_rotation = page.rotation
        geometry = page.get_chars_with_geometry()
        lines = get_lines_from_chars(geometry.chars)

    rendered_lines = [
        _render_chars(
            _ScriptFixture(
                chars=[char for span in line["spans"] for char in span.get("chars", [])],
                tight_bboxes=geometry.tight_bboxes,
                origins=geometry.origins,
            )
        )
        for line in lines
    ]

    assert page_rotation == 90
    assert any((char.get("font") or {}).get("name") == "CalifornianFB-Reg" for char in geometry.chars)
    assert len(rendered_lines) == 33
    assert rendered_lines[0] == "Energy Metabolism"
    assert all("<sup>" not in content and "<sub>" not in content for content in rendered_lines)
