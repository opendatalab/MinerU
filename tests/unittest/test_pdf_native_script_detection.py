from __future__ import annotations

import math
from pathlib import Path

import pytest
from pdftext.schema import Bbox, Char

from mineru.backend.analysis.pdf.text import native
from mineru.backend.analysis.pdf.text.models import _AnalyzeSpan
from mineru.types import ContentType
from mineru.utils.pdf_document import PDFDocument


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEMO_PDF_DIR = _PROJECT_ROOT / "demo" / "pdfs"
_FIXTURE_PDF_DIR = Path(__file__).resolve().parent / "pdfs"
_PDF_CONTROL_CHARS = {"\r", "\n", "\x02", "\ufffe", "\uffff"}


def _lines_chars_containing(pdf_path: Path, page_index: int, probe: str) -> list[list[Char]]:
    """从真实 PDF 指定页中返回全部包含探针的物理行字符。"""
    with PDFDocument(str(pdf_path)) as document:
        matching_lines = []
        for line in document.get_page_lines(page_index):
            line_text = "".join(span["text"] for span in line["spans"])
            line_text = "".join(char for char in line_text if char not in _PDF_CONTROL_CHARS)
            if probe in line_text:
                matching_lines.append(
                    [char for span in line["spans"] for char in span.get("chars", [])]
                )
    return matching_lines


def _line_chars_containing(pdf_path: Path, page_index: int, probe: str) -> list[Char]:
    """从真实 PDF 指定页中返回唯一包含探针的物理行字符。"""
    matching_lines = _lines_chars_containing(pdf_path, page_index, probe)

    assert len(matching_lines) == 1, (pdf_path, page_index, probe, len(matching_lines))
    return matching_lines[0]


def _contiguous_chars_containing(pdf_path: Path, page_index: int, probe: str) -> list[Char]:
    """忽略 PDF 控制字符后截取唯一连续探针，覆盖视觉同行但内部带换行的文本。"""
    with PDFDocument(str(pdf_path)) as document:
        visible_chars = [
            char for char in document[page_index].get_chars() if str(char.get("char", "")) not in _PDF_CONTROL_CHARS
        ]

    page_text = "".join(str(char.get("char", "")) for char in visible_chars)
    start_index = page_text.find(probe)
    assert start_index >= 0, (pdf_path, page_index, probe)
    assert page_text.find(probe, start_index + 1) < 0, (pdf_path, page_index, probe)
    return visible_chars[start_index : start_index + len(probe)]


def _render_chars(chars: list[Char]) -> str:
    """使用生产 chars_to_content 路径重建字符内容和上下标标签。"""
    span = _AnalyzeSpan(
        type=ContentType.TEXT,
        bbox=(0.0, 0.0, 1.0, 1.0),
        metadata={"chars": chars},
    )
    native.chars_to_content(span)
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


def test_weak_offset_component_without_seed_stays_body() -> None:
    """验证只有连续级弱偏移、没有高置信证据的组件不会产生标签。"""
    chars = [
        _script_char(0, "A", height=10.0, center_y=10.0),
        _script_char(1, "x", height=9.0, center_y=10.9),
    ]

    assert _render_chars(chars) == "Ax"


def test_strong_geometry_seed_absorbs_same_side_punctuation() -> None:
    """验证强几何种子会把同侧连续标点纳入同一个上标组件。"""
    chars = [
        _script_char(0, "A", height=10.0, center_y=10.0),
        _script_char(1, "2", height=7.0, center_y=7.0),
        _script_char(2, ",", height=9.0, center_y=8.5),
    ]

    assert _render_chars(chars) == "A<sup>2,</sup>"


def test_font_suffix_seed_keeps_its_base_character_body() -> None:
    """验证局部字体后缀只标记后缀，受保护的基字符不能被组件吸收。"""
    chars = [
        _script_char(0, "9", height=20.0, center_y=12.0),
        _script_char(1, " ", height=0.0, center_y=12.0),
        _script_char(2, "R", height=11.0, center_y=10.0, font_name="Base"),
        _script_char(3, "2", height=10.5, center_y=8.5, font_name="Suffix"),
    ]

    assert _render_chars(chars) == "9 R<sup>2</sup>"


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


def test_shifted_brackets_seed_the_complete_reference_component() -> None:
    """验证成对缩小括号会把内部数字整体作为结构上标证据。"""
    chars = [
        _script_char(0, "依", height=10.0, center_y=10.0),
        _script_char(1, "［", height=7.0, center_y=7.0),
        _script_char(2, "1", height=10.2, center_y=7.5, font_name="Reference"),
        _script_char(3, "］", height=7.0, center_y=7.0),
        _script_char(4, "。", height=10.0, center_y=10.0),
    ]

    assert _render_chars(chars) == "依<sup>［1］</sup>。"


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
    font_names = {
        str((char.get("font") or {}).get("name", ""))
        for chars in matching_lines
        for char in chars
    }

    assert contents
    assert len(font_names) >= minimum_font_count
    assert all("<sup>" not in content and "<sub>" not in content for content in contents)


def test_rotated_powerpoint_page_keeps_all_plain_text_on_baseline() -> None:
    """验证 PowerPoint 旋转页的紧字形框不会把普通升部和降部误判为上下标。"""
    pdf_path = _FIXTURE_PDF_DIR / "metabolic_pathway_page_3.pdf"
    with PDFDocument(str(pdf_path)) as document:
        page = document[0]
        page_chars = page.get_chars()
        lines = document.get_page_lines(0)

    rendered_lines = [
        _render_chars([char for span in line["spans"] for char in span.get("chars", [])])
        for line in lines
    ]

    assert page.rotation == 90
    assert any((char.get("font") or {}).get("name") == "CalifornianFB-Reg" for char in page_chars)
    assert len(rendered_lines) == 33
    assert rendered_lines[0] == "Energy Metabolism"
    assert all("<sup>" not in content and "<sub>" not in content for content in rendered_lines)
