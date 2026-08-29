from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
from pdftext.schema import Bbox, Char

from mineru.model.flash.pdf.geometry import _rotate_bbox_from_upright
from mineru.model.flash import PdfModel
from mineru.model.flash.pdf.document import PDFDocument
from mineru.model.flash.pdf.models import _AxisLine, _LineItem
from mineru.model.flash.pdf.pipeline import _analyze_native_document
from mineru.model.flash.pdf.script_geometry import ScriptRole
from mineru.model.flash.pdf.text_styles import (
    PDFTextLinkLine,
    PDFTextLinkRange,
    PDFTextScriptLine,
    PDFTextScriptRange,
    PDFTextStyleLine,
    PDFTextStyleRange,
    apply_pdf_text_links,
    apply_pdf_text_scripts,
    apply_pdf_text_styles,
    detect_pdf_text_script_lines,
    materialize_pdf_inline_spans,
    _refine_math_script_tokens,
)
from mineru.types import BBox


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEMO_PDF_DIR = _PROJECT_ROOT / "demo" / "pdfs"

_REVIEWED_SCRIPT_EXPECTATIONS = {
    "demo1.pdf": {
        (3, 45): (("2", "superscript"), ("2", "superscript")),
        (3, 50): (("2", "superscript"),),
    },
    "demo3.pdf": {
        (1, 74): (("1", "subscript"), ("2", "subscript"), ("n", "subscript")),
        (3, 5): (("i", "subscript"), ("j", "subscript")),
        (3, 7): (("i", "subscript"),),
        (3, 8): (("j", "subscript"), ("i", "subscript"), ("j", "subscript")),
        (3, 13): (("i", "subscript"), ("j", "subscript")),
        (3, 60): (),
    },
    "中文论文2.pdf": {
        (7, 46): (("1", "subscript"),),
        (7, 47): (("1", "subscript"),),
        (7, 80): (),
        (7, 82): (("m", "subscript"),),
        (7, 83): (("m", "subscript"),),
        (7, 87): (("i", "subscript"), ("i", "subscript")),
        (7, 88): (("m", "subscript"),),
        (8, 7): (("i", "subscript"),),
        (8, 8): (("m", "subscript"),),
        (8, 14): (),
        (8, 22): (),
        (10, 25): (("i", "subscript"),),
        (10, 26): (),
        (10, 27): (("[112]", "superscript"),),
        (11, 35): (("[127]", "superscript"), ("[128]", "superscript"), ("[129]", "superscript")),
        (11, 56): (
            ("1", "subscript"),
            ("2", "subscript"),
            ("n", "subscript"),
            ("1", "subscript"),
            ("2", "subscript"),
            ("m", "subscript"),
        ),
        (11, 66): (("1", "subscript"),),
        (11, 73): (("Score", "subscript"),),
        (11, 92): (),
        (11, 94): (),
        (11, 97): (),
    },
    "IEBM_A_2667169_O-5.pdf": {
        (0, 3): (),
        (0, 7): (),
        (0, 11): (),
        (0, 14): (),
    },
    "mixed_elements_pages_11_15.pdf": {
        (0, 30): (("2", "subscript"),),
    },
}

_RUN_CLOSURE_EXPECTATIONS = {
    "demo3.pdf": {
        (6, 38): (("BASE-SAT", "subscript"),),
        (6, 102): (("BASE-SO", "subscript"),),
        (6, 114): (("BASE-SO", "subscript"),),
    },
    "demo2.pdf": {
        (2, 22): (("t-1", "subscript"),),
        (2, 25): (("t-1", "subscript"),),
    },
    "mixed_elements_pages_03_06.pdf": {
        (0, 22): (("–3", "superscript"),),
        (0, 64): (("6.6", "subscript"),),
        (1, 25): (("–7", "superscript"),),
        (1, 26): (("–5", "superscript"),),
        (3, 19): (("1–x", "subscript"),),
    },
    "mixed_elements_pages_11_15.pdf": {
        (3, 156): (
            ("a)", "superscript"),
            ("b)", "superscript"),
            ("c)", "superscript"),
            ("d)", "superscript"),
        ),
    },
    "mixed_elements_pages_39_40.pdf": {
        (1, 44): (("i−1", "subscript"), ("i+1", "subscript")),
        (1, 46): (("i−1", "subscript"), ("i+1", "subscript")),
    },
}

_NO_SCRIPT_SOURCE_EXPECTATIONS = {
    "中文论文2.pdf": {
        (15, 15),
        (15, 35),
        (15, 37),
    }
}


def _origin_from_upright(
    origin: tuple[float, float],
    page_size: tuple[float, float],
    angle: int,
) -> tuple[float, float]:
    """把局部正向 origin 逆变换到页面坐标。"""
    x, y = origin
    page_width, page_height = page_size
    if angle == 270:
        return y, page_height - x
    if angle == 90:
        return page_width - y, x
    if angle == 180:
        return page_width - x, page_height - y
    return origin


def _script_fixture(
    *,
    angle: int = 0,
    formula_region: bool = True,
) -> tuple[_LineItem, dict[int, BBox], dict[int, tuple[float, float]], tuple[float, float]]:
    """构造同时含稳定上标和下标的 D-i-p 局部公式行。"""
    page_size = (100.0, 120.0)
    local_bboxes = (
        (10.0, 40.0, 20.0, 50.0),
        (20.0, 34.0, 26.0, 40.0),
        (20.0, 50.0, 26.0, 56.0),
    )
    local_origins = ((10.0, 49.0), (20.0, 40.0), (20.0, 56.0))
    page_bboxes = tuple(_rotate_bbox_from_upright(bbox, page_size, angle) for bbox in local_bboxes)
    page_origins = tuple(_origin_from_upright(origin, page_size, angle) for origin in local_origins)
    chars: list[Char] = [
        {
            "char": text,
            "char_idx": index,
            "bbox": Bbox(list(page_bboxes[index])),
            "rotation": 0.0,
            "font": {},
        }
        for index, text in enumerate("Dip")
    ]
    line = _LineItem(
        text="Dip",
        bbox=(
            min(bbox[0] for bbox in page_bboxes),
            min(bbox[1] for bbox in page_bboxes),
            max(bbox[2] for bbox in page_bboxes),
            max(bbox[3] for bbox in page_bboxes),
        ),
        angle=angle,
        source_index=0,
        chars=chars,
        inline_math_regions=[
            (
                min(bbox[0] for bbox in page_bboxes),
                min(bbox[1] for bbox in page_bboxes),
                max(bbox[2] for bbox in page_bboxes),
                max(bbox[3] for bbox in page_bboxes),
            )
        ]
        if formula_region
        else [],
    )
    return (
        line,
        dict(enumerate(page_bboxes)),
        dict(enumerate(page_origins)),
        page_size,
    )


def _compact_refinement_fixture(text: str) -> tuple[list[Char], dict[int, BBox], dict[int, tuple[float, float]]]:
    """构造同一 tight/origin 基线上的紧凑 token 精炼 fixture。"""
    chars: list[Char] = []
    tight_bboxes: dict[int, BBox] = {}
    origins: dict[int, tuple[float, float]] = {}
    for index, char_text in enumerate(text):
        left = 10.0 + index * 6.0
        bbox = (left, 40.0, left + 5.0, 46.0)
        chars.append(
            {
                "char": char_text,
                "char_idx": index,
                "bbox": Bbox(list(bbox)),
                "rotation": 0.0,
                "font": {},
            }
        )
        tight_bboxes[index] = bbox
        origins[index] = (left, 46.0)
    return chars, tight_bboxes, origins


def test_flash_compact_aligned_script_suffix_closes_complete_run() -> None:
    """验证可信 BASE 下标会把同基线的 ``-SAT`` 整体闭合。"""
    chars, tight_bboxes, origins = _compact_refinement_fixture("XBASE-SAT")

    roles = _refine_math_script_tokens(
        chars,
        ["body", *(["sub"] * len("BASE-SAT"))],
        tight_bboxes,
        origins,
        formula_region=False,
    )

    assert roles == ["body", *(["sub"] * len("BASE-SAT"))]


@pytest.mark.parametrize(
    ("text", "raw_roles"),
    [
        pytest.param("BASE-SAT", ["sub"] * len("BASE-SAT"), id="missing-body-anchor"),
        pytest.param("MODEL-SAT", ["body"] * len("MODEL-SAT"), id="plain-hyphenated-word"),
    ],
)
def test_flash_compact_script_suffix_requires_anchor_and_raw_geometry(
    text: str,
    raw_roles: list[ScriptRole],
) -> None:
    """验证无正文锚点或无原始角标证据时不会闭合连字符词。"""
    chars, tight_bboxes, origins = _compact_refinement_fixture(text)

    roles = _refine_math_script_tokens(
        chars,
        raw_roles,
        tight_bboxes,
        origins,
        formula_region=False,
    )

    assert roles == ["body"] * len(text)


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_flash_script_geometry_uses_upright_coordinates(angle: int) -> None:
    """验证四种页面方向下 loose/tight/origin 同步正向化后角色一致。"""
    line, tight_bboxes, origins, page_size = _script_fixture(angle=angle)

    script_lines = detect_pdf_text_script_lines(
        [line],
        page_size,
        tight_bboxes,
        origins,
    )

    assert [(item.start, item.end, item.style, item.formula_region) for item in script_lines[0].script_ranges] == [
        (1, 2, "superscript", True),
        (2, 3, "subscript", True),
    ]


def test_flash_formula_region_rebases_before_classification() -> None:
    """验证公式区域使用内部 D 基线，同时稳定输出 i 上标和 p 下标。"""
    line, tight_bboxes, origins, page_size = _script_fixture()

    script_line = detect_pdf_text_script_lines(
        [line],
        page_size,
        tight_bboxes,
        origins,
    )[0]

    assert all(item.stable_body_count == 1 for item in script_line.script_ranges)
    assert [item.style for item in script_line.script_ranges] == [
        "superscript",
        "subscript",
    ]


def test_flash_formula_region_without_internal_body_stays_plain() -> None:
    """验证公式区域全部字符同步偏移且无内部正文基线时不输出 style。"""
    line, tight_bboxes, origins, page_size = _script_fixture()
    line = replace(line, chars=line.chars[1:], text="ip")
    line.inline_math_regions = [line.bbox]

    script_line = detect_pdf_text_script_lines(
        [line],
        page_size,
        tight_bboxes,
        origins,
    )[0]

    assert script_line.script_ranges == ()


def test_flash_missing_extended_geometry_stays_plain() -> None:
    """验证 Flash 缺少 tight/origin 时不通过 loose bbox 猜测上下标。"""
    line, _tight_bboxes, _origins, page_size = _script_fixture()

    script_line = detect_pdf_text_script_lines([line], page_size, {}, {})[0]

    assert script_line.script_ranges == ()


def test_flash_fraction_bar_suppresses_stacked_script_candidate() -> None:
    """验证上下叠字被短横线分隔时按整处分式拒识上下标。"""
    page_size = (100.0, 120.0)
    line_chars: list[Char] = [
        {"char": "x", "char_idx": 0, "bbox": Bbox([10.0, 40.0, 20.0, 50.0]), "rotation": 0.0, "font": {}},
        {"char": "1", "char_idx": 1, "bbox": Bbox([22.0, 32.0, 30.0, 40.0]), "rotation": 0.0, "font": {}},
    ]
    denominator: Char = {
        "char": "M",
        "char_idx": 2,
        "bbox": Bbox([22.0, 44.0, 30.0, 52.0]),
        "rotation": 0.0,
        "font": {},
    }
    line = _LineItem(
        text="x1",
        bbox=(10.0, 32.0, 30.0, 52.0),
        angle=0,
        source_index=0,
        chars=line_chars,
    )
    tight_bboxes = {
        0: (10.0, 40.0, 20.0, 50.0),
        1: (22.0, 32.0, 30.0, 40.0),
        2: (22.0, 44.0, 30.0, 52.0),
    }
    origins = {0: (10.0, 49.0), 1: (22.0, 40.0), 2: (22.0, 52.0)}

    unfiltered = detect_pdf_text_script_lines([line], page_size, tight_bboxes, origins)[0]
    filtered = detect_pdf_text_script_lines(
        [line],
        page_size,
        tight_bboxes,
        origins,
        all_chars=[*line_chars, denominator],
        drawing_lines=[_AxisLine((21.0, 41.0, 31.0, 42.0), 1.0, "horizontal")],
    )[0]

    assert [(item.start, item.end, item.style) for item in unfiltered.script_ranges] == [(1, 2, "superscript")]
    assert filtered.script_ranges == ()


@pytest.mark.parametrize("text", ["Bm", "r1"])
def test_flash_formula_region_marks_only_the_index(text: str) -> None:
    """验证公式内部 B_m 与 r_1 只标记下移索引，不把整个 token 标成下标。"""
    line, tight_bboxes, origins, page_size = _script_fixture()
    body_char = {**line.chars[0], "char": text[0]}
    index_char = {**line.chars[2], "char": text[1]}
    line = replace(line, text=text, chars=[body_char, index_char])

    script_line = detect_pdf_text_script_lines(
        [line],
        page_size,
        tight_bboxes,
        origins,
    )[0]

    assert [(item.start, item.end, item.style) for item in script_line.script_ranges] == [(1, 2, "subscript")]


def test_flash_script_projection_combines_font_styles() -> None:
    """验证 Flash 上下标与已有粗体区间在同一 InlineSpan 流中组合。"""
    blocks = [
        {
            "type": "text",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "content": "Bm",
        }
    ]
    style_line = PDFTextStyleLine(
        bbox=(10.0, 10.0, 30.0, 20.0),
        text="Bm",
        style_ranges=(PDFTextStyleRange(0, 1, ("bold",)),),
        source_index=0,
    )
    script_line = PDFTextScriptLine(
        bbox=(10.0, 10.0, 30.0, 20.0),
        text="Bm",
        script_ranges=(
            PDFTextScriptRange(
                1,
                2,
                "subscript",
                (20.0, 15.0, 25.0, 20.0),
                1,
                True,
            ),
        ),
        source_index=0,
        angle=0,
    )

    apply_pdf_text_styles(blocks, [style_line], (100.0, 100.0))
    apply_pdf_text_scripts(blocks, [script_line], (100.0, 100.0))
    materialize_pdf_inline_spans(blocks)

    assert blocks[0]["content"] == [
        {"type": "text", "content": "B", "styles": ["bold"]},
        {"type": "text", "content": "m", "styles": ["subscript"]},
    ]


def test_flash_script_projection_does_not_leak_to_same_text() -> None:
    """验证公式内候选按整行区间投影，不会回退到同行同名普通字符。"""
    blocks = [
        {
            "type": "text",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "content": "Bm Bm",
        }
    ]
    script_line = PDFTextScriptLine(
        bbox=(10.0, 10.0, 90.0, 20.0),
        text="BmBm",
        script_ranges=(PDFTextScriptRange(3, 4, "subscript", (70.0, 15.0, 75.0, 20.0), 1, True),),
        source_index=0,
        angle=0,
    )

    apply_pdf_text_scripts(blocks, [script_line], (100.0, 100.0))
    materialize_pdf_inline_spans(blocks)

    assert blocks[0]["content"] == [
        {"type": "text", "content": "Bm B"},
        {"type": "text", "content": "m", "styles": ["subscript"]},
    ]


def test_flash_script_projection_combines_hyperlink() -> None:
    """验证超链接与上下标区间共同物化时保持子 Span 顺序和 URL。"""
    blocks = [
        {
            "type": "text",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "content": "x2",
        }
    ]
    link_line = PDFTextLinkLine(
        bbox=(10.0, 10.0, 30.0, 20.0),
        text="x2",
        link_ranges=(PDFTextLinkRange(0, 2, "https://example.test/x2"),),
        source_index=0,
    )
    script_line = PDFTextScriptLine(
        bbox=link_line.bbox,
        text="x2",
        script_ranges=(PDFTextScriptRange(1, 2, "superscript", (20.0, 10.0, 25.0, 15.0), 1, False),),
        source_index=0,
        angle=0,
    )

    apply_pdf_text_links(blocks, [link_line], (100.0, 100.0))
    apply_pdf_text_scripts(blocks, [script_line], (100.0, 100.0))
    materialize_pdf_inline_spans(blocks)

    assert blocks[0]["content"] == [
        {
            "type": "hyperlink",
            "content": [
                {"type": "text", "content": "x", "styles": []},
                {"type": "text", "content": "2", "styles": ["superscript"]},
            ],
            "url": "https://example.test/x2",
        }
    ]


@pytest.mark.parametrize("block_type", ["table", "code", "equation", "image"])
def test_flash_scripts_exclude_non_natural_blocks(block_type: str) -> None:
    """验证表格、代码、独立公式和图片容器不接收 Flash 上下标。"""
    blocks = [
        {
            "type": block_type,
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "content": "Bm",
        }
    ]
    script_line = PDFTextScriptLine(
        bbox=(10.0, 10.0, 30.0, 20.0),
        text="Bm",
        script_ranges=(
            PDFTextScriptRange(
                1,
                2,
                "subscript",
                (20.0, 15.0, 25.0, 20.0),
                1,
                False,
            ),
        ),
        source_index=0,
        angle=0,
    )

    apply_pdf_text_scripts(blocks, [script_line], (100.0, 100.0))
    materialize_pdf_inline_spans(blocks)

    assert blocks[0]["content"] == "Bm"


def test_late_inline_math_region_drops_unrebased_candidate() -> None:
    """验证后续恢复的公式区域会过滤未在公式内部重新基线的候选。"""
    blocks = [
        {
            "type": "text",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "content": "Bm",
            "_inline_math_regions": [[0.15, 0.1, 0.3, 0.3]],
        }
    ]
    script_line = PDFTextScriptLine(
        bbox=(10.0, 10.0, 30.0, 30.0),
        text="Bm",
        script_ranges=(
            PDFTextScriptRange(
                1,
                2,
                "subscript",
                (20.0, 15.0, 25.0, 20.0),
                1,
                False,
            ),
        ),
        source_index=0,
        angle=0,
    )

    apply_pdf_text_scripts(blocks, [script_line], (100.0, 100.0))
    materialize_pdf_inline_spans(blocks)

    assert blocks[0]["content"] == [{"type": "text", "content": "Bm"}]
    assert "_inline_math_regions" not in blocks[0]


def _flash_script_runs(pdf_name: str) -> list[tuple[str, tuple[str, ...]]]:
    """解析真实 Flash PDF 并收集最终 TextSpan 上下标。"""
    with PDFDocument(str(_DEMO_PDF_DIR / pdf_name)) as document:
        pages = PdfModel().predict(document)
    runs: list[tuple[str, tuple[str, ...]]] = []

    def walk(value: Any) -> None:
        """递归收集最终 TextSpan style。"""
        if isinstance(value, dict):
            styles = tuple(str(style) for style in value.get("styles", []))
            content = value.get("content")
            if value.get("type") == "text" and isinstance(content, str) and {"superscript", "subscript"}.intersection(styles):
                runs.append((content, styles))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(pages)
    return runs


@lru_cache(maxsize=None)
def _flash_script_diagnostics(pdf_name: str) -> tuple[dict[str, Any], ...]:
    """解析真实 Flash PDF 并缓存逐行上下标 sidecar。"""
    diagnostics: list[dict[str, Any]] = []
    with PDFDocument(str(_DEMO_PDF_DIR / pdf_name)) as document:
        _analyze_native_document(document, script_diagnostics=diagnostics)
    return tuple(diagnostics)


@pytest.mark.parametrize("pdf_name", tuple(_REVIEWED_SCRIPT_EXPECTATIONS))
def test_user_reviewed_flash_script_ranges(pdf_name: str) -> None:
    """逐项锁定人工审阅反馈涉及的 base、索引、分式和复杂公式边界。"""
    diagnostics = _flash_script_diagnostics(pdf_name)

    for (page_index, source_index), expected in _REVIEWED_SCRIPT_EXPECTATIONS[pdf_name].items():
        script_line = next(line for line in diagnostics[page_index]["script_lines"] if line.source_index == source_index)
        actual = tuple(
            (script_line.text[script_range.start : script_range.end], script_range.style)
            for script_range in script_line.script_ranges
        )
        assert actual == expected, (page_index + 1, source_index, script_line.text)
        materialized = tuple(
            (item["text"], item["role"])
            for item in diagnostics[page_index]["materialized_ranges"]
            if item["source_index"] == source_index
        )
        assert materialized == expected, (page_index + 1, source_index, script_line.text)


@pytest.mark.parametrize("pdf_name", tuple(_RUN_CLOSURE_EXPECTATIONS))
def test_user_reviewed_flash_script_run_closures(pdf_name: str) -> None:
    """验证运算符、小数点和作者括号与同基线角标保持为连续 run。"""
    diagnostics = _flash_script_diagnostics(pdf_name)

    for (page_index, source_index), expected in _RUN_CLOSURE_EXPECTATIONS[pdf_name].items():
        script_line = next(line for line in diagnostics[page_index]["script_lines"] if line.source_index == source_index)
        actual = tuple(
            (script_line.text[script_range.start : script_range.end], script_range.style)
            for script_range in script_line.script_ranges
        )
        materialized = tuple(
            (item["text"], item["role"])
            for item in diagnostics[page_index]["materialized_ranges"]
            if item["source_index"] == source_index
        )
        assert all(item in actual for item in expected), (page_index + 1, source_index, script_line.text)
        assert all(item in materialized for item in expected), (page_index + 1, source_index, script_line.text)


@pytest.mark.parametrize("pdf_name", tuple(_NO_SCRIPT_SOURCE_EXPECTATIONS))
def test_user_reviewed_unicode_math_tokens_stay_body(pdf_name: str) -> None:
    """验证没有内部 base 的独立 Greek token 不因 CJK 正文基线而成为下标。"""
    diagnostics = _flash_script_diagnostics(pdf_name)

    for page_index, source_index in _NO_SCRIPT_SOURCE_EXPECTATIONS[pdf_name]:
        script_line = next(line for line in diagnostics[page_index]["script_lines"] if line.source_index == source_index)
        assert script_line.script_ranges == ()
        assert not any(item["source_index"] == source_index for item in diagnostics[page_index]["materialized_ranges"])


def test_chinese_paper_page_12_recovers_numbered_formula_regions() -> None:
    """验证栏顶公式 6、7、8 被完整认领为 equation，且不吸收相邻正文。"""
    diagnostics: list[dict[str, Any]] = []
    with PDFDocument(str(_DEMO_PDF_DIR / "中文论文2.pdf")) as document:
        pages = _analyze_native_document(document, script_diagnostics=diagnostics)

    equations = {
        tag: block
        for block in pages[11]
        if block.get("type") == "equation" and isinstance((content := block.get("content")), str)
        for tag in ("6", "7", "8")
        if f"\\tag{{{tag}}}" in content
    }
    assert set(equations) == {"6", "7", "8"}
    assert 0.6 < equations["6"]["bbox"][0] < 0.7
    assert equations["6"]["bbox"][1] < equations["7"]["bbox"][1] < equations["8"]["bbox"][1]
    assert all(block["bbox"][2] > 0.9 for block in equations.values())
    assert all("如下" not in block["content"] and "然后根据" not in block["content"] for block in equations.values())
    assert not any(item["source_index"] in {61, 64} for item in diagnostics[11]["materialized_ranges"])


@pytest.mark.parametrize(
    ("pdf_name", "expected"),
    [
        pytest.param(
            "中文论文.pdf",
            (("2", "superscript"), ("［1］", "superscript"), ("-3", "superscript")),
            id="chinese-paper",
        ),
        pytest.param(
            "中文论文2.pdf",
            (("[1]", "superscript"), ("[82]", "superscript"), ("i", "subscript")),
            id="chinese-paper-2",
        ),
        pytest.param(
            "demo3.pdf",
            (
                ("i", "subscript"),
                ("BASE", "subscript"),
                ("LARGE", "subscript"),
                ("BASE-SAT", "subscript"),
                ("BASE-SO", "subscript"),
            ),
            id="model-subscripts",
        ),
        pytest.param(
            "demo4.pdf",
            (("18", "superscript"), ("2", "superscript")),
            id="isotope-and-unit",
        ),
    ],
)
def test_real_flash_pdfs_materialize_confirmed_scripts(
    pdf_name: str,
    expected: tuple[tuple[str, str], ...],
) -> None:
    """验证真实 Flash PDF 的已确认上下标进入最终 InlineSpan。"""
    runs = _flash_script_runs(pdf_name)

    assert all(
        any(content == expected_content and role in styles for content, styles in runs) for expected_content, role in expected
    )


def test_real_flash_plain_layouts_do_not_gain_scripts() -> None:
    """验证财经图表样本的旋转和容器文本不会产生 Flash 上下标。"""
    assert _flash_script_runs("caibao1.pdf") == []


def test_zh2_normal_english_words_stay_plain_in_flash() -> None:
    """验证中文论文2的普通混合字体英文不会被 Flash 误标上下标。"""
    styled_text = {content for content, _styles in _flash_script_runs("中文论文2.pdf")}

    assert styled_text.isdisjoint({"Source", "Hypothesis", "Reference", "BLEU", "ROUGE"})
