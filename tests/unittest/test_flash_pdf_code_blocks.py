from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mineru.model.flash import PdfModel
from mineru.model.flash.pdf import code_blocks, models, pipeline
from mineru.backend.postprocess.inline import (
    inline_plain_text,
    parse_inline_content,
)
from mineru.model.flash.pdf.document import PDFDocument, PDFPathInfo
from mineru.model.flash.pdf.spatial_text import project_pdf_spatial_text

from _flash_pdf_test_utils import _text_line


def _mono_line(
    text: str,
    bbox: tuple[float, float, float, float],
    source_index: int,
    *,
    font_name: str = "TestMono",
    glyph_widths: list[float] | None = None,
) -> models._LineItem:
    """构造带原生字符框的等宽或比例字体测试行。"""

    widths = glyph_widths or [5.0] * len(text)
    line = _text_line(
        text,
        bbox,
        source_index,
        effective_height=bbox[3] - bbox[1],
        font_signature=(font_name, 0),
        font_coverage=1.0,
        median_glyph_width=5.0,
    )
    x = bbox[0]
    chars: list[dict[str, Any]] = []
    for char_offset, (value, width) in enumerate(zip(text, widths, strict=True)):
        chars.append(
            {
                "bbox": (x, bbox[1], x + width, bbox[3]),
                "char": value,
                "rotation": 0.0,
                "font": {
                    "name": font_name,
                    "flags": 0,
                    "size": bbox[3] - bbox[1],
                    "weight": 400,
                },
                "char_idx": source_index * 100 + char_offset,
            }
        )
        x += width
    line.chars = chars
    return line


def _code_source(
    *lines: models._LineItem,
    fill_rgba: tuple[int, int, int, int] = (242, 242, 255, 255),
) -> models._PageSource:
    """构造带大幅浅色填充背景的代码页测试源。"""

    return models._PageSource(
        page_size=(200.0, 100.0),
        lines=list(lines),
        chars=[char for line in lines for char in line.chars],
        drawing_lines=[],
        path_infos=[
            PDFPathInfo(
                bbox=(10.0, 10.0, 190.0, 70.0),
                segment_count=5,
                fill_visible=True,
                stroke_visible=False,
                form_depth=0,
                source_index=0,
                fill_rgba=fill_rgba,
            )
        ],
    )


def test_colored_monospace_region_materializes_code_and_claims_text() -> None:
    """验证浅色等宽区域输出 code，并且内部文本不再重复进入正文。"""

    first = _mono_line("alpha", (15.0, 15.0, 40.0, 23.0), 0)
    second = _mono_line("beta", (20.0, 35.0, 40.0, 43.0), 1)
    source = _code_source(first, second)

    prepared = pipeline._prepare_page_source(source)
    blocks = pipeline._finalize_prepared_page(prepared, page_index=0)

    assert prepared.remaining_lines == []
    assert len(blocks) == 1
    assert blocks[0]["type"] == "code"
    assert blocks[0]["content"] == "alpha\n\n beta"


@pytest.mark.parametrize("rejection", ["white", "proportional", "table_overlap"])
def test_code_candidate_rejects_missing_background_or_monospace_evidence(
    rejection: str,
) -> None:
    """验证白框、比例字体和已确认表格均不会被升级为代码块。"""

    if rejection == "proportional":
        first = _mono_line(
            "alpha",
            (15.0, 15.0, 42.0, 23.0),
            0,
            font_name="Body",
            glyph_widths=[3.0, 7.0, 4.0, 8.0, 5.0],
        )
    else:
        first = _mono_line("alpha", (15.0, 15.0, 40.0, 23.0), 0)
    second = _mono_line("beta", (15.0, 35.0, 35.0, 43.0), 1)
    source = _code_source(
        first,
        second,
        fill_rgba=(255, 255, 255, 255) if rejection == "white" else (242, 242, 255, 255),
    )
    excluded = [(10.0, 10.0, 190.0, 70.0)] if rejection == "table_overlap" else []

    blocks, claimed = code_blocks._build_code_blocks(source, excluded, set())

    assert blocks == []
    assert claimed == set()


def test_code_projection_preserves_columns_and_blank_rows() -> None:
    """验证通用 PDF 空间投影保留等宽缩进、双列关系和明显空行。"""

    left = _mono_line("left", (5.0, 5.0, 25.0, 13.0), 0)
    right = _mono_line("right", (55.0, 5.0, 80.0, 13.0), 1)
    tail = _mono_line("tail", (10.0, 30.0, 30.0, 38.0), 2)
    chars = [char for line in (left, right, tail) for char in line.chars]

    content = project_pdf_spatial_text(
        chars,
        (0.0, 0.0, 100.0, 50.0),
        preserve_blank_rows=True,
    )

    assert content == "left      right\n\n tail"


def _rule_delimited_code_source(
    *,
    include_internal_grid: bool = False,
    page_height: float = 300.0,
    touch_right_edge: bool = False,
) -> models._PageSource:
    """构造带上下边界、行号槽和可选内部网格线的代码清单页面。"""

    lines: list[models._LineItem] = []
    for row_index, top in enumerate((30.0, 42.0, 54.0, 66.0, 78.0, 90.0)):
        number = _text_line(
            str(row_index + 1),
            (25.0, top, 31.0, top + 8.0),
            2 * row_index,
            visual_row_id=row_index,
            split_from_row=True,
            effective_height=8.0,
        )
        statement = _text_line(
            f"statement {row_index}",
            (
                42.0 + 6.0 * (row_index % 3),
                top,
                180.0 if touch_right_edge else 125.0,
                top + 8.0,
            ),
            2 * row_index + 1,
            visual_row_id=row_index,
            split_from_row=True,
            effective_height=8.0,
        )
        lines.extend((number, statement))
    drawing_lines = [
        models._AxisLine((20.0, 20.0, 180.0, 21.0), 1.0, "horizontal"),
        models._AxisLine((20.0, 104.0, 180.0, 105.0), 1.0, "horizontal"),
    ]
    if include_internal_grid:
        drawing_lines.append(
            models._AxisLine(
                (20.0, 60.0, 180.0, 61.0),
                1.0,
                "horizontal",
            )
        )
    return models._PageSource(
        page_size=(200.0, page_height),
        lines=lines,
        chars=[],
        drawing_lines=drawing_lines,
    )


def _full_width_indent_only_source() -> models._PageSource:
    """构造没有行号槽、仅靠多层缩进且横向占满的宽幅文本候选。"""

    lines = [
        _text_line(
            f"paragraph {row_index}",
            (20.0 + 12.0 * (row_index % 3), top, 180.0, top + 8.0),
            row_index,
            visual_row_id=row_index,
            effective_height=8.0,
        )
        for row_index, top in enumerate((30.0, 42.0, 54.0, 66.0, 78.0, 90.0))
    ]
    return models._PageSource(
        page_size=(200.0, 200.0),
        lines=lines,
        chars=[],
        drawing_lines=[
            models._AxisLine((20.0, 20.0, 180.0, 21.0), 1.0, "horizontal"),
            models._AxisLine((20.0, 104.0, 180.0, 105.0), 1.0, "horizontal"),
        ],
    )


def test_rule_delimited_listing_materializes_code_before_table_claim() -> None:
    """验证无内部网格的上下横线清单形成单一 code 并唯一认领全部来源行。"""

    source = _rule_delimited_code_source()
    blocks, claimed = code_blocks._build_rule_delimited_code_blocks(
        source,
        [],
    )

    assert len(blocks) == 1
    assert blocks[0]["type"] == "code"
    assert claimed == set(range(12))


def test_tall_full_width_rule_delimited_listing_materializes_code() -> None:
    """验证高占比且文本触边的强行号槽清单仍能形成单一 code。"""

    source = _rule_delimited_code_source(
        page_height=200.0,
        touch_right_edge=True,
    )

    blocks, claimed = code_blocks._build_rule_delimited_code_blocks(
        source,
        [],
    )

    assert len(blocks) == 1
    assert blocks[0]["type"] == "code"
    assert claimed == set(range(12))


def test_tall_full_width_indent_only_listing_is_rejected() -> None:
    """验证没有行号槽的宽幅缩进正文仍受占宽保护，不会误判为 code。"""

    blocks, claimed = code_blocks._build_rule_delimited_code_blocks(
        _full_width_indent_only_source(),
        [],
    )

    assert blocks == []
    assert claimed == set()


def test_rule_delimited_listing_rejects_real_table_internal_grid() -> None:
    """验证带内部横向网格的区域不会被规则代码路径侵蚀。"""

    blocks, claimed = code_blocks._build_rule_delimited_code_blocks(
        _rule_delimited_code_source(include_internal_grid=True),
        [],
    )

    assert blocks == []
    assert claimed == set()


def test_rule_delimited_listing_rejects_vertical_track_spanning_candidate() -> None:
    """验证跨越上下边界的长竖轨按候选高度计量并否决伪代码区域。"""

    source = _rule_delimited_code_source()
    source.drawing_lines.append(
        models._AxisLine(
            (70.0, 0.0, 71.0, 250.0),
            1.0,
            "vertical",
        )
    )

    blocks, claimed = code_blocks._build_rule_delimited_code_blocks(
        source,
        [],
    )

    assert (
        code_blocks._vertical_rule_candidate_height_coverage(
            source.drawing_lines[-1].bbox,
            (20.0, 20.0, 180.0, 105.0),
        )
        == 1.0
    )
    assert blocks == []
    assert claimed == set()


def test_kvcache_algorithm_pdf_materializes_caption_and_code() -> None:
    """验证真实长算法页面输出相邻 caption/code，且来源文本不再重复。"""

    pdf_path = Path(__file__).parents[2] / "demo" / "pdfs" / "2407.00079v4_origi-10.pdf"
    with PDFDocument(str(pdf_path)) as pdf_doc:
        page = PdfModel().predict(pdf_doc)[0]

    caption_text = "Algorithm 1 KVCache-centric Scheduling Algorithm"
    captions = [
        block
        for block in page
        if block["type"] == "caption"
        and inline_plain_text(
            parse_inline_content(str(block.get("content") or ""))
        )
        == caption_text
    ]
    code = [block for block in page if block["type"] == "code"]

    assert len(captions) == 1
    assert captions[0]["bbox"] == [0.176, 0.094, 0.528, 0.106]
    assert len(code) == 1
    assert code[0]["bbox"] == [0.176, 0.108, 0.824, 0.539]
    assert page.index(captions[0]) + 1 == page.index(code[0])
    assert not [
        block
        for block in page
        if block["type"] == "header"
        and inline_plain_text(
            parse_inline_content(str(block.get("content") or ""))
        )
        == caption_text
    ]

    code_content = str(code[0]["content"])
    probes = (
        "Input: prefill instance pool P",
        "1: block_keys",
        "31: return (p, d)",
        "KVCache hot-spot migration",
    )
    assert all(probe in code_content for probe in probes)
    for probe in probes:
        assert sum(probe in str(block.get("content", "")) for block in page) == 1
