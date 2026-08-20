from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock
from zipfile import ZipFile

import pytest

from mineru.backend.analysis.pdf import window
from mineru.backend.analysis.pdf.text import content as text_content
from mineru.backend.analysis.pdf.text.models import _AnalyzeLine, _AnalyzeSpan
from mineru.backend.analysis.pdf.text.native import txt_spans_extract
from mineru.backend.postprocess.pages import model_list_to_pages
from mineru.model.flash import PdfModel
from mineru.render import render_docx, render_html, render_markdown
from mineru.types import BlockType, ContentType, MiddleJson
from mineru.utils.pdf_document import PDFDocument
from mineru.utils.pdf_text_styles import (
    PDFTextStyleLine,
    PDFTextStyleRange,
    apply_pdf_strikethrough_styles,
    detect_pdf_strikethrough_lines,
)


def _build_native_text_style_pdf() -> bytes:
    """构造包含删除线、下划线、整栏横线和短横线的单页原生 PDF。"""

    content = b"""0.8 w
BT /F1 12 Tf 50 250 Td (strike: alpha ) Tj (deleted) Tj ( omega) Tj ET
150.8 254 m 201.2 254 l S
BT /F1 12 Tf 50 210 Td (underline: alpha ) Tj (underlined) Tj ( omega) Tj ET
172.4 208 m 244.4 208 l S
BT /F1 12 Tf 50 170 Td (separator: alpha crossed omega) Tj ET
20 174 m 400 174 l S
BT /F1 12 Tf 50 130 Td (short: alpha ) Tj (x) Tj ( omega) Tj ET
143.6 134 m 150.8 134 l S
BT /F1 12 Tf 50 90 Td (filled: alpha ) Tj (filled strike) Tj ( omega) Tj ET
150.8 93.4 93.6 1.2 re f
"""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 420 300] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"endstream",
    ]
    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_index, body in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{object_index} 0 obj\n".encode("ascii"))
        payload.extend(body)
        payload.extend(b"\nendobj\n")
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(payload)


def _text_line(
    text: str,
    *,
    x: float = 10.0,
    y: float = 10.0,
    height: float = 10.0,
    angle: int = 0,
    source_index: int = 0,
) -> SimpleNamespace:
    """构造带真实字符 bbox 的轻量视觉文本行。"""

    chars = []
    cursor = x
    for char_index, char in enumerate(text):
        width = 3.0 if char.isspace() else 6.0
        chars.append(
            {
                "char": char,
                "bbox": (cursor, y, cursor + width, y + height),
                "char_idx": char_index,
            }
        )
        cursor += width
    return SimpleNamespace(
        text=text,
        bbox=(x, y, cursor, y + height),
        angle=angle,
        source_index=source_index,
        chars=chars,
    )


def _drawing(
    x0: float,
    x1: float,
    y: float,
    *,
    width: float = 1.0,
    orientation: str = "horizontal",
) -> SimpleNamespace:
    """构造删除线检测使用的轻量 drawing。"""

    return SimpleNamespace(
        bbox=(x0, y - width / 2, x1, y + width / 2),
        width=width,
        orientation=orientation,
    )


def _char_span(line: SimpleNamespace, start: int, end: int) -> tuple[float, float]:
    """返回指定字符区间的左右边缘。"""

    return line.chars[start]["bbox"][0], line.chars[end - 1]["bbox"][2]


def test_detects_long_strikethrough_and_preserves_internal_spaces() -> None:
    """验证长删除线按字符范围生成单个紧凑样式区间。"""

    line = _text_line("alpha deleted text omega")
    start = line.text.index("deleted")
    end = start + len("deleted text")
    x0, x1 = _char_span(line, start, end)

    detected = detect_pdf_strikethrough_lines(
        [line],
        [_drawing(x0, x1, 15.0)],
    )

    assert len(detected) == 1
    assert detected[0].text == "alphadeletedtextomega"
    assert detected[0].text[
        detected[0].strikethrough_ranges[0].start : detected[0].strikethrough_ranges[0].end
    ] == "deletedtext"


def test_rejects_underline_thick_short_and_column_rule() -> None:
    """验证下划线、粗线、短线和贯穿整栏的分隔线均不产生样式。"""

    line = _text_line("alpha deleted omega")
    start = line.text.index("deleted")
    end = start + len("deleted")
    x0, x1 = _char_span(line, start, end)
    short_line = _text_line("alpha xy omega", y=30.0, source_index=1)
    short_start = short_line.text.index("xy")
    short_x0, short_x1 = _char_span(short_line, short_start, short_start + 2)

    detected = detect_pdf_strikethrough_lines(
        [line, short_line],
        [
            _drawing(x0, x1, 19.5),
            _drawing(x0, x1, 15.0, width=2.1),
            _drawing(short_x0, short_x1, 35.0),
            _drawing(0.0, 200.0, 15.0),
        ],
    )

    assert all(not item.strikethrough_ranges for item in detected)


def test_accepts_one_aligned_endpoint_and_merges_adjacent_drawings() -> None:
    """验证尾部延长 drawing 可由左端点确认，邻接命中区间会合并。"""

    line = _text_line("alpha deleted text omega")
    deleted_start = line.text.index("deleted")
    deleted_end = deleted_start + len("deleted")
    text_start = line.text.index("text")
    text_end = text_start + len("text")
    deleted_x0, deleted_x1 = _char_span(line, deleted_start, deleted_end)
    text_x0, text_x1 = _char_span(line, text_start, text_end)

    detected = detect_pdf_strikethrough_lines(
        [line],
        [
            _drawing(deleted_x0, deleted_x1 + 5.0, 15.0),
            _drawing(text_x0, text_x1, 15.0),
        ],
    )

    assert len(detected[0].strikethrough_ranges) == 1
    style_range = detected[0].strikethrough_ranges[0]
    assert detected[0].text[style_range.start : style_range.end] == "deletedtext"


def test_keeps_two_disjoint_strikethrough_ranges() -> None:
    """验证同一文本行内被普通字符隔开的两条删除线保持两个样式区间。"""

    line = _text_line("alpha deleted middle removed omega")
    deleted_start = line.text.index("deleted")
    removed_start = line.text.index("removed")
    deleted_x0, deleted_x1 = _char_span(
        line,
        deleted_start,
        deleted_start + len("deleted"),
    )
    removed_x0, removed_x1 = _char_span(
        line,
        removed_start,
        removed_start + len("removed"),
    )

    detected = detect_pdf_strikethrough_lines(
        [line],
        [
            _drawing(deleted_x0, deleted_x1, 15.0),
            _drawing(removed_x0, removed_x1, 15.0),
        ],
    )

    assert [
        detected[0].text[style_range.start : style_range.end]
        for style_range in detected[0].strikethrough_ranges
    ] == ["deleted", "removed"]


def test_drawing_is_assigned_to_the_closest_overlapping_line() -> None:
    """验证一条 drawing 同时接近两行时只归属中线距离更小的文本行。"""

    first = _text_line("deleted", y=10.0, source_index=0)
    second = _text_line("deleted", y=11.0, source_index=1)
    x0, x1 = _char_span(first, 0, len(first.text))

    detected = detect_pdf_strikethrough_lines(
        [first, second],
        [_drawing(x0, x1, 15.1)],
    )

    ranges_by_source = {
        line.source_index: line.strikethrough_ranges
        for line in detected
    }
    assert ranges_by_source[0] == (PDFTextStyleRange(0, 7),)
    assert ranges_by_source[1] == ()


def test_rotated_text_is_not_a_style_candidate() -> None:
    """验证首版忽略真正旋转的视觉文本。"""

    line = _text_line("rotated text", angle=90)

    assert detect_pdf_strikethrough_lines(
        [line],
        [_drawing(10.0, 80.0, 15.0)],
    ) == []


def test_invalid_line_and_char_bboxes_are_ignored() -> None:
    """验证退化 line 或 char bbox 不会进入删除线检测。"""

    invalid_line = SimpleNamespace(
        bbox=(10.0, 10.0, 10.0, 20.0),
        angle=0,
        source_index=0,
        chars=[{"char": "x", "bbox": (10.0, 10.0, 10.0, 20.0)}],
    )

    assert detect_pdf_strikethrough_lines(
        [invalid_line],
        [_drawing(10.0, 40.0, 15.0)],
    ) == []


def test_applies_style_to_plain_content_and_duplicate_second_line() -> None:
    """验证物理行顺序可将重复文本的删除线写到第二次出现位置。"""

    blocks = [
        {
            "type": "text",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "content": "deleted deleted",
        }
    ]
    lines = [
        PDFTextStyleLine((10.0, 10.0, 50.0, 20.0), "deleted", (), 0),
        PDFTextStyleLine(
            (10.0, 30.0, 50.0, 40.0),
            "deleted",
            (PDFTextStyleRange(0, 7),),
            1,
        ),
    ]

    apply_pdf_strikethrough_styles(blocks, lines, (100.0, 100.0))

    assert blocks[0]["content"] == 'deleted <text style="strikethrough">deleted</text>'


def test_applies_style_inside_superscript_without_crossing_tags() -> None:
    """验证删除线只包装 sup 内的文本叶子，不破坏原有上下标标签。"""

    blocks = [
        {
            "type": "text",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "content": "value <sup>deleted</sup> tail",
        }
    ]
    lines = [
        PDFTextStyleLine(
            (10.0, 10.0, 90.0, 20.0),
            "valuedeletedtail",
            (PDFTextStyleRange(5, 12),),
            0,
        )
    ]

    apply_pdf_strikethrough_styles(blocks, lines, (100.0, 100.0))

    assert blocks[0]["content"] == (
        'value <sup><text style="strikethrough">deleted</text></sup> tail'
    )


def test_never_styles_equations_or_excluded_blocks() -> None:
    """验证公式内容与 table block 不会被 PDF 删除线富化。"""

    blocks = [
        {
            "type": "text",
            "bbox": [0.0, 0.0, 1.0, 0.5],
            "content": "before <eq>deleted</eq> after",
        },
        {
            "type": "table",
            "bbox": [0.0, 0.5, 1.0, 1.0],
            "content": "deleted",
        },
    ]
    lines = [
        PDFTextStyleLine(
            (10.0, 10.0, 90.0, 20.0),
            "beforedeletedafter",
            (PDFTextStyleRange(6, 13),),
            0,
        ),
        PDFTextStyleLine(
            (10.0, 60.0, 90.0, 70.0),
            "deleted",
            (PDFTextStyleRange(0, 7),),
            1,
        ),
    ]

    apply_pdf_strikethrough_styles(blocks, lines, (100.0, 100.0))

    assert blocks[0]["content"] == "before <eq>deleted</eq> after"
    assert blocks[1]["content"] == "deleted"


def test_flash_native_pdf_strikethrough_reaches_model_middle_and_renderers() -> None:
    """验证真实 PDF drawing 删除线贯穿 model、MiddleJson 和三种渲染输出。"""

    document = PDFDocument(_build_native_text_style_pdf())
    try:
        model_list = PdfModel().predict(document)
    finally:
        document.close()
    model_contents = [
        str(block.get("content") or "")
        for page in model_list
        for block in page
    ]
    joined_model_content = "\n".join(model_contents)
    assert '<text style="strikethrough">deleted</text>' in joined_model_content
    assert '<text style="strikethrough">filled strike</text>' in joined_model_content
    assert "underlined" in joined_model_content
    assert '<text style="strikethrough">underlined</text>' not in joined_model_content
    assert '<text style="strikethrough">x</text>' not in joined_model_content
    assert '<text style="strikethrough">separator' not in joined_model_content

    middle = MiddleJson(
        pages=model_list_to_pages(model_list),
        file_suffix="pdf",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )
    markdown = render_markdown(middle)
    html = render_html(middle, standalone=False)
    document_xml = ZipFile(BytesIO(render_docx(middle))).read(
        "word/document.xml"
    ).decode("utf-8")

    assert "~~deleted~~" in markdown
    assert "<s>deleted</s>" in html
    assert "<w:strike" in document_xml


def test_hybrid_txt_reuses_loaded_chars_and_applies_styles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Medium/High/XHigh 共用 TXT 回填入口复用 chars 并写入删除线。"""

    page_chars = [{"char": "d", "bbox": (10.0, 10.0, 16.0, 20.0), "char_idx": 0}]
    style_lines = [
        PDFTextStyleLine(
            (10.0, 10.0, 52.0, 20.0),
            "deleted",
            (PDFTextStyleRange(0, 7),),
            0,
        )
    ]
    build_styles = MagicMock(return_value=(page_chars, [], style_lines))
    observed_chars: list[object] = []

    def fake_fill_native(
        _pdf_page: object,
        _page_spans: object,
        _page_image: object,
        _scale: object,
        _page_size: object,
        *,
        page_chars: object,
    ) -> list[_AnalyzeSpan]:
        """记录传入原生回填的 chars，并返回稳定文本 span。"""

        observed_chars.append(page_chars)
        return [
            _AnalyzeSpan(
                type=ContentType.TEXT,
                bbox=(10.0, 10.0, 52.0, 20.0),
                content="deleted",
                score=1.0,
            )
        ]

    monkeypatch.setattr(
        text_content,
        "build_pdf_native_visual_lines_and_styles",
        build_styles,
    )
    monkeypatch.setattr(text_content, "_fill_native_pdf_text_spans", fake_fill_native)
    monkeypatch.setattr(
        text_content,
        "_build_page_text_formula_spans",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        text_content,
        "_group_page_spans_by_block",
        lambda *_args: {
            0: [
                _AnalyzeLine(
                    bbox=(10.0, 10.0, 52.0, 20.0),
                    spans=[
                        _AnalyzeSpan(
                            type=ContentType.TEXT,
                            bbox=(10.0, 10.0, 52.0, 20.0),
                            content="deleted",
                            score=1.0,
                        )
                    ],
                )
            ]
        },
    )
    pdf_page = MagicMock(size=(100.0, 100.0))
    pdf_page.get_char_count.return_value = 1
    model_list = [[{"type": BlockType.TEXT, "bbox": [0.1, 0.1, 0.52, 0.2], "content": ""}]]

    text_content._fill_window_block_content_and_lines(
        [{"img_pil": object(), "scale": 1.0}],
        [pdf_page],
        model_list,
        [[]],
        [[]],
        "txt",
        {BlockType.TEXT},
        MagicMock(),
    )

    assert observed_chars == [page_chars]
    assert model_list[0][0]["content"] == '<text style="strikethrough">deleted</text>'


def test_low_txt_applies_styles_after_native_content_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Low/TXT 在 block content 完成后应用同一删除线协议。"""

    span = _AnalyzeSpan(
        type=ContentType.TEXT,
        bbox=(10.0, 10.0, 52.0, 20.0),
        content="deleted",
        score=1.0,
    )
    style_line = PDFTextStyleLine(
        (10.0, 10.0, 52.0, 20.0),
        "deleted",
        (PDFTextStyleRange(0, 7),),
        0,
    )
    monkeypatch.setattr(
        window,
        "_build_pdf_text_visual_run_data",
        lambda _page: ([span], [style_line]),
    )
    monkeypatch.setattr(window, "_fill_low_table_contents", lambda *_args: None)
    monkeypatch.setattr(
        window,
        "_fill_low_txt_native_formula_contents",
        lambda *_args: None,
    )
    model_list = [[{"type": BlockType.TEXT, "bbox": [0.1, 0.1, 0.52, 0.2], "content": ""}]]

    window._process_low_text(
        [{"img_pil": object(), "scale": 1.0}],
        [MagicMock(size=(100.0, 100.0))],
        model_list,
        "txt",
        MagicMock(),
        [[]],
    )

    assert model_list[0][0]["content"] == '<text style="strikethrough">deleted</text>'


def test_ocr_path_does_not_collect_or_apply_pdf_styles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 OCR 路径不会读取 drawing 或应用原生文本删除线。"""

    build_styles = MagicMock()
    monkeypatch.setattr(
        text_content,
        "build_pdf_native_visual_lines_and_styles",
        build_styles,
    )
    monkeypatch.setattr(
        text_content,
        "_build_page_text_formula_spans",
        lambda *_args: [
            _AnalyzeSpan(
                type=ContentType.TEXT,
                bbox=(10.0, 10.0, 52.0, 20.0),
                content="deleted",
                score=1.0,
            )
        ],
    )
    monkeypatch.setattr(
        text_content,
        "_group_page_spans_by_block",
        lambda *_args: {
            0: [
                _AnalyzeLine(
                    bbox=(10.0, 10.0, 52.0, 20.0),
                    spans=[
                        _AnalyzeSpan(
                            type=ContentType.TEXT,
                            bbox=(10.0, 10.0, 52.0, 20.0),
                            content="deleted",
                            score=1.0,
                        )
                    ],
                )
            ]
        },
    )
    model_list = [[{"type": BlockType.TEXT, "bbox": [0.1, 0.1, 0.52, 0.2], "content": ""}]]

    text_content._fill_window_block_content_and_lines(
        [{"img_pil": object(), "scale": 1.0}],
        [MagicMock(size=(100.0, 100.0))],
        model_list,
        [[]],
        [[]],
        "ocr",
        {BlockType.TEXT},
        MagicMock(),
    )

    build_styles.assert_not_called()
    assert model_list[0][0]["content"] == "deleted"


def test_native_span_fill_does_not_read_preloaded_page_chars_twice() -> None:
    """验证删除线检测已加载 chars 后，原生 span 回填不会再次读取当前页字符。"""

    page_chars = [
        {
            "char": char,
            "bbox": (10.0 + index * 6.0, 10.0, 16.0 + index * 6.0, 20.0),
            "char_idx": index,
            "rotation": 0.0,
            "font": {"name": "Helvetica", "flags": 0},
        }
        for index, char in enumerate("deleted")
    ]
    span = _AnalyzeSpan(
        type=ContentType.TEXT,
        bbox=(10.0, 10.0, 52.0, 20.0),
        score=1.0,
    )
    pdf_page = MagicMock()
    pdf_page.get_char_count.return_value = len(page_chars)

    result = txt_spans_extract(
        pdf_page,
        [span],
        object(),
        1.0,
        [(0.0, 0.0, 100.0, 100.0, None, None, None, BlockType.TEXT)],
        [],
        page_chars=page_chars,
    )

    assert result == [span]
    assert span.content == "deleted"
    pdf_page.get_chars.assert_not_called()
