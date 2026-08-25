from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from zipfile import ZipFile

import pytest

from mineru.backend.analysis.pdf.text import content as text_content
from mineru.backend.analysis.pdf.text.models import _AnalyzeLine, _AnalyzeSpan
from mineru.backend.analysis.pdf.text.native import txt_spans_extract
from mineru.backend.postprocess.pages import model_json_to_pages
from mineru.model.flash import PdfModel
from mineru.render import render_docx, render_html, render_markdown, render_structured_content
from mineru.types import (
    RAW_CAPTION,
    RAW_FOOTNOTE,
    BlockType,
    ContentType,
    MiddleJson,
    ModelJson,
    PageInfo,
)
from mineru.utils.pdf_document import PDFDocument, PDFLinkAnnotation
from mineru.utils.pdf_text_styles import (
    PDF_FONT_FORCE_BOLD_FLAG,
    PDF_FONT_ITALIC_FLAG,
    PDFTextLinkLine,
    PDFTextLinkRange,
    PDFTextStyleLine,
    PDFTextStyleRange,
    apply_pdf_text_links,
    apply_pdf_text_styles,
    detect_pdf_text_link_lines,
    detect_pdf_text_style_lines,
)


def _model_json(pages: list[list[dict[str, Any]]]) -> ModelJson:
    """为 PDF 文本样式渲染测试构造最小严格 ModelJson。"""
    return ModelJson(
        pages=pages,
        page_index_map=[],
        file_suffix="pdf",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )


def _build_native_text_style_pdf() -> bytes:
    """构造包含字体、删除线、下划线及其反例的单页原生 PDF。"""

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
BT /F2 12 Tf 50 50 Td (bold sample) Tj ET
BT /F3 12 Tf 50 30 Td (italic sample) Tj ET
BT /F4 12 Tf 50 10 Td (bold italic sample) Tj ET
"""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 420 300] "
        b"/Resources << /Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R /F4 7 0 R >> >> "
        b"/Contents 8 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-BoldOblique >>",
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
    font_name: str = "",
    font_flags: int = 0,
    font_weight: object = None,
    rotation_degrees: float = 0.0,
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
                "rotation": math.radians(rotation_degrees),
                "font": {
                    "name": font_name,
                    "flags": font_flags,
                    "weight": font_weight,
                },
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
    """构造文本装饰线检测使用的轻量 drawing。"""

    return SimpleNamespace(
        bbox=(x0, y - width / 2, x1, y + width / 2),
        width=width,
        orientation=orientation,
    )


def _char_span(line: SimpleNamespace, start: int, end: int) -> tuple[float, float]:
    """返回指定字符区间的左右边缘。"""

    return line.chars[start]["bbox"][0], line.chars[end - 1]["bbox"][2]


def _link_evidence_line(
    text: str,
    target: str,
    source_index: int,
    *,
    start: int = 0,
    end: int | None = None,
) -> PDFTextLinkLine:
    """构造跨行合并测试使用的轻量链接证据。"""

    resolved_end = len(text) if end is None else end
    top = 10.0 + (source_index % 10) * 10.0
    return PDFTextLinkLine(
        bbox=(10.0, top, 90.0, top + 8.0),
        text=text,
        link_ranges=(PDFTextLinkRange(start, resolved_end, target),),
        source_index=source_index,
    )


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_detect_and_apply_pdf_text_link_ranges_with_styles(angle: int) -> None:
    """验证标准方向链接映射、内部空格、URL 转义、样式顺序和幂等性。"""

    line = _text_line("alpha beta", angle=angle)
    annotation = PDFLinkAnnotation(
        target="https://example.test/a?x=1&y=2",
        bboxes=((line.bbox[0], line.bbox[1], line.bbox[2], line.bbox[3]),),
        source_index=0,
    )
    link_lines = detect_pdf_text_link_lines([line], [annotation])

    assert link_lines == [
        PDFTextLinkLine(
            bbox=line.bbox,
            text="alphabeta",
            link_ranges=(
                PDFTextLinkRange(
                    0,
                    9,
                    "https://example.test/a?x=1&y=2",
                ),
            ),
            source_index=0,
        )
    ]

    blocks = [
        {
            "type": BlockType.TEXT,
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "content": "alpha beta",
        }
    ]
    apply_pdf_text_links(blocks, link_lines, (100.0, 100.0))
    apply_pdf_text_styles(
        blocks,
        [
            PDFTextStyleLine(
                bbox=line.bbox,
                text="alphabeta",
                style_ranges=(PDFTextStyleRange(5, 9, ("underline",)),),
                source_index=0,
            )
        ],
        (100.0, 100.0),
    )
    expected = (
        "<hyperlink>alpha beta"
        "<url>https://example.test/a?x=1&amp;y=2</url></hyperlink>"
    )
    assert blocks[0]["content"] == expected

    apply_pdf_text_links(blocks, link_lines, (100.0, 100.0))
    apply_pdf_text_styles(
        blocks,
        [
            PDFTextStyleLine(
                bbox=line.bbox,
                text="alphabeta",
                style_ranges=(PDFTextStyleRange(5, 9, ("underline",)),),
                source_index=0,
            )
        ],
        (100.0, 100.0),
    )
    assert blocks[0]["content"] == expected


def test_pdf_underline_keeps_only_non_link_overlap() -> None:
    """验证同一下划线跨越普通文本和链接时，仅保留链接外的真实下划线。"""

    line = _text_line("under hyperlink")
    annotation = PDFLinkAnnotation(
        target="https://example.test/partial",
        bboxes=(
            (
                line.chars[6]["bbox"][0],
                line.bbox[1],
                line.chars[-1]["bbox"][2],
                line.bbox[3],
            ),
        ),
        source_index=0,
    )
    link_lines = detect_pdf_text_link_lines([line], [annotation])
    blocks = [
        {
            "type": BlockType.TEXT,
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "content": "under hyperlink",
        }
    ]
    style_lines = [
        PDFTextStyleLine(
            bbox=line.bbox,
            text="underhyperlink",
            style_ranges=(PDFTextStyleRange(0, 14, ("underline",)),),
            source_index=0,
        )
    ]

    apply_pdf_text_links(blocks, link_lines, (100.0, 100.0))
    apply_pdf_text_styles(blocks, style_lines, (100.0, 100.0))

    assert blocks[0]["content"] == (
        '<text style="underline">under</text> '
        "<hyperlink>hyperlink"
        "<url>https://example.test/partial</url></hyperlink>"
    )


def test_detect_pdf_text_links_keeps_partial_ranges_and_drops_conflicts() -> None:
    """验证局部字符链接可保留，而不同目标重叠字符按歧义丢弃。"""

    line = _text_line("alpha")
    partial_bbox = (
        line.chars[2]["bbox"][0],
        line.bbox[1],
        line.chars[3]["bbox"][2],
        line.bbox[3],
    )
    partial = PDFLinkAnnotation(
        target="https://partial.example.test",
        bboxes=(partial_bbox,),
        source_index=0,
    )
    partial_lines = detect_pdf_text_link_lines([line], [partial])
    blocks = [
        {
            "type": BlockType.TEXT,
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "content": "alpha",
        }
    ]
    apply_pdf_text_links(blocks, partial_lines, (100.0, 100.0))
    assert blocks[0]["content"] == (
        "al<hyperlink>ph<url>https://partial.example.test</url></hyperlink>a"
    )

    conflict = PDFLinkAnnotation(
        target="https://conflict.example.test",
        bboxes=(partial_bbox,),
        source_index=1,
    )
    assert detect_pdf_text_link_lines([line], [partial, conflict]) == []


def test_apply_pdf_text_links_splits_formula_gaps_and_maps_repeated_labels() -> None:
    """验证链接不跨公式包装，并按物理行顺序定位重复标签。"""

    formula_blocks = [
        {
            "type": BlockType.TEXT,
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "content": "pre<eq>x</eq>post",
        }
    ]
    apply_pdf_text_links(
        formula_blocks,
        [
            PDFTextLinkLine(
                bbox=(10.0, 10.0, 60.0, 20.0),
                text="prexpost",
                link_ranges=(
                    PDFTextLinkRange(0, 8, "https://formula.example.test"),
                ),
                source_index=0,
            )
        ],
        (100.0, 100.0),
    )
    assert formula_blocks[0]["content"] == (
        "<hyperlink>pre<url>https://formula.example.test</url></hyperlink>"
        "<eq>x</eq>"
        "<hyperlink>post<url>https://formula.example.test</url></hyperlink>"
    )

    repeated_blocks = [
        {
            "type": BlockType.TEXT,
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "content": "link and link",
        }
    ]
    apply_pdf_text_links(
        repeated_blocks,
        [
            PDFTextLinkLine(
                bbox=(10.0, 10.0, 35.0, 20.0),
                text="link",
                link_ranges=(PDFTextLinkRange(0, 4, "https://first.test"),),
                source_index=0,
            ),
            PDFTextLinkLine(
                bbox=(10.0, 30.0, 35.0, 40.0),
                text="link",
                link_ranges=(PDFTextLinkRange(0, 4, "https://second.test"),),
                source_index=1,
            ),
        ],
        (100.0, 100.0),
    )
    assert repeated_blocks[0]["content"] == (
        "<hyperlink>link<url>https://first.test</url></hyperlink> and "
        "<hyperlink>link<url>https://second.test</url></hyperlink>"
    )


def test_apply_pdf_text_links_merges_three_line_visible_url() -> None:
    """验证三行同 href URL 合成一个标签，并吸收边界点号且不插入空格。"""

    target = "https://github.com/google-research/tapas/blob/master/TABLEFORMER.md"
    blocks = [
        {
            "type": BlockType.PAGE_FOOTNOTE,
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "content": f"1Code has been released at {target}",
        }
    ]
    link_lines = [
        _link_evidence_line(
            "1Codehasbeenreleasedathttps://github.",
            target,
            51,
            start=22,
            end=36,
        ),
        _link_evidence_line(
            "com/google-research/tapas/blob/master/",
            target,
            52,
        ),
        _link_evidence_line("TABLEFORMER.md", target, 53),
    ]

    apply_pdf_text_links(blocks, link_lines, (100.0, 100.0))

    expected = (
        "1Code has been released at "
        f"<hyperlink>{target}<url>{target}</url></hyperlink>"
    )
    assert blocks[0]["content"] == expected
    apply_pdf_text_links(blocks, link_lines, (100.0, 100.0))
    assert blocks[0]["content"] == expected


@pytest.mark.parametrize(
    (
        "content",
        "first_text",
        "first_start",
        "second_text",
        "second_end",
        "label",
    ),
    [
        (
            "Müller. 2020. Understanding tables with intermediate pre-training. In Findings",
            "Müller.2020.Understandingtableswithinterme-",
            len("Müller.2020."),
            "diatepre-training.InFindings",
            len("diatepre-training"),
            "Understanding tables with intermediate pre-training",
        ),
        (
            "Search-based neural structured learning for sequential question answering. In Proceedings",
            "Search-basedneuralstructuredlearningforsequen-",
            0,
            "tialquestionanswering.InProceedings",
            len("tialquestionanswering"),
            "Search-based neural structured learning for sequential question answering",
        ),
        (
            "Yasemin Altun. 2019. Answering conversational questions",
            "YaseminAltun.2019.An-",
            len("YaseminAltun.2019."),
            "sweringconversationalquestions",
            len("sweringconversationalquestions"),
            "Answering conversational questions",
        ),
        (
            "Percy Liang. 2015. Compositional semantic parsing",
            "PercyLiang.2015.Compo-",
            len("PercyLiang.2015."),
            "sitionalsemanticparsing",
            len("sitionalsemanticparsing"),
            "Compositional semantic parsing",
        ),
    ],
)
def test_apply_pdf_text_links_maps_dehyphenated_first_fragment(
    content: str,
    first_text: str,
    first_start: int,
    second_text: str,
    second_end: int,
    label: str,
) -> None:
    """验证同 href 相邻行的首段断词符被回填删除后仍能完整映射。"""

    target = "https://doi.org/10.18653/v1/example"
    blocks = [
        {
            "type": BlockType.REF_TEXT,
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "content": content,
        }
    ]
    apply_pdf_text_links(
        blocks,
        [
            _link_evidence_line(
                first_text,
                target,
                80,
                start=first_start,
            ),
            _link_evidence_line(
                second_text,
                target,
                81,
                end=second_end,
            ),
        ],
        (100.0, 100.0),
    )

    expected_link = f"<hyperlink>{label}<url>{target}</url></hyperlink>"
    assert blocks[0]["content"] == content.replace(label, expected_link)


def test_apply_pdf_text_links_maps_dehyphenated_middle_fragment() -> None:
    """验证三行链接中间行的行末断词符不再阻断同 href 合并。"""

    target = "https://doi.org/10.18653/v1/N19-1423"
    label = (
        "BERT: Pre-training of deep bidirectional transformers for language "
        "understanding"
    )
    content = f"Kristina Toutanova. 2019. {label}. In Proceedings"
    first_text = "KristinaToutanova.2019.BERT:Pre-trainingof"
    blocks = [
        {
            "type": BlockType.REF_TEXT,
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "content": content,
        }
    ]
    link_lines = [
        _link_evidence_line(
            first_text,
            target,
            90,
            start=len("KristinaToutanova.2019."),
        ),
        _link_evidence_line(
            "deepbidirectionaltransformersforlanguageunder-",
            target,
            91,
        ),
        _link_evidence_line(
            "standing.InProceedings",
            target,
            92,
            end=len("standing"),
        ),
    ]

    apply_pdf_text_links(blocks, link_lines, (100.0, 100.0))

    expected = content.replace(
        label,
        f"<hyperlink>{label}<url>{target}</url></hyperlink>",
    )
    assert blocks[0]["content"] == expected
    apply_pdf_text_links(blocks, link_lines, (100.0, 100.0))
    assert blocks[0]["content"] == expected


@pytest.mark.parametrize(
    ("content", "first_line", "second_line"),
    [
        (
            "international",
            _link_evidence_line("inter-", "https://same.test", 100),
            _link_evidence_line("national", "https://same.test", 102),
        ),
        (
            "international",
            _link_evidence_line("inter-", "https://same.test", 100),
            _link_evidence_line("national", "https://different.test", 101),
        ),
        (
            "interxnational",
            _link_evidence_line("inter-", "https://same.test", 100),
            _link_evidence_line(
                "xnational",
                "https://same.test",
                101,
                start=1,
            ),
        ),
        (
            "interxnational",
            _link_evidence_line(
                "interx-",
                "https://same.test",
                100,
                end=len("inter"),
            ),
            _link_evidence_line("national", "https://same.test", 101),
        ),
    ],
)
def test_apply_pdf_text_links_rejects_unsafe_dehyphenated_boundaries(
    content: str,
    first_line: PDFTextLinkLine,
    second_line: PDFTextLinkLine,
) -> None:
    """验证不同 href、非相邻行或非行首行尾链接不会启用断词投影。"""

    blocks = [
        {
            "type": BlockType.TEXT,
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "content": content,
        }
    ]

    apply_pdf_text_links(
        blocks,
        [first_line, second_line],
        (100.0, 100.0),
    )

    assert f"<hyperlink>{content}<url>" not in blocks[0]["content"]


def test_apply_pdf_text_links_preserves_hyphen_before_uppercase_continuation() -> None:
    """验证下一行以大写字母开头时保留可见连字符并正常合并链接。"""

    target = "https://same.test"
    blocks = [
        {
            "type": BlockType.TEXT,
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "content": "inter-National",
        }
    ]

    apply_pdf_text_links(
        blocks,
        [
            _link_evidence_line("inter-", target, 110),
            _link_evidence_line("National", target, 111),
        ],
        (100.0, 100.0),
    )

    assert blocks[0]["content"] == (
        f"<hyperlink>inter-National<url>{target}</url></hyperlink>"
    )


def test_apply_pdf_text_links_skips_ambiguous_dehyphenated_occurrence() -> None:
    """验证断词候选在 block 中重复时不会把前后片段错误桥接。"""

    blocks = [
        {
            "type": BlockType.TEXT,
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "content": "international international",
        }
    ]

    apply_pdf_text_links(
        blocks,
        [
            _link_evidence_line("inter-", "https://same.test", 120),
            _link_evidence_line("national", "https://same.test", 121),
        ],
        (100.0, 100.0),
    )

    assert "<hyperlink>international<url>" not in blocks[0]["content"]
    assert blocks[0]["content"].startswith("inter<hyperlink>national")
    assert blocks[0]["content"].endswith(" international")


@pytest.mark.parametrize(
    ("second_styles", "expected_label"),
    [
        (
            ("bold",),
            '<text style="bold">ETC: Encoding long and structured inputs in transformers</text>',
        ),
        (
            ("underline",),
            (
                '<text style="bold">ETC: Encoding long and structured inputs</text> '
                "in transformers"
            ),
        ),
    ],
)
def test_apply_pdf_text_links_merges_title_and_preserves_non_underline_styles(
    second_styles: tuple[str, ...],
    expected_label: str,
) -> None:
    """验证普通英文标签保留空格，并仅保留链接内非下划线样式。"""

    target = "https://doi.org/10.18653/v1/2020.emnlp-main.19"
    first_text = "ETC:Encodinglongandstructuredinputs"
    second_text = "intransformers"
    blocks = [
        {
            "type": BlockType.TEXT,
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "content": "ETC: Encoding long and structured inputs in transformers",
        }
    ]
    link_lines = [
        _link_evidence_line(first_text, target, 60),
        _link_evidence_line(second_text, target, 61),
    ]
    style_lines = [
        PDFTextStyleLine(
            link_lines[0].bbox,
            first_text,
            (PDFTextStyleRange(0, len(first_text), ("bold",)),),
            60,
        ),
        PDFTextStyleLine(
            link_lines[1].bbox,
            second_text,
            (PDFTextStyleRange(0, len(second_text), second_styles),),
            61,
        ),
    ]

    apply_pdf_text_links(blocks, link_lines, (100.0, 100.0))
    apply_pdf_text_styles(blocks, style_lines, (100.0, 100.0))

    assert blocks[0]["content"] == (
        f"<hyperlink>{expected_label}<url>{target}</url></hyperlink>"
    )


@pytest.mark.parametrize(
    ("content", "first_source", "second_source", "second_target"),
    [
        ("first unlinked second", 70, 71, "https://same.test"),
        ("first<eq>x</eq>second", 70, 71, "https://same.test"),
        ("first second", 70, 72, "https://same.test"),
        ("first second", 70, 71, "https://different.test"),
    ],
)
def test_apply_pdf_text_links_does_not_cross_invalid_merge_boundaries(
    content: str,
    first_source: int,
    second_source: int,
    second_target: str,
) -> None:
    """验证无 href 正文、公式、非相邻行或不同目标都会阻断链接合并。"""

    first_target = "https://same.test"
    blocks = [
        {
            "type": BlockType.TEXT,
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "content": content,
        }
    ]
    apply_pdf_text_links(
        blocks,
        [
            _link_evidence_line("first", first_target, first_source),
            _link_evidence_line("second", second_target, second_source),
        ],
        (100.0, 100.0),
    )

    assert blocks[0]["content"].count("<hyperlink>") == 2


@pytest.mark.parametrize(
    ("block_type", "should_link"),
    [
        (BlockType.TEXT, True),
        (BlockType.REF_TEXT, True),
        (BlockType.DOC_TITLE, True),
        (BlockType.PARAGRAPH_TITLE, True),
        (BlockType.LIST, True),
        (BlockType.INDEX, True),
        (BlockType.HEADER, True),
        (BlockType.FOOTER, True),
        (BlockType.PAGE_NUMBER, True),
        (BlockType.ASIDE_TEXT, True),
        (BlockType.PAGE_FOOTNOTE, True),
        (RAW_CAPTION, True),
        (RAW_FOOTNOTE, True),
        (BlockType.TABLE, False),
        (BlockType.CODE, False),
        (BlockType.EQUATION, False),
        (BlockType.IMAGE, False),
    ],
)
def test_apply_pdf_text_links_only_enriches_natural_language_blocks(
    block_type: str,
    should_link: bool,
) -> None:
    """验证 PDF Link 只进入已批准的自然语言 block 类型。"""

    blocks = [
        {
            "type": block_type,
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "content": "label",
        }
    ]
    apply_pdf_text_links(
        blocks,
        [
            PDFTextLinkLine(
                bbox=(10.0, 10.0, 40.0, 20.0),
                text="label",
                link_ranges=(PDFTextLinkRange(0, 5, "https://target.test"),),
                source_index=0,
            )
        ],
        (100.0, 100.0),
    )

    expected = (
        "<hyperlink>label<url>https://target.test</url></hyperlink>"
        if should_link
        else "label"
    )
    assert blocks[0]["content"] == expected


@pytest.mark.parametrize(
    ("line_options", "expected_styles"),
    [
        ({"font_flags": PDF_FONT_FORCE_BOLD_FLAG}, ("bold",)),
        ({"font_weight": 599}, None),
        ({"font_weight": 600}, ("bold",)),
        ({"font_weight": -1}, None),
        ({"font_weight": "invalid"}, None),
        ({"font_name": "ABCDEF+FrankRuhlHofshi-Bold"}, ("bold",)),
        ({"font_name": "SourceSansPro-Semibold"}, ("bold",)),
        ({"font_name": "Arial-Black"}, ("bold",)),
        ({"font_name": "HiraginoSansGB-W6"}, ("bold",)),
        ({"font_name": "HiraginoSansGB-W3"}, None),
        (
            {
                "font_name": "LinuxLibertineGBI",
                "font_flags": PDF_FONT_ITALIC_FLAG,
            },
            ("bold",),
        ),
        ({"font_name": "MyriadPro-SemiCn"}, None),
        ({"font_flags": PDF_FONT_ITALIC_FLAG}, None),
        ({"font_name": "TimesNewRomanPS-ItalicMT"}, None),
        ({"font_name": "Helvetica-Oblique"}, None),
        ({"rotation_degrees": 4.9}, None),
        ({"rotation_degrees": 5.0}, None),
        ({"rotation_degrees": 19.0}, None),
        ({"rotation_degrees": 30.0}, None),
        ({"rotation_degrees": 30.1}, None),
    ],
)
def test_detects_font_styles_only_from_approved_direct_evidence(
    line_options: dict[str, object],
    expected_styles: tuple[str, ...] | None,
) -> None:
    """验证 PDF 只从批准的直接证据生成粗体，所有斜体证据均被忽略。"""

    line = _text_line("styled", **line_options)

    detected = detect_pdf_text_style_lines([line], [])

    if expected_styles is None:
        assert detected == []
    else:
        assert detected == [
            PDFTextStyleLine(
                line.bbox,
                "styled",
                (PDFTextStyleRange(0, 6, expected_styles),),
                0,
            )
        ]


def test_rotated_line_never_uses_local_rotation_as_italic_evidence() -> None:
    """验证整行非标准方向即使字符角差落入区间也不会标为斜体。"""

    line = _text_line(
        "watermark",
        angle=340,
        rotation_degrees=20.0,
    )

    assert detect_pdf_text_style_lines([line], []) == []


@pytest.mark.parametrize(
    ("text", "expected_text"),
    [
        ("A", None),
        ("中", None),
        ("AB", "AB"),
        ("ﬁ", "fi"),
    ],
)
def test_filters_pdf_bold_runs_shorter_than_two_comparable_characters(
    text: str,
    expected_text: str | None,
) -> None:
    """验证单字符粗体被过滤，双字符和展开为双字符的 ligature 保留。"""

    line = _text_line(text, font_name="Helvetica-Bold")

    detected = detect_pdf_text_style_lines([line], [])

    if expected_text is None:
        assert detected == []
    else:
        assert detected == [
            PDFTextStyleLine(
                line.bbox,
                expected_text,
                (PDFTextStyleRange(0, len(expected_text), ("bold",)),),
                0,
            )
        ]


def test_filters_isolated_leading_bold_list_marker_cluster() -> None:
    """验证达到最小长度的行首项目符号簇仍不会污染普通正文。"""

    line = _text_line("•• body")
    for char in line.chars[:2]:
        char["font"] = {
            "name": "Helvetica-Bold",
            "flags": 0,
            "weight": 400,
        }

    assert detect_pdf_text_style_lines([line], []) == []


def test_filters_bold_bullet_but_keeps_later_bold_list_text() -> None:
    """验证真实列表形态只移除行首粗体圆点，保留后续粗体正文。"""

    line = _text_line("• 无序列表项 1：bold 粗体文本")
    bold_indices = {0}
    bold_start = line.text.index("bold")
    bold_indices.update(range(bold_start, len(line.text)))
    for char_index in bold_indices:
        line.chars[char_index]["font"] = {
            "name": "Helvetica-Bold",
            "flags": 0,
            "weight": 400,
        }

    detected = detect_pdf_text_style_lines([line], [])
    bold_fragments = [
        detected[0].text[style_range.start : style_range.end]
        for style_range in detected[0].style_ranges
        if "bold" in style_range.styles
    ]

    assert bold_fragments == ["bold粗体文本"]


def test_keeps_list_marker_when_it_is_part_of_a_full_bold_run() -> None:
    """验证项目符号与后续正文同属完整粗体 run 时不会被单独删除。"""

    line = _text_line("• bold", font_name="Helvetica-Bold")

    detected = detect_pdf_text_style_lines([line], [])

    assert detected[0].style_ranges == (
        PDFTextStyleRange(0, 5, ("bold",)),
    )


def test_combines_bold_and_strikethrough_styles_per_character() -> None:
    """验证斜体字体保持普通文本，粗体与删除线仍按字符正确组合。"""

    line = _text_line("ABCDEF", height=5.0)
    for char_index in (1, 2):
        line.chars[char_index]["font"] = {
            "name": "Helvetica-Bold",
            "flags": 0,
            "weight": 400,
        }
    line.chars[3]["font"] = {
        "name": "Helvetica-Oblique",
        "flags": 0,
        "weight": 400,
    }
    for char_index in (4, 5):
        line.chars[char_index]["font"] = {
            "name": "Helvetica-BoldOblique",
            "flags": 0,
            "weight": 400,
        }
    x0, x1 = _char_span(line, 1, 6)

    detected = detect_pdf_text_style_lines(
        [line],
        [_drawing(x0, x1, 12.5)],
    )

    assert detected[0].style_ranges == (
        PDFTextStyleRange(1, 3, ("bold", "strikethrough")),
        PDFTextStyleRange(3, 4, ("strikethrough",)),
        PDFTextStyleRange(4, 6, ("bold", "strikethrough")),
    )


def test_detects_long_strikethrough_and_preserves_internal_spaces() -> None:
    """验证长删除线按字符范围生成单个紧凑样式区间。"""

    line = _text_line("alpha deleted text omega")
    start = line.text.index("deleted")
    end = start + len("deleted text")
    x0, x1 = _char_span(line, start, end)

    detected = detect_pdf_text_style_lines(
        [line],
        [_drawing(x0, x1, 15.0)],
    )

    assert len(detected) == 1
    assert detected[0].text == "alphadeletedtextomega"
    assert detected[0].text[
        detected[0].style_ranges[0].start : detected[0].style_ranges[0].end
    ] == "deletedtext"


@pytest.mark.parametrize("drawing_y", [18.0, 20.0, 22.0])
def test_detects_underline_inside_strict_bottom_band(drawing_y: float) -> None:
    """验证主体字符下边界上下 0.20h 内的长横线产生下划线。"""

    line = _text_line("alpha underlined omega")
    start = line.text.index("underlined")
    end = start + len("underlined")
    x0, x1 = _char_span(line, start, end)

    detected = detect_pdf_text_style_lines(
        [line],
        [_drawing(x0, x1, drawing_y)],
    )

    assert detected[0].style_ranges == (
        PDFTextStyleRange(5, 15, ("underline",)),
    )


@pytest.mark.parametrize("drawing_y", [17.99, 22.01])
def test_rejects_underline_outside_strict_bottom_band(
    drawing_y: float,
) -> None:
    """验证超过主体下边界 0.20h 的横线不产生下划线。"""

    line = _text_line("alpha underlined omega")
    start = line.text.index("underlined")
    end = start + len("underlined")
    x0, x1 = _char_span(line, start, end)

    assert detect_pdf_text_style_lines(
        [line],
        [_drawing(x0, x1, drawing_y)],
    ) == []


def test_combines_bold_underline_and_strikethrough_in_protocol_order() -> None:
    """验证同一字符范围的粗体、下划线和删除线按固定顺序合并。"""

    line = _text_line("styled", font_name="Helvetica-Bold")
    x0, x1 = _char_span(line, 0, len(line.text))

    detected = detect_pdf_text_style_lines(
        [line],
        [
            _drawing(x0, x1, 15.0),
            _drawing(x0, x1, 20.0),
        ],
    )

    assert detected[0].style_ranges == (
        PDFTextStyleRange(
            0,
            6,
            ("bold", "underline", "strikethrough"),
        ),
    )


def test_merges_repeated_underline_drawings() -> None:
    """验证双线或重复共线 drawing 只生成一个下划线语义区间。"""

    line = _text_line("underlined")
    x0, x1 = _char_span(line, 0, len(line.text))

    detected = detect_pdf_text_style_lines(
        [line],
        [
            _drawing(x0, x1, 19.0),
            _drawing(x0, x1, 20.0),
        ],
    )

    assert detected[0].style_ranges == (
        PDFTextStyleRange(0, 10, ("underline",)),
    )


def test_rejects_fraction_bar_with_tightly_contained_lower_run() -> None:
    """验证横线下方紧邻且被覆盖的分母 run 会阻止下划线误判。"""

    numerator = _text_line("numerator", source_index=0)
    denominator = _text_line(
        "den",
        x=28.0,
        y=21.4,
        height=8.0,
        source_index=1,
    )
    x0, x1 = _char_span(numerator, 0, len(numerator.text))

    assert detect_pdf_text_style_lines(
        [numerator, denominator],
        [_drawing(x0, x1, 20.0)],
    ) == []


def test_keeps_underline_when_following_text_is_outside_fraction_gap() -> None:
    """验证普通下一行即使水平重叠也不会触发分数线排除。"""

    first = _text_line("underlined", source_index=0)
    second = _text_line(
        "next",
        x=28.0,
        y=22.1,
        height=8.0,
        source_index=1,
    )
    x0, x1 = _char_span(first, 0, len(first.text))

    detected = detect_pdf_text_style_lines(
        [first, second],
        [_drawing(x0, x1, 20.0)],
    )

    ranges_by_source = {
        line.source_index: line.style_ranges
        for line in detected
    }
    assert ranges_by_source[0] == (
        PDFTextStyleRange(0, 10, ("underline",)),
    )
    assert ranges_by_source[1] == ()


def test_rejects_thick_short_and_column_rules() -> None:
    """验证粗线、短线和贯穿整栏的分隔线均不产生文本样式。"""

    line = _text_line("alpha deleted omega")
    start = line.text.index("deleted")
    end = start + len("deleted")
    x0, x1 = _char_span(line, start, end)
    short_line = _text_line("alpha xy omega", y=30.0, source_index=1)
    short_start = short_line.text.index("xy")
    short_x0, short_x1 = _char_span(short_line, short_start, short_start + 2)

    detected = detect_pdf_text_style_lines(
        [line, short_line],
        [
            _drawing(x0, x1, 15.0, width=2.1),
            _drawing(short_x0, short_x1, 35.0),
            _drawing(0.0, 200.0, 15.0),
        ],
    )

    assert all(not item.style_ranges for item in detected)


@pytest.mark.parametrize(
    ("drawing_length", "expected_styles"),
    [
        (17.99, ()),
        (18.0, (PDFTextStyleRange(0, 3, ("strikethrough",)),)),
    ],
)
def test_text_decoration_minimum_length_is_one_point_eight_heights(
    drawing_length: float,
    expected_styles: tuple[PDFTextStyleRange, ...],
) -> None:
    """验证文本装饰线长度达到 1.8 倍中位字高时才允许生成样式。"""

    line = _text_line("abc")
    detected = detect_pdf_text_style_lines(
        [line],
        [_drawing(line.bbox[0], line.bbox[0] + drawing_length, 15.0)],
    )

    if expected_styles:
        assert detected[0].style_ranges == expected_styles
    else:
        assert detected == []


def test_accepts_one_aligned_endpoint_and_merges_adjacent_drawings() -> None:
    """验证尾部延长 drawing 可由左端点确认，邻接命中区间会合并。"""

    line = _text_line("alpha deleted text omega")
    deleted_start = line.text.index("deleted")
    deleted_end = deleted_start + len("deleted")
    text_start = line.text.index("text")
    text_end = text_start + len("text")
    deleted_x0, deleted_x1 = _char_span(line, deleted_start, deleted_end)
    text_x0, text_x1 = _char_span(line, text_start, text_end)

    detected = detect_pdf_text_style_lines(
        [line],
        [
            _drawing(deleted_x0, deleted_x1 + 5.0, 15.0),
            _drawing(text_x0, text_x1, 15.0),
        ],
    )

    assert len(detected[0].style_ranges) == 1
    style_range = detected[0].style_ranges[0]
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

    detected = detect_pdf_text_style_lines(
        [line],
        [
            _drawing(deleted_x0, deleted_x1, 15.0),
            _drawing(removed_x0, removed_x1, 15.0),
        ],
    )

    assert [
        detected[0].text[style_range.start : style_range.end]
        for style_range in detected[0].style_ranges
    ] == ["deleted", "removed"]


def test_drawing_is_assigned_to_the_closest_overlapping_line() -> None:
    """验证一条 drawing 同时接近两行时只归属中线距离更小的文本行。"""

    first = _text_line("deleted", y=10.0, source_index=0)
    second = _text_line("deleted", y=11.0, source_index=1)
    x0, x1 = _char_span(first, 0, len(first.text))

    detected = detect_pdf_text_style_lines(
        [first, second],
        [_drawing(x0, x1, 15.1)],
    )

    ranges_by_source = {
        line.source_index: line.style_ranges
        for line in detected
    }
    assert ranges_by_source[0] == (
        PDFTextStyleRange(0, 7, ("strikethrough",)),
    )
    assert ranges_by_source[1] == ()


def test_rotated_text_is_not_a_style_candidate() -> None:
    """验证首版忽略真正旋转的视觉文本。"""

    line = _text_line("rotated text", angle=90)

    assert detect_pdf_text_style_lines(
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

    assert detect_pdf_text_style_lines(
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
            (PDFTextStyleRange(0, 7, ("strikethrough",)),),
            1,
        ),
    ]

    apply_pdf_text_styles(blocks, lines, (100.0, 100.0))

    assert blocks[0]["content"] == 'deleted <text style="strikethrough">deleted</text>'


def test_applies_style_across_dehyphenated_line_boundary() -> None:
    """验证行末断词符被正文回填删除后，两行粗体仍合并为完整区间。"""

    blocks = [
        {
            "type": BlockType.TEXT,
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "content": "Abstract—Stereo matching was designed for sequences.",
        }
    ]
    lines = [
        PDFTextStyleLine(
            (10.0, 10.0, 90.0, 20.0),
            "Abstract—Stereomatchingwasde-",
            (PDFTextStyleRange(0, len("Abstract—Stereomatchingwasde-"), ("bold",)),),
            10,
        ),
        PDFTextStyleLine(
            (10.0, 30.0, 90.0, 40.0),
            "signedforsequences.",
            (PDFTextStyleRange(0, len("signedforsequences."), ("bold",)),),
            11,
        ),
    ]

    apply_pdf_text_styles(blocks, lines, (100.0, 100.0))

    assert blocks[0]["content"] == (
        '<text style="bold">Abstract—Stereo matching was designed for sequences.</text>'
    )


@pytest.mark.parametrize(
    ("content", "first_text", "second_text", "second_source_index", "expected"),
    [
        (
            "inter-national",
            "inter-",
            "national",
            21,
            '<text style="bold">inter-national</text>',
        ),
        (
            "inter-National",
            "inter-",
            "National",
            21,
            '<text style="bold">inter-National</text>',
        ),
        (
            "international",
            "inter-",
            "national",
            22,
            'inter<text style="bold">national</text>',
        ),
        (
            "international international",
            "inter-",
            "national",
            21,
            'inter<text style="bold">national</text> international',
        ),
    ],
)
def test_style_dehyphenation_preserves_safe_boundaries(
    content: str,
    first_text: str,
    second_text: str,
    second_source_index: int,
    expected: str,
) -> None:
    """验证保留连字符、大小写、非相邻行和重复候选均不会错误桥接。"""

    blocks = [
        {
            "type": BlockType.TEXT,
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "content": content,
        }
    ]
    lines = [
        PDFTextStyleLine(
            (10.0, 10.0, 90.0, 20.0),
            first_text,
            (PDFTextStyleRange(0, len(first_text), ("bold",)),),
            20,
        ),
        PDFTextStyleLine(
            (10.0, 30.0, 90.0, 40.0),
            second_text,
            (PDFTextStyleRange(0, len(second_text), ("bold",)),),
            second_source_index,
        ),
    ]

    apply_pdf_text_styles(blocks, lines, (100.0, 100.0))

    assert blocks[0]["content"] == expected


def test_same_visual_row_style_runs_follow_source_order_despite_bbox_jitter() -> None:
    """验证同行 run 的细微顶边抖动不会把后方粗体片段提前映射。"""

    attention = _text_line(
        "Attention",
        x=10.0,
        y=10.01,
        source_index=0,
        font_name="Helvetica-Bold",
    )
    bias = _text_line(
        "Bias",
        x=80.0,
        y=10.01,
        source_index=1,
        font_name="Helvetica-Bold",
    )
    scaling = _text_line(
        "Scaling. Unlike",
        x=120.0,
        y=10.0,
        source_index=2,
    )
    for char in scaling.chars[: len("Scaling.")]:
        char["font"] = {
            "name": "Helvetica-Bold",
            "flags": 0,
            "weight": 400,
        }

    style_lines = detect_pdf_text_style_lines(
        [scaling, attention, bias],
        [],
    )
    assert [line.source_index for line in style_lines] == [0, 1, 2]

    blocks = [
        {
            "type": BlockType.TEXT,
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "content": "Attention Bias Scaling. Unlike",
        }
    ]
    apply_pdf_text_styles(blocks, style_lines, (250.0, 100.0))

    assert blocks[0]["content"] == (
        '<text style="bold">Attention Bias Scaling.</text> Unlike'
    )


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
            (PDFTextStyleRange(5, 12, ("strikethrough",)),),
            0,
        )
    ]

    apply_pdf_text_styles(blocks, lines, (100.0, 100.0))

    assert blocks[0]["content"] == (
        'value <sup><text style="strikethrough">deleted</text></sup> tail'
    )


def test_filters_italic_before_merging_overlapping_style_ranges() -> None:
    """验证过滤斜体后仍正确合并粗体、下划线与删除线区间。"""

    blocks = [
        {
            "type": "text",
            "bbox": [0.0, 0.0, 1.0, 0.4],
            "content": "abcdef",
        }
    ]
    lines = [
        PDFTextStyleLine(
            (10.0, 10.0, 90.0, 20.0),
            "abcdef",
            (
                PDFTextStyleRange(0, 4, ("bold",)),
                PDFTextStyleRange(2, 6, ("italic",)),
                PDFTextStyleRange(1, 5, ("underline",)),
                PDFTextStyleRange(3, 5, ("strikethrough",)),
            ),
            0,
        )
    ]

    apply_pdf_text_styles(blocks, lines, (100.0, 100.0))

    assert blocks[0]["content"] == (
        '<text style="bold">a</text>'
        '<text style="bold,underline">bc</text>'
        '<text style="bold,underline,strikethrough">d</text>'
        '<text style="underline,strikethrough">e</text>f'
    )


def test_preserves_internal_spaces_and_is_idempotent() -> None:
    """验证字体样式保留内部空格，重复富化不会增加嵌套标签。"""

    blocks = [
        {
            "type": "text",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "content": "bold text",
        }
    ]
    lines = [
        PDFTextStyleLine(
            (10.0, 10.0, 90.0, 20.0),
            "boldtext",
            (PDFTextStyleRange(0, 8, ("bold",)),),
            0,
        )
    ]

    apply_pdf_text_styles(blocks, lines, (100.0, 100.0))
    first_result = blocks[0]["content"]
    apply_pdf_text_styles(blocks, lines, (100.0, 100.0))

    assert first_result == '<text style="bold">bold text</text>'
    assert blocks[0]["content"] == first_result


def test_underline_does_not_wrap_boundary_spaces() -> None:
    """验证下划线只保留命中字符之间的内部空格，不扩散到边界空格。"""

    blocks = [
        {
            "type": BlockType.TEXT,
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "content": "  under lined  ",
        }
    ]
    lines = [
        PDFTextStyleLine(
            (10.0, 10.0, 90.0, 20.0),
            "underlined",
            (PDFTextStyleRange(0, 10, ("underline",)),),
            0,
        )
    ]

    apply_pdf_text_styles(blocks, lines, (100.0, 100.0))

    assert blocks[0]["content"] == (
        '  <text style="underline">under lined</text>  '
    )


@pytest.mark.parametrize(
    "block_type",
    [
        BlockType.TEXT,
        BlockType.REF_TEXT,
        BlockType.DOC_TITLE,
        BlockType.PARAGRAPH_TITLE,
        BlockType.LIST,
        BlockType.INDEX,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.ASIDE_TEXT,
        BlockType.PAGE_FOOTNOTE,
        RAW_CAPTION,
        RAW_FOOTNOTE,
    ],
)
def test_applies_style_specific_scope_to_natural_language_blocks(
    block_type: str,
) -> None:
    """验证粗体和下划线仅进入 text，删除线保持自然语言范围。"""

    blocks = [
        {
            "type": block_type,
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "content": "styled",
        }
    ]
    lines = [
        PDFTextStyleLine(
            (10.0, 10.0, 90.0, 20.0),
            "styled",
            (
                PDFTextStyleRange(
                    0,
                    6,
                    ("bold", "italic", "underline", "strikethrough"),
                ),
            ),
            0,
        )
    ]

    apply_pdf_text_styles(blocks, lines, (100.0, 100.0))

    expected_styles = (
        "bold,underline,strikethrough"
        if block_type == BlockType.TEXT
        else "strikethrough"
    )
    assert blocks[0]["content"] == (
        f'<text style="{expected_styles}">styled</text>'
    )


@pytest.mark.parametrize(
    "block_type",
    [
        BlockType.TABLE,
        BlockType.CODE,
        BlockType.EQUATION,
        BlockType.IMAGE,
    ],
)
def test_does_not_apply_styles_to_visual_or_code_blocks(block_type: str) -> None:
    """验证表格、代码、公式和图片 block 不进入 PDF 文本样式富化。"""

    blocks = [
        {
            "type": block_type,
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "content": "styled",
        }
    ]
    lines = [
        PDFTextStyleLine(
            (10.0, 10.0, 90.0, 20.0),
            "styled",
            (
                PDFTextStyleRange(
                    0,
                    6,
                    ("bold", "italic", "underline", "strikethrough"),
                ),
            ),
            0,
        )
    ]

    apply_pdf_text_styles(blocks, lines, (100.0, 100.0))

    assert blocks[0]["content"] == "styled"


def test_respects_existing_styles_and_skips_equation_and_url_payloads() -> None:
    """验证已有样式保持幂等，公式和 URL 不进入 PDF 样式富化。"""

    blocks = [
        {
            "type": "text",
            "bbox": [0.0, 0.0, 1.0, 0.4],
            "content": (
                "A<sup>B</sup><eq>C</eq>"
                "<hyperlink><text>D</text><url>u</url></hyperlink>"
            ),
        },
        {
            "type": "text",
            "bbox": [0.0, 0.6, 1.0, 1.0],
            "content": (
                "<b>A</b><i>B</i><u>C</u><s>D</s>"
                '<text style="underline">E</text>'
            ),
        },
    ]
    lines = [
        PDFTextStyleLine(
            (10.0, 10.0, 90.0, 20.0),
            "ABCD",
            (
                PDFTextStyleRange(0, 1, ("bold",)),
                PDFTextStyleRange(1, 2, ("italic",)),
                PDFTextStyleRange(2, 3, ("bold",)),
                PDFTextStyleRange(3, 4, ("bold", "italic", "underline")),
            ),
            0,
        ),
        PDFTextStyleLine(
            (10.0, 70.0, 90.0, 80.0),
            "ABCDE",
            (
                PDFTextStyleRange(0, 1, ("bold",)),
                PDFTextStyleRange(1, 2, ("italic",)),
                PDFTextStyleRange(2, 3, ("underline",)),
                PDFTextStyleRange(3, 4, ("strikethrough",)),
                PDFTextStyleRange(4, 5, ("underline",)),
            ),
            1,
        ),
    ]

    apply_pdf_text_styles(blocks, lines[:1], (100.0, 100.0))
    apply_pdf_text_styles(blocks[1:], lines[1:], (100.0, 100.0))
    first_results = [block["content"] for block in blocks]
    apply_pdf_text_styles(blocks, lines, (100.0, 100.0))

    assert '<text style="bold">A</text>' in str(blocks[0]["content"])
    assert "<sup>B</sup>" in str(blocks[0]["content"])
    assert "<eq>C</eq>" in str(blocks[0]["content"])
    assert '<text style="bold">D</text>' in str(blocks[0]["content"])
    assert 'style="bold,underline"' not in str(blocks[0]["content"])
    assert "<url>u</url>" in str(blocks[0]["content"])
    assert blocks[1]["content"] == (
        "<b>A</b><i>B</i><u>C</u><s>D</s>"
        '<text style="underline">E</text>'
    )
    assert [block["content"] for block in blocks] == first_results


def test_formula_style_does_not_fall_back_to_same_plain_character() -> None:
    """验证公式内字体样式不会误映射到同一 block 的同名普通字符。"""

    blocks = [
        {
            "type": "text",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "content": "A<eq>C</eq>C",
        }
    ]
    lines = [
        PDFTextStyleLine(
            (10.0, 10.0, 90.0, 20.0),
            "ACC",
            (PDFTextStyleRange(1, 2, ("bold",)),),
            0,
        )
    ]

    apply_pdf_text_styles(blocks, lines, (100.0, 100.0))

    assert blocks[0]["content"] == "A<eq>C</eq>C"


def test_unique_long_style_range_survives_omitted_list_marker() -> None:
    """验证 layout 省略行首项目符号时，较长唯一字体片段仍能安全对齐。"""

    blocks = [
        {
            "type": "text",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "content": "无序列表项1：bold 粗体文本",
        }
    ]
    lines = [
        PDFTextStyleLine(
            (10.0, 10.0, 90.0, 20.0),
            "•无序列表项1：bold粗体文本",
            (
                PDFTextStyleRange(0, 1, ("bold",)),
                PDFTextStyleRange(8, 16, ("bold",)),
            ),
            0,
        )
    ]

    apply_pdf_text_styles(blocks, lines, (100.0, 100.0))

    assert blocks[0]["content"] == (
        '无序列表项1：<text style="bold">bold 粗体文本</text>'
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
            (PDFTextStyleRange(6, 13, ("strikethrough",)),),
            0,
        ),
        PDFTextStyleLine(
            (10.0, 60.0, 90.0, 70.0),
            "deleted",
            (PDFTextStyleRange(0, 7, ("strikethrough",)),),
            1,
        ),
    ]

    apply_pdf_text_styles(blocks, lines, (100.0, 100.0))

    assert blocks[0]["content"] == "before <eq>deleted</eq> after"
    assert blocks[1]["content"] == "deleted"


def test_flash_native_pdf_styles_reach_model_middle_and_renderers() -> None:
    """验证真实 PDF 字体和 drawing 样式贯穿 model、MiddleJson 与 renderer。"""

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
    assert '<text style="underline">underlined</text>' in joined_model_content
    assert '<text style="strikethrough">underlined</text>' not in joined_model_content
    assert '<text style="strikethrough">x</text>' not in joined_model_content
    assert '<text style="strikethrough">separator' not in joined_model_content
    assert '<text style="bold">bold sample</text>' in joined_model_content
    assert "italic sample" in joined_model_content
    assert '<text style="italic">' not in joined_model_content
    assert '<text style="bold">bold italic sample</text>' in joined_model_content

    middle = MiddleJson(
        pages=model_json_to_pages(_model_json(model_list)),
        is_full_document=True,
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
    assert "<u>underlined</u>" in markdown
    assert "**bold sample**" in markdown
    assert "*italic sample*" not in markdown
    assert "**bold italic sample**" in markdown
    assert "<s>deleted</s>" in html
    assert "<u>underlined</u>" in html
    assert "<strong>bold sample</strong>" in html
    assert "<em>italic sample</em>" not in html
    assert "<strong>bold italic sample</strong>" in html
    assert "<w:strike" in document_xml
    assert "<w:u" in document_xml
    assert "<w:b" in document_xml
    assert "<w:i/>" not in document_xml
    assert '<w:i w:val="1"' not in document_xml


def test_demo1_pdf_link_reaches_model_middle_and_all_renderers() -> None:
    """验证真实 demo1 URI Link 贯穿 model、MiddleJson 和四类 renderer。"""

    pdf_path = Path(__file__).parents[2] / "demo/pdfs/demo1.pdf"
    with PDFDocument(str(pdf_path)) as document:
        model_list = PdfModel().predict(document)

    target = "http://www.elsevier.com/locate/jhydrol"
    label = "www.elsevier.com/locate/jhydrol"
    inline_markup = f"<hyperlink>{label}<url>{target}</url></hyperlink>"
    assert any(
        block.get("content") == inline_markup
        for block in model_list[0]
    )

    middle = MiddleJson(
        pages=model_json_to_pages(_model_json(model_list)),
        is_full_document=True,
        file_suffix="pdf",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )
    link_block = next(
        block
        for block in middle.pages[0].blocks
        if getattr(block, "content", None) == inline_markup
    )
    link_middle = MiddleJson(
        pages=[PageInfo(page_idx=0, blocks=[link_block])],
        is_full_document=True,
        file_suffix="pdf",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )
    assert f"[{label}]({target})" in render_markdown(link_middle)
    assert f'href="{target}"' in render_html(link_middle, standalone=False)
    relationships = ZipFile(BytesIO(render_docx(link_middle))).read(
        "word/_rels/document.xml.rels"
    ).decode("utf-8")
    assert target in relationships
    assert f"[{label}]({target})" in str(render_structured_content(link_middle))


def test_hybrid_txt_reuses_loaded_chars_and_applies_styles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Medium/High/XHigh 复用 chars，并应用允许的组合样式。"""

    page_chars = [{"char": "d", "bbox": (10.0, 10.0, 16.0, 20.0), "char_idx": 0}]
    style_lines = [
        PDFTextStyleLine(
            (10.0, 10.0, 52.0, 20.0),
            "deleted",
            (
                PDFTextStyleRange(
                    0,
                    7,
                    ("bold", "italic", "underline", "strikethrough"),
                ),
            ),
            0,
        )
    ]
    link_lines = [
        PDFTextLinkLine(
            (10.0, 10.0, 52.0, 20.0),
            "deleted",
            (PDFTextLinkRange(0, 7, "https://hybrid.example.test"),),
            0,
        )
    ]
    build_styles = MagicMock(
        return_value=(page_chars, [], style_lines, link_lines)
    )
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
    assert model_list[0][0]["content"] == (
        '<hyperlink><text style="bold,strikethrough">deleted</text>'
        "<url>https://hybrid.example.test</url></hyperlink>"
    )


@pytest.mark.parametrize("effort", ["medium", "high", "xhigh"])
def test_hybrid_txt_efforts_apply_dehyphenated_links(
    monkeypatch: pytest.MonkeyPatch,
    effort: str,
) -> None:
    """验证 Medium、High 与 XHigh 共享的 TXT 回填路径应用断词链接投影。"""

    target = f"https://{effort}.example.test"
    link_lines = [
        PDFTextLinkLine(
            (10.0, 10.0, 52.0, 20.0),
            "inter-",
            (PDFTextLinkRange(0, 6, target),),
            0,
        ),
        PDFTextLinkLine(
            (10.0, 30.0, 52.0, 40.0),
            "national",
            (PDFTextLinkRange(0, 8, target),),
            1,
        ),
    ]
    monkeypatch.setattr(
        text_content,
        "build_pdf_native_visual_lines_and_styles",
        lambda *_args, **_kwargs: ([], [], [], link_lines),
    )
    monkeypatch.setattr(
        text_content,
        "_build_page_text_formula_spans",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        text_content,
        "_fill_native_pdf_text_spans",
        lambda _page, spans, *_args, **_kwargs: spans,
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
                            content="inter-",
                            score=1.0,
                        )
                    ],
                ),
                _AnalyzeLine(
                    bbox=(10.0, 30.0, 52.0, 40.0),
                    spans=[
                        _AnalyzeSpan(
                            type=ContentType.TEXT,
                            bbox=(10.0, 30.0, 52.0, 40.0),
                            content="national",
                            score=1.0,
                        )
                    ],
                ),
            ]
        },
    )
    monkeypatch.setattr(text_content, "_apply_window_post_ocr", lambda *_args: None)
    pdf_page = MagicMock(size=(100.0, 100.0))
    pdf_page.get_char_count.return_value = 1
    model_list = [
        [
            {
                "type": BlockType.TEXT,
                "bbox": [0.1, 0.1, 0.52, 0.4],
                "content": "",
            }
        ]
    ]

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

    assert model_list[0][0]["content"] == (
        f"<hyperlink>international<url>{target}</url></hyperlink>"
    )


def test_ocr_path_does_not_collect_or_apply_pdf_styles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 OCR 路径不会读取或应用 PDF 原生文本样式与超链接。"""

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


def test_high_char_count_txt_path_skips_pdf_text_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证超高字符页走 post-OCR 兜底时不读取样式或 Link 注解。"""

    build_enrichment = MagicMock()
    monkeypatch.setattr(
        text_content,
        "build_pdf_native_visual_lines_and_styles",
        build_enrichment,
    )
    plain_span = _AnalyzeSpan(
        type=ContentType.TEXT,
        bbox=(10.0, 10.0, 52.0, 20.0),
        content="plain",
        score=1.0,
    )
    monkeypatch.setattr(
        text_content,
        "_build_page_text_formula_spans",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        text_content,
        "_fill_native_pdf_text_spans",
        lambda *_args, **_kwargs: [plain_span],
    )
    monkeypatch.setattr(
        text_content,
        "_group_page_spans_by_block",
        lambda *_args: {
            0: [
                _AnalyzeLine(
                    bbox=plain_span.bbox,
                    spans=[plain_span],
                )
            ]
        },
    )
    pdf_page = MagicMock(size=(100.0, 100.0))
    pdf_page.get_char_count.return_value = (
        text_content.MAX_NATIVE_TEXT_CHARS_PER_PAGE + 1
    )
    model_list = [
        [
            {
                "type": BlockType.TEXT,
                "bbox": [0.1, 0.1, 0.52, 0.2],
                "content": "",
            }
        ]
    ]

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

    build_enrichment.assert_not_called()
    assert model_list[0][0]["content"] == "plain"


def test_native_span_fill_does_not_read_preloaded_page_chars_twice() -> None:
    """验证样式检测已加载 chars 后，原生 span 回填不会再次读取当前页字符。"""

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
