# Copyright (c) Opendatalab. All rights reserved.
"""DOCX renderer 的页面几何与 Word 样式定义。"""

from __future__ import annotations

from docx.document import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor
from docx.styles.style import _ParagraphStyle

BODY_STYLE = "Normal"
CODE_STYLE = "MinerU Code"
CAPTION_STYLE = "MinerU Caption"
FOOTNOTE_STYLE = "MinerU Footnote"
AUXILIARY_STYLE = "MinerU Auxiliary"
FORMULA_FALLBACK_STYLE = "MinerU Formula Fallback"

_HEADING_SIZES = (20, 18, 16, 14, 13, 12, 11, 10.5, 10.5)


def configure_document(document: Document) -> None:
    """设置 A4 页面几何，并创建 renderer 使用的全部显式样式。"""
    section = document.sections[0]
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    section.top_margin = Mm(20)
    section.bottom_margin = Mm(20)
    section.left_margin = Mm(20)
    section.right_margin = Mm(20)

    normal = document.styles[BODY_STYLE]
    _configure_paragraph_style(
        normal,
        western_font="Times New Roman",
        east_asia_font="宋体",
        size_pt=10.5,
        space_before_pt=0,
        space_after_pt=6,
        line_spacing=1.15,
    )
    for level, size_pt in enumerate(_HEADING_SIZES, start=1):
        heading = document.styles[f"Heading {level}"]
        _configure_paragraph_style(
            heading,
            western_font="Arial",
            east_asia_font="黑体",
            size_pt=size_pt,
            bold=True,
            space_before_pt=10 if level <= 3 else 8,
            space_after_pt=5 if level <= 3 else 4,
            line_spacing=1.0,
        )
        heading.paragraph_format.keep_with_next = True

    code = _get_or_add_paragraph_style(document, CODE_STYLE)
    _configure_paragraph_style(
        code,
        western_font="Courier New",
        east_asia_font="宋体",
        size_pt=9,
        space_before_pt=4,
        space_after_pt=6,
        line_spacing=1.0,
    )
    _set_paragraph_shading(code, "F3F4F6")

    caption = _get_or_add_paragraph_style(document, CAPTION_STYLE)
    _configure_paragraph_style(
        caption,
        western_font="Times New Roman",
        east_asia_font="宋体",
        size_pt=9,
        space_before_pt=3,
        space_after_pt=5,
        line_spacing=1.0,
    )
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.keep_with_next = True

    footnote = _get_or_add_paragraph_style(document, FOOTNOTE_STYLE)
    _configure_paragraph_style(
        footnote,
        western_font="Times New Roman",
        east_asia_font="宋体",
        size_pt=8,
        color="666666",
        space_before_pt=2,
        space_after_pt=4,
        line_spacing=1.0,
    )

    auxiliary = _get_or_add_paragraph_style(document, AUXILIARY_STYLE)
    _configure_paragraph_style(
        auxiliary,
        western_font="Times New Roman",
        east_asia_font="宋体",
        size_pt=8,
        italic=True,
        color="777777",
        space_before_pt=2,
        space_after_pt=3,
        line_spacing=1.0,
    )

    formula_fallback = _get_or_add_paragraph_style(document, FORMULA_FALLBACK_STYLE)
    _configure_paragraph_style(
        formula_fallback,
        western_font="Courier New",
        east_asia_font="宋体",
        size_pt=9,
        color="555555",
        space_before_pt=4,
        space_after_pt=5,
        line_spacing=1.0,
    )
    formula_fallback.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER


def usable_width_twips(document: Document) -> int:
    """返回首个 section 扣除左右页边距后的可用宽度，单位 twip。"""
    section = document.sections[0]
    usable_emu = int(section.page_width) - int(section.left_margin) - int(section.right_margin)
    return max(1, round(usable_emu / 635))


def usable_width_emu(document: Document) -> int:
    """返回首个 section 扣除左右页边距后的可用宽度，单位 EMU。"""
    section = document.sections[0]
    return max(1, int(section.page_width) - int(section.left_margin) - int(section.right_margin))


def _get_or_add_paragraph_style(document: Document, name: str) -> _ParagraphStyle:
    """获取既有段落样式，缺失时创建同名样式。"""
    styles = document.styles
    try:
        style = styles[name]
    except KeyError:
        style = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    if not isinstance(style, _ParagraphStyle):
        raise TypeError(f"Style is not a paragraph style: {name}")
    return style


def _configure_paragraph_style(
    style: _ParagraphStyle,
    *,
    western_font: str,
    east_asia_font: str,
    size_pt: float,
    bold: bool = False,
    italic: bool = False,
    color: str | None = None,
    space_before_pt: float,
    space_after_pt: float,
    line_spacing: float,
) -> None:
    """给一个 Word 段落样式写入确定性的字体与段落节奏。"""
    style.font.name = western_font
    style.font.size = Pt(size_pt)
    style.font.bold = bold
    style.font.italic = italic
    if color is not None:
        style.font.color.rgb = RGBColor.from_string(color)
    run_properties = style._element.get_or_add_rPr()
    run_fonts = run_properties.get_or_add_rFonts()
    run_fonts.set(qn("w:ascii"), western_font)
    run_fonts.set(qn("w:hAnsi"), western_font)
    run_fonts.set(qn("w:eastAsia"), east_asia_font)
    paragraph_format = style.paragraph_format
    paragraph_format.space_before = Pt(space_before_pt)
    paragraph_format.space_after = Pt(space_after_pt)
    paragraph_format.line_spacing = line_spacing


def _set_paragraph_shading(style: _ParagraphStyle, fill: str) -> None:
    """给段落样式设置稳定的背景色。"""
    paragraph_properties = style._element.get_or_add_pPr()
    shading = paragraph_properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        paragraph_properties.append(shading)
    shading.set(qn("w:fill"), fill)


__all__ = [
    "AUXILIARY_STYLE",
    "BODY_STYLE",
    "CAPTION_STYLE",
    "CODE_STYLE",
    "FOOTNOTE_STYLE",
    "FORMULA_FALLBACK_STYLE",
    "configure_document",
    "usable_width_emu",
    "usable_width_twips",
]
