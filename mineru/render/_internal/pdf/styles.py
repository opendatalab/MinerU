# Copyright (c) Opendatalab. All rights reserved.
"""PDF renderer 的页面几何、字体与 MinerU 打印样式。"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import cidfonts, pdfmetrics

PAGE_MARGIN = 20 * mm
BODY_FONT = "Helvetica"
BODY_BOLD_FONT = "Helvetica-Bold"
MONO_FONT = "Courier"
HAN_FONT = "STSong-Light"
JAPANESE_FONT = "HeiseiMin-W3"
KOREAN_FONT = "HYSMyeongJo-Medium"

ACCENT_COLOR = colors.HexColor("#0b6fc2")
BACKGROUND_COLOR = colors.HexColor("#ffffff")
BORDER_COLOR = colors.HexColor("#d1d5db")
MUTED_COLOR = colors.HexColor("#6b7280")
SURFACE_COLOR = colors.HexColor("#f8fafc")
TEXT_COLOR = colors.HexColor("#1f2937")

_FONT_LOCK = RLock()
_HEADING_SIZES = (20.0, 16.0, 14.0, 12.0, 11.0, 10.0)


@dataclass(frozen=True, slots=True)
class PdfStyleSet:
    """保存 PDF renderer 使用的全部稳定段落样式。"""

    body: ParagraphStyle
    headings: tuple[ParagraphStyle, ...]
    caption: ParagraphStyle
    footnote: ParagraphStyle
    code: ParagraphStyle
    spatial_table: ParagraphStyle
    formula_fallback: ParagraphStyle
    placeholder: ParagraphStyle
    table_cell: ParagraphStyle
    table_header: ParagraphStyle

    def heading(self, level: int) -> ParagraphStyle:
        """按公开一到六级标题返回对应的 PDF 样式。"""
        return self.headings[min(max(level, 1), len(self.headings)) - 1]


def build_pdf_styles() -> PdfStyleSet:
    """注册标准 CID 字体并构造无外部字体依赖的打印样式集。"""
    _register_cid_fonts()
    body = ParagraphStyle(
        "MinerU PDF Body",
        fontName=BODY_FONT,
        fontSize=10.5,
        leading=16,
        textColor=TEXT_COLOR,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=7,
        splitLongWords=True,
        allowWidows=1,
        allowOrphans=1,
    )
    headings = tuple(
        ParagraphStyle(
            f"MinerU PDF Heading {level}",
            parent=body,
            fontName=BODY_BOLD_FONT,
            fontSize=size,
            leading=size * 1.3,
            spaceBefore=10 if level <= 3 else 8,
            spaceAfter=6 if level <= 3 else 4,
            keepWithNext=True,
        )
        for level, size in enumerate(_HEADING_SIZES, start=1)
    )
    caption = ParagraphStyle(
        "MinerU PDF Caption",
        parent=body,
        fontSize=9,
        leading=12,
        textColor=MUTED_COLOR,
        spaceBefore=3,
        spaceAfter=5,
    )
    footnote = ParagraphStyle(
        "MinerU PDF Footnote",
        parent=body,
        fontSize=8.5,
        leading=11,
        textColor=MUTED_COLOR,
        spaceBefore=2,
        spaceAfter=4,
    )
    code = ParagraphStyle(
        "MinerU PDF Code",
        parent=body,
        fontName=MONO_FONT,
        fontSize=8.5,
        leading=11,
        backColor=SURFACE_COLOR,
        borderColor=BORDER_COLOR,
        borderWidth=0.5,
        borderPadding=7,
        borderRadius=3,
        spaceBefore=5,
        spaceAfter=7,
    )
    spatial_table = ParagraphStyle(
        "MinerU PDF Spatial Table",
        parent=code,
        backColor=BACKGROUND_COLOR,
        borderPadding=5,
    )
    formula_fallback = ParagraphStyle(
        "MinerU PDF Formula Fallback",
        parent=body,
        fontName=MONO_FONT,
        fontSize=9,
        leading=13,
        textColor=MUTED_COLOR,
        alignment=TA_CENTER,
        spaceBefore=5,
        spaceAfter=6,
    )
    placeholder = ParagraphStyle(
        "MinerU PDF Placeholder",
        parent=body,
        fontSize=9,
        leading=12,
        textColor=MUTED_COLOR,
        spaceBefore=0,
        spaceAfter=0,
    )
    table_cell = ParagraphStyle(
        "MinerU PDF Table Cell",
        parent=body,
        fontSize=8.5,
        leading=11,
        spaceBefore=0,
        spaceAfter=0,
    )
    table_header = ParagraphStyle(
        "MinerU PDF Table Header",
        parent=table_cell,
        fontName=BODY_BOLD_FONT,
    )
    return PdfStyleSet(
        body=body,
        headings=headings,
        caption=caption,
        footnote=footnote,
        code=code,
        spatial_table=spatial_table,
        formula_fallback=formula_fallback,
        placeholder=placeholder,
        table_cell=table_cell,
        table_header=table_header,
    )


def _register_cid_fonts() -> None:
    """在进程级 ReportLab registry 中幂等注册中日韩标准 CID 字体。"""
    with _FONT_LOCK:
        registered = set(pdfmetrics.getRegisteredFontNames())
        for font_name in (HAN_FONT, JAPANESE_FONT, KOREAN_FONT):
            if font_name not in registered:
                pdfmetrics.registerFont(cidfonts.UnicodeCIDFont(font_name))


__all__ = [
    "ACCENT_COLOR",
    "BACKGROUND_COLOR",
    "BODY_BOLD_FONT",
    "BODY_FONT",
    "BORDER_COLOR",
    "HAN_FONT",
    "JAPANESE_FONT",
    "KOREAN_FONT",
    "MONO_FONT",
    "MUTED_COLOR",
    "PAGE_MARGIN",
    "PdfStyleSet",
    "SURFACE_COLOR",
    "TEXT_COLOR",
    "build_pdf_styles",
]
