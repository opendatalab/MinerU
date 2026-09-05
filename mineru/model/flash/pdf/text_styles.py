# Copyright (c) Opendatalab. All rights reserved.
"""保留既有入口的 Flash PDF 门面，内部实现按职责显式组织。"""

from .inline.detection import (
    detect_pdf_text_link_lines as detect_pdf_text_link_lines,
    detect_pdf_text_style_lines as detect_pdf_text_style_lines,
)
from .inline.matching import (
    _assign_lines_to_blocks as _assign_lines_to_blocks,
    _realign_repaired_text_evidence as _realign_repaired_text_evidence,
    _partition_resplit_text_evidence as _partition_resplit_text_evidence,
)
from .inline.materialize import (
    apply_pdf_text_links as apply_pdf_text_links,
    apply_pdf_text_scripts as apply_pdf_text_scripts,
    apply_pdf_text_styles as apply_pdf_text_styles,
    materialize_pdf_inline_spans as materialize_pdf_inline_spans,
)
from .inline.scripts import (
    detect_pdf_text_script_lines as detect_pdf_text_script_lines,
    _fraction_member_indices as _fraction_member_indices,
    _refine_math_script_tokens as _refine_math_script_tokens,
    _script_line_char_roles as _script_line_char_roles,
)
from .inline.types import (
    PDF_NATIVE_SCRIPT_MARKUP_KEY as PDF_NATIVE_SCRIPT_MARKUP_KEY,
    PDF_FONT_FORCE_BOLD_FLAG as PDF_FONT_FORCE_BOLD_FLAG,
    PDF_FONT_ITALIC_FLAG as PDF_FONT_ITALIC_FLAG,
    PDFTextLinkLine as PDFTextLinkLine,
    PDFTextLinkRange as PDFTextLinkRange,
    PDFTextScriptLine as PDFTextScriptLine,
    PDFTextScriptRange as PDFTextScriptRange,
    PDFTextStyle as PDFTextStyle,
    PDFTextStyleLine as PDFTextStyleLine,
    PDFTextStyleRange as PDFTextStyleRange,
)

__all__ = [
    "PDF_NATIVE_SCRIPT_MARKUP_KEY",
    "PDF_FONT_FORCE_BOLD_FLAG",
    "PDF_FONT_ITALIC_FLAG",
    "PDFTextLinkLine",
    "PDFTextLinkRange",
    "PDFTextScriptLine",
    "PDFTextScriptRange",
    "PDFTextStyle",
    "PDFTextStyleLine",
    "PDFTextStyleRange",
    "apply_pdf_text_links",
    "apply_pdf_text_scripts",
    "apply_pdf_text_styles",
    "detect_pdf_text_link_lines",
    "detect_pdf_text_script_lines",
    "detect_pdf_text_style_lines",
    "materialize_pdf_inline_spans",
]
