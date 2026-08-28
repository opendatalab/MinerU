# Copyright (c) Opendatalab. All rights reserved.
"""Flash EPUB 与 HTML 共用的静态标记文档投影能力。"""

from .anchors import (
    AnchorTextNormalization,
    AnchorVisibilityScope,
    MarkupAnchorDocument,
    MarkupAnchorPolicy,
    MarkupAnchorRegistry,
    canonical_anchor,
    element_id,
    visible_element_text,
)
from .formula import FormulaDisplay, FormulaExtraction, FormulaSourceKind, extract_formula, strip_formula_delimiters
from .projector import MarkupContext, MarkupProjector, ResolvedMarkupImage
from .styles import ElementStyle, MarkupStylesheet, TextStyle, TextStyleDelta

__all__ = [
    "AnchorTextNormalization",
    "AnchorVisibilityScope",
    "ElementStyle",
    "FormulaDisplay",
    "FormulaExtraction",
    "FormulaSourceKind",
    "MarkupAnchorDocument",
    "MarkupAnchorPolicy",
    "MarkupAnchorRegistry",
    "MarkupContext",
    "MarkupProjector",
    "MarkupStylesheet",
    "ResolvedMarkupImage",
    "TextStyle",
    "TextStyleDelta",
    "canonical_anchor",
    "element_id",
    "extract_formula",
    "strip_formula_delimiters",
    "visible_element_text",
]
