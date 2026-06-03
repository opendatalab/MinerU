# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Literal


class BlockType:
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    IMAGE_BODY = "image_body"
    TABLE_BODY = "table_body"
    CHART_BODY = "chart_body"
    CAPTION = "caption"  # generic caption type (e.g., for Word documents)
    IMAGE_CAPTION = "image_caption"
    TABLE_CAPTION = "table_caption"
    CHART_CAPTION = "chart_caption"
    ALGORITHM_CAPTION = "algorithm_caption"
    FOOTNOTE = "footnote"  # pp_layout中的vision_footnote
    IMAGE_FOOTNOTE = "image_footnote"
    TABLE_FOOTNOTE = "table_footnote"
    CHART_FOOTNOTE = "chart_footnote"
    TEXT = "text"
    TITLE = "title"
    INTERLINE_EQUATION = "interline_equation"
    EQUATION = "equation"  # 公式(独立公式)
    LIST = "list"
    INDEX = "index"
    DISCARDED = "discarded"

    # Added in vlm 2.5
    CODE = "code"
    CODE_BODY = "code_body"
    CODE_CAPTION = "code_caption"
    CODE_FOOTNOTE = "code_footnote"
    ALGORITHM = "algorithm"
    REF_TEXT = "ref_text"
    PHONETIC = "phonetic"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    ASIDE_TEXT = "aside_text"
    PAGE_FOOTNOTE = "page_footnote"

    # Added in pp_doclayout_v2
    ABSTRACT = "abstract"
    DOC_TITLE = "doc_title"
    PARAGRAPH_TITLE = "paragraph_title"
    VERTICAL_TEXT = "vertical_text"
    HEADER_IMAGE = "header_image"
    FOOTER_IMAGE = "footer_image"
    FORMULA_NUMBER = "formula_number"


class ContentType:
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    TEXT = "text"
    INTERLINE_EQUATION = "interline_equation"
    INLINE_EQUATION = "inline_equation"
    EQUATION = "equation"
    HYPERLINK = "hyperlink"


class ContentTypeV2:
    CODE = "code"
    ALGORITHM = "algorithm"
    EQUATION_INTERLINE = "equation_interline"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    TABLE_SIMPLE = "simple_table"
    TABLE_COMPLEX = "complex_table"
    LIST = "list"
    LIST_TEXT = "text_list"
    LIST_REF = "reference_list"
    INDEX = "index"
    TITLE = "title"
    PARAGRAPH = "paragraph"
    SPAN_TEXT = "text"
    SPAN_EQUATION_INLINE = "equation_inline"
    SPAN_PHONETIC = "phonetic"
    SPAN_MD = "md"
    SPAN_CODE_INLINE = "code_inline"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    PAGE_NUMBER = "page_number"
    PAGE_ASIDE_TEXT = "page_aside_text"
    PAGE_FOOTNOTE = "page_footnote"


BlockTypes = Literal[
    BlockType.IMAGE,
    BlockType.TABLE,
    BlockType.CHART,
    BlockType.IMAGE_BODY,
    BlockType.TABLE_BODY,
    BlockType.CHART_BODY,
    BlockType.CAPTION,
    BlockType.IMAGE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.CHART_CAPTION,
    BlockType.ALGORITHM_CAPTION,
    BlockType.FOOTNOTE,
    BlockType.IMAGE_FOOTNOTE,
    BlockType.TABLE_FOOTNOTE,
    BlockType.CHART_FOOTNOTE,
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.INTERLINE_EQUATION,
    BlockType.EQUATION,
    BlockType.LIST,
    BlockType.INDEX,
    BlockType.DISCARDED,
    BlockType.CODE,
    BlockType.CODE_BODY,
    BlockType.CODE_CAPTION,
    BlockType.CODE_FOOTNOTE,
    BlockType.ALGORITHM,
    BlockType.REF_TEXT,
    BlockType.PHONETIC,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
    BlockType.PAGE_FOOTNOTE,
    BlockType.ABSTRACT,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
    BlockType.VERTICAL_TEXT,
    BlockType.HEADER_IMAGE,
    BlockType.FOOTER_IMAGE,
    BlockType.FORMULA_NUMBER,
]

BLOCK_TYPES = {
    BlockType.IMAGE,
    BlockType.TABLE,
    BlockType.CHART,
    BlockType.IMAGE_BODY,
    BlockType.TABLE_BODY,
    BlockType.CHART_BODY,
    BlockType.CAPTION,
    BlockType.IMAGE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.CHART_CAPTION,
    BlockType.ALGORITHM_CAPTION,
    BlockType.FOOTNOTE,
    BlockType.IMAGE_FOOTNOTE,
    BlockType.TABLE_FOOTNOTE,
    BlockType.CHART_FOOTNOTE,
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.INTERLINE_EQUATION,
    BlockType.EQUATION,
    BlockType.LIST,
    BlockType.INDEX,
    BlockType.DISCARDED,
    BlockType.CODE,
    BlockType.CODE_BODY,
    BlockType.CODE_CAPTION,
    BlockType.CODE_FOOTNOTE,
    BlockType.ALGORITHM,
    BlockType.REF_TEXT,
    BlockType.PHONETIC,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
    BlockType.PAGE_FOOTNOTE,
    BlockType.ABSTRACT,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
    BlockType.VERTICAL_TEXT,
    BlockType.HEADER_IMAGE,
    BlockType.FOOTER_IMAGE,
    BlockType.FORMULA_NUMBER,
}

# NOT_EXTRACT_TYPES = {
#     BlockType.TEXT,
#     BlockType.TITLE,
#     BlockType.HEADER,
#     BlockType.FOOTER,
#     BlockType.PAGE_NUMBER,
#     BlockType.PAGE_FOOTNOTE,
#     BlockType.REF_TEXT,
#     BlockType.TABLE_CAPTION,
#     BlockType.IMAGE_CAPTION,
#     BlockType.TABLE_FOOTNOTE,
#     BlockType.IMAGE_FOOTNOTE,
#     BlockType.CODE_CAPTION,
#     BlockType.PHONETIC,
# }
# HYBRID_OCR_DET_TEXT_TYPES = NOT_EXTRACT_TYPES
#
#
# class NotExtractType(Enum):
#     TEXT = BlockType.TEXT
#     TITLE = BlockType.TITLE
#     HEADER = BlockType.HEADER
#     FOOTER = BlockType.FOOTER
#     PAGE_NUMBER = BlockType.PAGE_NUMBER
#     PAGE_FOOTNOTE = BlockType.PAGE_FOOTNOTE
#     REF_TEXT = BlockType.REF_TEXT
#     TABLE_CAPTION = BlockType.TABLE_CAPTION
#     IMAGE_CAPTION = BlockType.IMAGE_CAPTION
#     TABLE_FOOTNOTE = BlockType.TABLE_FOOTNOTE
#     IMAGE_FOOTNOTE = BlockType.IMAGE_FOOTNOTE
#     CODE_CAPTION = BlockType.CODE_CAPTION
#     PHONETIC = BlockType.PHONETIC
#
#
# not_extract_list = [item.value for item in NotExtractType] + [
#     BlockType.CAPTION,
#     BlockType.FOOTNOTE,
#     BlockType.DOC_TITLE,
#     BlockType.PARAGRAPH_TITLE,
# ]
# OCR_DET_LINES_KEY = "_ocr_det_lines"
# OCR_DET_LINE_BLOCK_TYPES = set(not_extract_list) | {
#     BlockType.LIST,
#     BlockType.INDEX,
#     BlockType.ABSTRACT,
#     BlockType.ASIDE_TEXT,
#     BlockType.PHONETIC,
#     BlockType.CHART_CAPTION,
#     BlockType.CHART_FOOTNOTE,
#     BlockType.CODE_FOOTNOTE,
# }


# 文本类 block 共用 text bbox 样式，避免新增文本形态时遗漏多个绘制入口。
TEXT_LIKE_BLOCK_TYPES_FOR_BBOX = {
    BlockType.TEXT,
    BlockType.REF_TEXT,
    BlockType.ABSTRACT,
    BlockType.PHONETIC,
}

# layout.pdf 中这些 block 直接使用自身 bbox，复合 block 则使用子 block bbox。
DIRECT_LAYOUT_BBOX_BLOCK_TYPES = TEXT_LIKE_BLOCK_TYPES_FOR_BBOX | {
    BlockType.TITLE,
    BlockType.INTERLINE_EQUATION,
    BlockType.LIST,
    BlockType.INDEX,
}

# span.pdf 从这些结构性 block 中收集内部 span bbox。
SPAN_SOURCE_BLOCK_TYPES = DIRECT_LAYOUT_BBOX_BLOCK_TYPES


IMAGE_BLOCK_BODY = "image_block_body"
GENERIC_CHILD_TYPES = (BlockType.CAPTION, BlockType.FOOTNOTE)
INLINE_CAPTION_FRAGMENT_TYPES = {BlockType.TEXT, BlockType.FOOTNOTE}
STACKED_TABLE_CAPTION_CLUSTER_TYPES = {
    BlockType.CAPTION,
    BlockType.TEXT,
    BlockType.FOOTNOTE,
}
VISUAL_RELATION_IGNORED_TYPES = {
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.PAGE_FOOTNOTE,
    BlockType.ASIDE_TEXT,
}
VISUAL_MAIN_TYPES = {
    BlockType.IMAGE_BODY: BlockType.IMAGE,
    IMAGE_BLOCK_BODY: BlockType.IMAGE,
    BlockType.TABLE_BODY: BlockType.TABLE,
    BlockType.CHART_BODY: BlockType.CHART,
    BlockType.CODE_BODY: BlockType.CODE,
}
VISUAL_TYPE_MAPPING = {
    BlockType.IMAGE: {
        "body": BlockType.IMAGE_BODY,
        "caption": BlockType.IMAGE_CAPTION,
        "footnote": BlockType.IMAGE_FOOTNOTE,
    },
    BlockType.TABLE: {
        "body": BlockType.TABLE_BODY,
        "caption": BlockType.TABLE_CAPTION,
        "footnote": BlockType.TABLE_FOOTNOTE,
    },
    BlockType.CHART: {
        "body": BlockType.CHART_BODY,
        "caption": BlockType.CHART_CAPTION,
        "footnote": BlockType.CHART_FOOTNOTE,
    },
    BlockType.CODE: {
        "body": BlockType.CODE_BODY,
        "caption": BlockType.CODE_CAPTION,
        "footnote": BlockType.CODE_FOOTNOTE,
    },
}


# ── dict-compatibility for dataclass-backed model objects ──────────
#
# Block / Line / Span / PageInfo are dataclasses whose canonical fields
# are declared below.  Backend code (MagicModel, para_split, union_make)
# also attaches *internal processing fields* (``index``, ``page_num``,
# ``_ocr_det_lines``, etc.) via ``block["key"] = value``.  Those
# non-canonical keys are stored in ``_extra`` so that the dataclass
# signature stays clean while still providing full dict compatibility
# during migration.


def _extra_getitem(self, key: str) -> Any:
    try:
        return getattr(self, key)
    except AttributeError:
        pass
    try:
        return self._extra[key]
    except (AttributeError, KeyError):
        raise KeyError(key) from None


def _extra_setitem(self, key: str, value: Any) -> None:
    if hasattr(self, key):
        setattr(self, key, value)
    else:
        self._extra[key] = value


def _extra_get(self, key: str, default: Any = None) -> Any:
    try:
        val = getattr(self, key)
    except AttributeError:
        return self._extra.get(key, default)
    return default if val is None and default is not None else val


def _extra_contains(self, key: str) -> bool:
    return hasattr(self, key) or key in self._extra


def _extra_delitem(self, key: str) -> None:
    if hasattr(self, key):
        raise KeyError(f"Cannot delete canonical field '{key}'")
    del self._extra[key]


def _extra_pop(self, key: str, default: Any = None) -> Any:
    val = _extra_get(self, key, default)
    if key in self._extra:
        del self._extra[key]
    return val


def _install_dict_compat(*classes) -> None:
    for cls in classes:
        cls.__getitem__ = _extra_getitem
        cls.__setitem__ = _extra_setitem
        cls.__delitem__ = _extra_delitem
        cls.get = _extra_get
        cls.pop = _extra_pop
        cls.__contains__ = _extra_contains


# ── model types ─────────────────────────────────────────────────────


@dataclass
class Span:
    """Leaf node of the block tree.  Holds text, formula, image, or table content."""

    type: str
    bbox: tuple[float, float, float, float] | None = None
    content: str = ""
    score: float = 0.0
    image_path: str = ""
    image_base64: str = ""
    html: str = ""
    latex: str = ""
    _extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> Span:
        bbox = d.get("bbox")
        return cls(
            type=d["type"],
            bbox=tuple(bbox) if bbox else None,
            content=d.get("content", ""),
            score=d.get("score", 0.0),
            image_path=d.get("image_path", ""),
            image_base64=d.get("image_base64", ""),
            html=d.get("html", ""),
            latex=d.get("latex", ""),
        )


@dataclass
class Line:
    """A line within a block, containing one or more spans."""

    spans: list[Span] = field(default_factory=list)
    bbox: tuple[float, float, float, float] | None = None
    _extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> Line:
        return cls(spans=[Span.from_dict(s) for s in d.get("spans", [])])


@dataclass
class Block:
    """A layout block on a page.  May nest child blocks (e.g. list items, image body)."""

    type: str
    bbox: tuple[float, float, float, float] | None = None
    lines: list[Line] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    level: int | None = None
    section_number: str = ""
    sub_type: str = ""
    html: str = ""
    merge_prev: bool = False
    index: int | None = None
    guess_lang: str = ""
    _extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> Block:
        bbox = d.get("bbox")
        return cls(
            type=d["type"],
            bbox=tuple(bbox) if bbox else None,
            lines=[Line.from_dict(line) for line in d.get("lines", [])],
            blocks=[Block.from_dict(b) for b in d.get("blocks", [])],
            level=d.get("level"),
            section_number=d.get("section_number", ""),
            sub_type=d.get("sub_type", ""),
            html=d.get("html", ""),
            merge_prev=d.get("merge_prev", False),
            index=d.get("index"),
            guess_lang=d.get("guess_lang", ""),
        )

    def all_spans(self) -> Iterator[Span]:
        """Depth-first yield every span in this block tree."""
        for line in self.lines:
            yield from line.spans
        for child in self.blocks:
            yield from child.all_spans()


@dataclass
class PageInfo:
    """Parsed content of a single page."""

    page_idx: int
    page_size: tuple[int, int] | list[int, int] | None = None
    preproc_blocks: list[Block] = field(default_factory=list)
    para_blocks: list[Block] = field(default_factory=list)
    discarded_blocks: list[Block] = field(default_factory=list)
    _extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> PageInfo:
        ps = d.get("page_size")
        return cls(
            page_idx=d["page_idx"],
            page_size=tuple(ps) if ps else None,
            preproc_blocks=[Block.from_dict(b) for b in d.get("preproc_blocks", [])],
            para_blocks=[Block.from_dict(b) for b in d.get("para_blocks", [])],
            discarded_blocks=[Block.from_dict(b) for b in d.get("discarded_blocks", [])],
        )


_install_dict_compat(Span, Line, Block, PageInfo)
