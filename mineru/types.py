# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import MISSING, Field, InitVar, dataclass, field, fields
from typing import Any, Iterator, Literal, TypeAlias, TypeVar, get_type_hints

T = TypeVar("T", bound="_DocElement")


Tier = Literal[
    "flash",
    "medium",
    "high",
    "xhigh",
]

TIERS: set[Tier] = {
    "flash",
    "medium",
    "high",
    "xhigh",
}

TIER_ORDER: dict[Tier, int] = {
    "flash": 0,
    "medium": 1,
    "high": 2,
    "xhigh": 3,
}

QUALITY_TIER_SELECTION_ORDER: tuple[Tier, ...] = ("high", "xhigh", "medium")
QUALITY_TIERS: frozenset[Tier] = frozenset(QUALITY_TIER_SELECTION_ORDER)
CACHED_TIER_SELECTION_ORDER: tuple[Tier, ...] = ("xhigh", "high", "medium", "flash")


def validate_tier(tier: str | None) -> Tier:
    """校验公开 tier 取值，保证入口只接受 flash/medium/high/xhigh。"""
    normalized = (tier or "").strip().lower()
    if normalized in TIERS:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"Unsupported tier '{tier}'. Supported tiers: {', '.join(TIERS)}")


def _validated_tier_set(available_tiers: Iterable[object] | str) -> set[Tier]:
    if isinstance(available_tiers, str):
        return {validate_tier(available_tiers)}
    return {validate_tier(str(item)) for item in available_tiers}


def select_default_quality_tier(available_tiers: Iterable[object] | str) -> Tier | None:
    """Select the default quality tier from discovered parse-server capabilities."""
    available = _validated_tier_set(available_tiers)
    for candidate in QUALITY_TIER_SELECTION_ORDER:
        if candidate in available:
            return candidate
    return None


def select_highest_cached_tier(available_tiers: Iterable[object] | str) -> Tier | None:
    """Select the highest already-cached tier without creating a new parse."""
    available = _validated_tier_set(available_tiers)
    for candidate in CACHED_TIER_SELECTION_ORDER:
        if candidate in available:
            return candidate
    return None


class BlockType:
    IMAGE = "image"
    IMAGE_BODY = "image_body"
    IMAGE_CAPTION = "image_caption"
    IMAGE_FOOTNOTE = "image_footnote"

    TABLE = "table"
    TABLE_BODY = "table_body"
    TABLE_CAPTION = "table_caption"
    TABLE_FOOTNOTE = "table_footnote"

    CHART = "chart"
    CHART_BODY = "chart_body"
    CHART_CAPTION = "chart_caption"
    CHART_FOOTNOTE = "chart_footnote"

    # Added in vlm 2.5
    CODE = "code"
    CODE_BODY = "code_body"
    CODE_CAPTION = "code_caption"
    CODE_FOOTNOTE = "code_footnote"

    CAPTION = "caption"  # generic caption type (e.g., for Word documents)
    ALGORITHM_CAPTION = "algorithm_caption"
    FOOTNOTE = "footnote"  # pp_layout中的vision_footnote
    TEXT = "text"
    TITLE = "title"
    INTERLINE_EQUATION = "interline_equation"
    EQUATION = "equation"  # 公式(独立公式)
    LIST = "list"
    INDEX = "index"
    DISCARDED = "discarded"

    # Added in vlm 2.5
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

NOT_EXTRACT_TYPES = {
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.PAGE_FOOTNOTE,
    BlockType.REF_TEXT,
    BlockType.TABLE_CAPTION,
    BlockType.IMAGE_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.IMAGE_FOOTNOTE,
    BlockType.CODE_CAPTION,
    BlockType.PHONETIC,
}

# NOT_EXTRACT_TYPES = {
#     BlockType.TEXT,
#     BlockType.HEADER,
#     BlockType.FOOTER,
#     BlockType.PAGE_NUMBER,
#     BlockType.PAGE_FOOTNOTE,
#     BlockType.REF_TEXT,
#     BlockType.PHONETIC,
#     # Titles
#     BlockType.TITLE,
#     BlockType.DOC_TITLE,
#     BlockType.PARAGRAPH_TITLE,
#     # Captions
#     BlockType.CAPTION,
#     BlockType.IMAGE_CAPTION,
#     BlockType.TABLE_CAPTION,
#     BlockType.CHART_CAPTION,
#     BlockType.CODE_CAPTION,
#     # Footnotes
#     BlockType.FOOTNOTE,
#     BlockType.IMAGE_FOOTNOTE,
#     BlockType.TABLE_FOOTNOTE,
#     BlockType.CHART_FOOTNOTE,
#     BlockType.CODE_FOOTNOTE,
#     # Others
#     BlockType.LIST,
#     BlockType.INDEX,
#     BlockType.ABSTRACT,
#     BlockType.ASIDE_TEXT,
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


# ── model types ─────────────────────────────────────────────────────

BBox: TypeAlias = tuple[float, float, float, float]
IntBBox: TypeAlias = tuple[int, int, int, int]


def _is_default_value(f: Field, value: Any) -> bool:
    if f.default is not MISSING:
        return value == f.default
    if f.default_factory is not MISSING:
        return value == f.default_factory()
    return False


def _initvar_default(value: Any, default: Any) -> Any:
    """修正 InitVar 与同名 property 共存时产生的 property 默认值。"""
    return default if isinstance(value, property) else value


EMPTY_BBOX: BBox = (0.0, 0.0, 0.0, 0.0)


def _origin_is_list(tp: Any) -> bool:
    return getattr(tp, "__origin__", None) is list


def _list_arg(tp: Any) -> Any | None:
    args = getattr(tp, "__args__", None)
    return args[0] if args else None


@dataclass
class _DocElement:
    """Base class for document-model nodes (Span, Line, Block, PageInfo)."""

    def to_dict(self, *, skip_defaults: bool = True) -> dict[str, Any]:
        """Serialize to dict, excluding private fields (prefixed with ``_``).

        When *skip_defaults* is True, fields whose value equals their
        default are also omitted.
        """
        result: dict[str, Any] = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            value = getattr(self, f.name)
            if skip_defaults and _is_default_value(f, value):
                continue
            if isinstance(value, list):
                result[f.name] = [v.to_dict(skip_defaults=skip_defaults) if isinstance(v, _DocElement) else v for v in value]
            else:
                result[f.name] = value
        return result

    @classmethod
    def from_dict(cls: type[T], d: dict) -> T:
        """Reconstruct from dict (inverse of ``to_dict``). Handles nested dataclass lists and list→tuple conversion."""
        hints = get_type_hints(cls)
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name.startswith("_"):
                continue
            if f.name not in d:
                continue
            value = d[f.name]
            f_type = hints.get(f.name)
            if f_type is not None and _origin_is_list(f_type):
                elem_type = _list_arg(f_type)
                if isinstance(elem_type, type) and issubclass(elem_type, _DocElement):
                    value = [elem_type.from_dict(v) for v in value]
            if f.name in ("bbox", "page_size") and isinstance(value, list):
                value = tuple(value)
            kwargs[f.name] = value
        return cls(**kwargs)


@dataclass
class Span(_DocElement):
    """Leaf node of the block tree.  Holds text, formula, image, or table content."""

    type: str
    bbox: BBox
    content: str = ""
    score: float = 0.0
    image_path: str = ""

    # Internal
    _cross_page: bool = False
    _np_img: Any = None

    _url: str = ""
    _style: list[str] = field(default_factory=list)
    _children: list[Span] = field(default_factory=list)

    _extra: dict = field(default_factory=dict)


@dataclass
class Line(_DocElement):
    """A line within a block, containing one or more spans."""

    bbox: BBox
    spans: list[Span] = field(default_factory=list)

    # Internal
    _is_list_start: bool = False
    _is_list_end: bool = False

    _code_type: str | None = None
    _code_guess_lang: str | None = None


@dataclass
class Block(_DocElement):
    """A layout block on a page.  May nest child blocks (e.g. list items, image body)."""

    # Required
    index: int
    type: str
    bbox: BBox
    lines: list[Line] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)

    # Optional
    level: int | None = None
    sub_type: str = ""
    guess_lang: str = ""
    section_number: str = ""

    # Office
    anchor: str = ""
    start: int | None = None
    ilevel: int | None = None

    # Init-only draft fields.  These are accepted for backend conversion
    # compatibility, but are not middle_json block fields and never serialize.
    angle: InitVar[int | None] = None
    score: InitVar[float | None] = None
    merge_prev: InitVar[bool] = False
    is_numbered_style: InitVar[bool] = False

    # Internal
    _cross_page: bool = False
    _angle: int | None = None
    _layout_score: float | None = None
    _merge_prev: bool = False
    _is_numbered_style: bool = False
    _lines_deleted: bool = False
    _ocr_det_lines: list[Line] = field(default_factory=list)
    _line_avg_height: int = 0
    _cell_merge: list[int] = field(default_factory=list)
    _fix_spans: list[Span] = field(default_factory=list)
    _list_attribute: str = ""
    _page_num: int | None = None
    _page_size: tuple[float, float] | None = None
    _bbox_fs: BBox | None = None
    _sub_images: list[Block] = field(default_factory=list)

    def __post_init__(
        self,
        angle: int | None,
        score: float | None,
        merge_prev: bool,
        is_numbered_style: bool,
    ) -> None:
        """接收转换阶段临时字段，但只存入内部属性，避免污染 middle_json 输出。"""
        self._angle = _initvar_default(angle, None)
        self._layout_score = _initvar_default(score, None)
        self._merge_prev = _initvar_default(merge_prev, False)
        self._is_numbered_style = _initvar_default(is_numbered_style, False)

    def to_dict(self, *, skip_defaults: bool = True) -> dict[str, Any]:
        """仅在 staged middle_json 需要时输出 merge_prev 合并提示。"""
        result = super().to_dict(skip_defaults=skip_defaults)
        if self.merge_prev or not skip_defaults:
            result["merge_prev"] = self.merge_prev
        return result

    @classmethod
    def from_dict(cls, d: dict) -> Block:
        """读取 staged middle_json 中的 merge_prev 合并提示。"""
        block = _DocElement.from_dict.__func__(cls, d)
        if "merge_prev" in d:
            block.merge_prev = bool(d.get("merge_prev", False))
        return block

    @property
    def angle(self) -> int | None:
        """转换阶段读取的原始旋转角，默认不作为 middle_json block 字段输出。"""
        return self._angle

    @angle.setter
    def angle(self, value: int | None) -> None:
        self._angle = value

    @property
    def score(self) -> float | None:
        """layout 置信度只作为内部调试/传递信息保存，不进入 public block。"""
        return self._layout_score

    @score.setter
    def score(self, value: float | None) -> None:
        self._layout_score = value

    @property
    def merge_prev(self) -> bool:
        """raw content block 的合并提示，仅供段落合并阶段消费。"""
        return self._merge_prev

    @merge_prev.setter
    def merge_prev(self, value: bool) -> None:
        self._merge_prev = value

    @property
    def is_numbered_style(self) -> bool:
        """Office 标题编号中间状态，生成 section_number 后不输出。"""
        return self._is_numbered_style

    @is_numbered_style.setter
    def is_numbered_style(self, value: bool) -> None:
        self._is_numbered_style = value

    def all_spans(self) -> Iterator[Span]:
        """Depth-first yield every span in this block tree."""
        for line in self.lines:
            yield from line.spans
        for child in self.blocks:
            yield from child.all_spans()


@dataclass
class PageInfo(_DocElement):
    """Parsed content of a single page."""

    page_idx: int
    page_size: tuple[int, int] | None = None
    preproc_blocks: list[Block] = field(default_factory=list)
    para_blocks: list[Block] = field(default_factory=list)
    discarded_blocks: list[Block] = field(default_factory=list)

    # Temporary — will be removed once the render layer converges.
    _backend: Literal["hybrid", "office"] | None = None


@dataclass
class ContentItem(_DocElement):
    """A single item in the Content List (V1) output format."""

    type: str
    page_idx: int = 0
    bbox: IntBBox | None = None
    text: str | None = None
    text_level: int | None = None
    text_format: str | None = None
    img_path: str | None = None
    content: str | None = None
    sub_type: str | None = None
    list_items: list[str] = field(default_factory=list)
    image_caption: list[str] = field(default_factory=list)
    image_footnote: list[str] = field(default_factory=list)
    table_caption: list[str] = field(default_factory=list)
    table_footnote: list[str] = field(default_factory=list)
    table_body: str | None = None
    chart_caption: list[str] = field(default_factory=list)
    chart_footnote: list[str] = field(default_factory=list)
    code_caption: list[str] = field(default_factory=list)
    code_footnote: list[str] = field(default_factory=list)
    code_body: str | None = None
