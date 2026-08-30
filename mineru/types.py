# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, ClassVar, Literal, TypeAlias, Union, cast, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from .utils.hyperlink import OFFICE_EXTERNAL_HYPERLINK_SCHEMES, sanitize_hyperlink_target

# 这些字符串不能作为公开 Block.type discriminator，只用于 raw 阶段或 Block 内部枚举值。
RawBlockType: TypeAlias = Literal[
    "algorithm",
    "caption",
    "footnote",
    "formula_number",
    "phonetic",
]

RAW_ALGORITHM: RawBlockType = "algorithm"
RAW_CAPTION: RawBlockType = "caption"
RAW_FOOTNOTE: RawBlockType = "footnote"
RAW_FORMULA_NUMBER: RawBlockType = "formula_number"
RAW_PHONETIC: RawBlockType = "phonetic"

RAW_ONLY_BLOCK_TYPES = frozenset(
    {
        RAW_ALGORITHM,
        RAW_CAPTION,
        RAW_FOOTNOTE,
        RAW_FORMULA_NUMBER,
        RAW_PHONETIC,
    }
)

FileSuffix: TypeAlias = Literal[
    "pdf",
    "doc",
    "docx",
    "ppt",
    "pptx",
    "xls",
    "xlsx",
    "rtf",
    "csv",
    "epub",
    "html",
    "ofd",
    "odt",
    "ods",
    "odp",
]
FILE_SUFFIXES: frozenset[FileSuffix] = frozenset(cast(tuple[FileSuffix, ...], get_args(FileSuffix)))

Tier = Literal[
    "flash",
    "basic",
    "standard",
    "advanced",
]

TIERS: tuple[Tier, ...] = (
    "flash",
    "basic",
    "standard",
    "advanced",
)

TIER_ORDER: dict[Tier, int] = {
    "flash": 0,
    "basic": 1,
    "standard": 2,
    "advanced": 3,
}

ServerTier = Literal[
    "flash",
    "basic",
    "standard",
]

SERVER_TIERS: tuple[ServerTier, ...] = (
    "flash",
    "basic",
    "standard",
)

TIERS_BY_SERVER_TIER: dict[ServerTier, tuple[Tier, ...]] = {
    "flash": ("flash",),
    "basic": ("flash", "basic"),
    "standard": ("flash", "basic", "standard", "advanced"),
}

DeploymentTier = Literal[
    "basic",
    "standard",
]

DEPLOYMENT_TIERS: tuple[DeploymentTier, ...] = (
    "basic",
    "standard",
)


DEFAULT_QUALITY_TIER_SELECTION_ORDER: tuple[Tier, ...] = ("standard", "basic")
QUALITY_TIERS: frozenset[Tier] = frozenset(("basic", "standard", "advanced"))
CACHED_TIER_SELECTION_ORDER: tuple[Tier, ...] = ("advanced", "standard", "basic", "flash")
PARSING_RULE_TIER_SELECTION_ORDER: tuple[Tier, ...] = (*DEFAULT_QUALITY_TIER_SELECTION_ORDER, "flash")


def validate_tier(tier: str | None) -> Tier:
    """校验公开 tier 取值，保证入口只接受 flash/basic/standard/advanced。"""
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
    for candidate in DEFAULT_QUALITY_TIER_SELECTION_ORDER:
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


def select_parsing_rule_tier(available_tiers: Iterable[object] | str | None = None) -> Tier:
    """Select parsing-rule default tier, allowing flash as a final fallback."""
    available = _validated_tier_set(available_tiers or PARSING_RULE_TIER_SELECTION_ORDER)
    for candidate in PARSING_RULE_TIER_SELECTION_ORDER:
        if candidate in available:
            return candidate
    return "flash"


class BlockType(str, Enum):
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
    ALGORITHM_BODY = "algorithm_body"
    CODE_CAPTION = "code_caption"
    CODE_FOOTNOTE = "code_footnote"

    TEXT = "text"
    EQUATION = "equation"  # 行间公式（独立公式）
    LIST = "list"
    INDEX = "index"

    # Added in vlm 2.5
    REF_TEXT = "ref_text"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    ASIDE_TEXT = "aside_text"
    PAGE_FOOTNOTE = "page_footnote"

    # Added in pp_doclayout_v2
    DOC_TITLE = "doc_title"
    PARAGRAPH_TITLE = "paragraph_title"

    def __str__(self) -> str:
        return self.value


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
    BlockType.IMAGE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.CHART_CAPTION,
    BlockType.IMAGE_FOOTNOTE,
    BlockType.TABLE_FOOTNOTE,
    BlockType.CHART_FOOTNOTE,
    BlockType.TEXT,
    BlockType.EQUATION,
    BlockType.LIST,
    BlockType.INDEX,
    BlockType.CODE,
    BlockType.CODE_BODY,
    BlockType.ALGORITHM_BODY,
    BlockType.CODE_CAPTION,
    BlockType.CODE_FOOTNOTE,
    BlockType.REF_TEXT,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
    BlockType.PAGE_FOOTNOTE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
]

PageBlockTypes = Literal[
    BlockType.IMAGE,
    BlockType.TABLE,
    BlockType.CHART,
    BlockType.TEXT,
    BlockType.EQUATION,
    BlockType.LIST,
    BlockType.INDEX,
    BlockType.CODE,
    BlockType.REF_TEXT,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
    BlockType.PAGE_FOOTNOTE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
]

BLOCK_TYPES = {
    BlockType.IMAGE,
    BlockType.TABLE,
    BlockType.CHART,
    BlockType.IMAGE_BODY,
    BlockType.TABLE_BODY,
    BlockType.CHART_BODY,
    BlockType.IMAGE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.CHART_CAPTION,
    BlockType.IMAGE_FOOTNOTE,
    BlockType.TABLE_FOOTNOTE,
    BlockType.CHART_FOOTNOTE,
    BlockType.TEXT,
    BlockType.EQUATION,
    BlockType.LIST,
    BlockType.INDEX,
    BlockType.CODE,
    BlockType.CODE_BODY,
    BlockType.ALGORITHM_BODY,
    BlockType.CODE_CAPTION,
    BlockType.CODE_FOOTNOTE,
    BlockType.REF_TEXT,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
    BlockType.PAGE_FOOTNOTE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
}

PAGE_BLOCK_TYPES = {
    BlockType.IMAGE,
    BlockType.TABLE,
    BlockType.CHART,
    BlockType.TEXT,
    BlockType.EQUATION,
    BlockType.LIST,
    BlockType.INDEX,
    BlockType.CODE,
    BlockType.REF_TEXT,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
    BlockType.PAGE_FOOTNOTE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
}

# 页面装饰与辅助文本不参与正文、列表和视觉对象之间的语义边界判断。
PAGE_AUXILIARY_BLOCK_TYPES = {
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
}
# 页面脚注需要参与输出，但不会阻断正文、列表、续表或视觉对象之间的关系判断。
MERGE_TRANSPARENT_BLOCK_TYPES = {
    *PAGE_AUXILIARY_BLOCK_TYPES,
    BlockType.PAGE_FOOTNOTE,
}
VISUAL_RELATION_IGNORED_TYPES = MERGE_TRANSPARENT_BLOCK_TYPES
VISUAL_MAIN_TYPES = {
    BlockType.IMAGE_BODY: BlockType.IMAGE,
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


def _remove_block_fields(value: Any, excluded_fields: set[str]) -> Any:
    """递归删除序列化结果中指定的 block 字段，覆盖任意深度的容器。"""
    if isinstance(value, list):
        return [_remove_block_fields(item, excluded_fields) for item in value]
    if not isinstance(value, dict):
        return value

    result = {
        key: _remove_block_fields(item, excluded_fields)
        for key, item in value.items()
        if not ("type" in value and key in excluded_fields)
    }
    return result


class _StrictMiddleModel(BaseModel):
    """Model/Middle JSON 严格模型基类，提供无副作用的统一序列化入口。"""

    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    def to_dict(
        self,
        *,
        skip_defaults: bool = True,
        exclude_none: bool = False,
        exclude_block_fields: set[str] | None = None,
    ) -> dict[str, Any]:
        """序列化对象，并按字段名递归排除任意层级的 block 字段。"""
        payload = self.model_dump(
            mode="json",
            exclude_defaults=skip_defaults,
            exclude_none=exclude_none,
        )
        if exclude_block_fields:
            payload = _remove_block_fields(payload, set(exclude_block_fields))
        return payload

    def to_json(
        self,
        *,
        skip_defaults: bool = True,
        exclude_none: bool = False,
        exclude_block_fields: set[str] | None = None,
        indent: int | None = 4,
    ) -> str:
        """将对象编码为 UTF-8 友好的 JSON 字符串，不执行图片文件写入。"""
        return json.dumps(
            self.to_dict(
                skip_defaults=skip_defaults,
                exclude_none=exclude_none,
                exclude_block_fields=exclude_block_fields,
            ),
            ensure_ascii=False,
            indent=indent,
        )


InlineStyle: TypeAlias = Literal[
    "bold",
    "italic",
    "underline",
    "emphasis",
    "strikethrough",
    "superscript",
    "subscript",
]

INLINE_STYLE_ORDER: tuple[InlineStyle, ...] = (
    "bold",
    "italic",
    "underline",
    "emphasis",
    "strikethrough",
    "superscript",
    "subscript",
)


class TextSpan(_StrictMiddleModel):
    """保存普通行内文字及其可见字体样式。"""

    type: Literal["text"]
    content: str = Field(min_length=1)
    styles: list[InlineStyle] = Field(default_factory=list)

    @field_validator("styles")
    @classmethod
    def _normalize_styles(cls, value: list[InlineStyle]) -> list[InlineStyle]:
        """按公开固定顺序去重样式，并禁止同时声明上下标。"""
        unique = set(value)
        if "superscript" in unique and "subscript" in unique:
            raise ValueError("text span cannot be both superscript and subscript")
        return [style for style in INLINE_STYLE_ORDER if style in unique]


class EquationInlineSpan(_StrictMiddleModel):
    """保存不含外层定界符的行内 LaTeX。"""

    type: Literal["equation_inline"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        """拒绝只包含空白的行内公式，同时保留公式原始空白。"""
        if not value.strip():
            raise ValueError("inline equation content must not be blank")
        return value


class CodeInlineSpan(_StrictMiddleModel):
    """保存需要按字面量显示的行内代码。"""

    type: Literal["code_inline"]
    content: str = Field(min_length=1)


NonLinkInlineSpan: TypeAlias = Annotated[
    Union[TextSpan, EquationInlineSpan, CodeInlineSpan],
    Field(discriminator="type"),
]


class HyperlinkSpan(_StrictMiddleModel):
    """保存安全超链接目标及其非链接行内子节点。"""

    type: Literal["hyperlink"]
    url: str = Field(min_length=1)
    content: list[NonLinkInlineSpan] = Field(min_length=1)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        """复用统一策略拒绝危险协议、本地路径、畸形 URL 和控制字符。"""
        normalized = sanitize_hyperlink_target(
            value,
            allowed_schemes=OFFICE_EXTERNAL_HYPERLINK_SCHEMES,
            allow_relative=True,
            allow_fragment=True,
        )
        if normalized is None:
            raise ValueError("hyperlink span url is unsafe or malformed")
        return normalized


InlineSpan: TypeAlias = Annotated[
    Union[TextSpan, EquationInlineSpan, CodeInlineSpan, HyperlinkSpan],
    Field(discriminator="type"),
]

INLINE_SPAN_ADAPTER = TypeAdapter(InlineSpan)
INLINE_SPAN_LIST_ADAPTER = TypeAdapter(list[InlineSpan])


def _normalize_typed_inline_spans(spans: list[InlineSpan]) -> list[InlineSpan]:
    """递归合并相邻同样式文字及相邻同目标链接。"""
    normalized: list[InlineSpan] = []
    for span in spans:
        current: InlineSpan
        if isinstance(span, HyperlinkSpan):
            children = _normalize_typed_inline_spans(list(span.content))
            non_link_children = [child for child in children if not isinstance(child, HyperlinkSpan)]
            if not non_link_children:
                continue
            current = span.model_copy(update={"content": non_link_children}, deep=True)
        else:
            current = span.model_copy(deep=True)
        if (
            normalized
            and isinstance(normalized[-1], TextSpan)
            and isinstance(current, TextSpan)
            and normalized[-1].styles == current.styles
        ):
            previous = normalized[-1]
            normalized[-1] = previous.model_copy(update={"content": f"{previous.content}{current.content}"})
            continue
        if (
            normalized
            and isinstance(normalized[-1], HyperlinkSpan)
            and isinstance(current, HyperlinkSpan)
            and normalized[-1].url == current.url
        ):
            previous_link = normalized[-1]
            merged_children = _normalize_typed_inline_spans([*previous_link.content, *current.content])
            normalized[-1] = previous_link.model_copy(update={"content": merged_children})
            continue
        normalized.append(current)
    return normalized


def parse_inline_span(value: Any) -> InlineSpan:
    """把字典或现有模型严格解析为一个公开行内 Span。"""
    return INLINE_SPAN_ADAPTER.validate_python(value)


def parse_inline_spans(value: Any) -> list[InlineSpan]:
    """严格解析并规范化完整行内 Span 列表。"""
    return _normalize_typed_inline_spans(INLINE_SPAN_LIST_ADAPTER.validate_python(value))


class BlockBase(_StrictMiddleModel):
    """所有公开 Middle JSON block 的最小公共字段。"""

    type: BlockTypes
    index: int | None = Field(default=None, ge=0)
    bbox: BBox | None = None

    @field_validator("bbox", mode="before")
    @classmethod
    def _validate_bbox(cls, value: Any) -> BBox | None:
        """接受 JSON 数组形式的 bbox，并严格校验归一化坐标。"""
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("bbox must contain exactly four numbers")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
            raise ValueError("bbox values must be numbers")
        bbox = tuple(float(item) for item in value)
        if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in bbox):
            raise ValueError("bbox values must be finite normalized coordinates")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("bbox must satisfy x1 > x0 and y1 > y0")
        return bbox  # type: ignore[return-value]


class StringContentBlock(BlockBase):
    """所有字符串内容 block 的共享结构。"""

    content: str


class InlineContentBlock(BlockBase):
    """所有结构化行内内容 block 的共享结构。"""

    content: list[InlineSpan]

    @field_validator("content")
    @classmethod
    def _normalize_content(cls, value: list[InlineSpan]) -> list[InlineSpan]:
        """在严格对象边界合并相邻同语义 Span。"""
        return _normalize_typed_inline_spans(value)


class ContinuableTextBlockBase(InlineContentBlock):
    """正文与参考文献共享的跨块续接结构。"""

    continues_prev: bool | None = None


class TextBlock(ContinuableTextBlockBase):
    type: Literal[BlockType.TEXT]  # type: ignore[reportIncompatibleVariableOverride]


class RefTextBlock(ContinuableTextBlockBase):
    type: Literal[BlockType.REF_TEXT]  # type: ignore[reportIncompatibleVariableOverride]


class TitleBlockBase(InlineContentBlock):
    """文档标题与段落标题的全局层级公共结构。"""

    anchor: str | None = None
    level: int


class DocTitleBlock(TitleBlockBase):
    type: Literal[BlockType.DOC_TITLE]  # type: ignore[reportIncompatibleVariableOverride]
    level: int = Field(ge=1, le=1)


class ParagraphTitleBlock(TitleBlockBase):
    type: Literal[BlockType.PARAGRAPH_TITLE]  # type: ignore[reportIncompatibleVariableOverride]
    level: int = Field(ge=2, le=6)


class PageAuxTextBlock(InlineContentBlock):
    """页眉、页脚、页码和边栏的共享文本结构。"""

    type: Literal[  # type: ignore[reportIncompatibleVariableOverride]
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.ASIDE_TEXT,
    ]


class PageFootnoteBlock(InlineContentBlock):
    """保存需要参与默认输出并可被文档内链接引用的页面脚注。"""

    type: Literal[BlockType.PAGE_FOOTNOTE]  # type: ignore[reportIncompatibleVariableOverride]
    anchor: str | None = None


class ImagePayloadBlock(BlockBase):
    """统一携带 sidecar、data URI 或远程 URL 的图片 block 基类。"""

    image_base64: str | None = Field(default=None, repr=False)
    image_path: str | None = None
    image_url: str | None = None

    @field_validator("image_path")
    @classmethod
    def _validate_image_path(cls, value: str | None) -> str | None:
        """校验已记录的图片路径只能是安全的 POSIX 相对路径。"""
        if value is None:
            return None
        from .utils.image_payload import validate_image_sidecar_path

        return validate_image_sidecar_path(value)

    @field_validator("image_url")
    @classmethod
    def _validate_image_url(cls, value: str | None) -> str | None:
        """校验远程图片 URL，禁止活动协议、相对地址与内嵌凭据。"""
        if value is None:
            return None
        from .utils.image_payload import validate_remote_image_url

        return validate_remote_image_url(value)


class ImagePayloadContentBlock(ImagePayloadBlock):
    """统一携带字符串内容和图片载荷的 block 结构。"""

    content: str


class EquationBlock(ImagePayloadContentBlock):
    type: Literal[BlockType.EQUATION]  # type: ignore[reportIncompatibleVariableOverride]


class ImageBodyBlock(ImagePayloadContentBlock):
    type: Literal[BlockType.IMAGE_BODY]  # type: ignore[reportIncompatibleVariableOverride]


class TableBodyBlock(ImagePayloadContentBlock):
    type: Literal[BlockType.TABLE_BODY]  # type: ignore[reportIncompatibleVariableOverride]


class ChartBodyBlock(ImagePayloadContentBlock):
    type: Literal[BlockType.CHART_BODY]  # type: ignore[reportIncompatibleVariableOverride]


class CodeBodyBlock(StringContentBlock):
    type: Literal[BlockType.CODE_BODY]  # type: ignore[reportIncompatibleVariableOverride]


class AlgorithmBodyBlock(InlineContentBlock):
    """保存预格式算法文字与行内公式 Span。"""

    type: Literal[BlockType.ALGORITHM_BODY]  # type: ignore[reportIncompatibleVariableOverride]


class ImageAnnotationBlock(InlineContentBlock):
    """图片标题与图片脚注的共享结构。"""

    type: Literal[BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE]  # type: ignore[reportIncompatibleVariableOverride]


class TableAnnotationBlock(InlineContentBlock):
    """表格标题与表格脚注的共享结构。"""

    type: Literal[BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE]  # type: ignore[reportIncompatibleVariableOverride]


class ChartAnnotationBlock(InlineContentBlock):
    """图表标题与图表脚注的共享结构。"""

    type: Literal[BlockType.CHART_CAPTION, BlockType.CHART_FOOTNOTE]  # type: ignore[reportIncompatibleVariableOverride]


class CodeAnnotationBlock(InlineContentBlock):
    """代码标题与代码脚注的共享结构。"""

    type: Literal[BlockType.CODE_CAPTION, BlockType.CODE_FOOTNOTE]  # type: ignore[reportIncompatibleVariableOverride]


ListChildBlock: TypeAlias = Annotated[
    Union[TextBlock, RefTextBlock, "ListBlock"],
    Field(discriminator="type"),
]


class ListBlock(BlockBase):
    type: Literal[BlockType.LIST]  # type: ignore[reportIncompatibleVariableOverride]
    content: list[ListChildBlock]
    sub_type: Literal[BlockType.TEXT, BlockType.REF_TEXT] | None = None
    continues_prev: bool | None = None


IndexChildBlock: TypeAlias = Annotated[
    Union[TextBlock, DocTitleBlock, ParagraphTitleBlock, "IndexBlock"],
    Field(discriminator="type"),
]


class IndexBlock(BlockBase):
    type: Literal[BlockType.INDEX]  # type: ignore[reportIncompatibleVariableOverride]
    content: list[IndexChildBlock]


class _VisualBlockBase(BlockBase):
    """视觉父块的共享结构约束。"""

    _body_types: ClassVar[tuple[str, ...]]

    @model_validator(mode="after")
    def _validate_visual_children(self) -> _VisualBlockBase:
        """校验视觉父块只有一个 body，且父子定位字段保持一致。"""
        children = getattr(self, "content", [])
        bodies = [child for child in children if child.type in self._body_types]
        if len(bodies) != 1:
            expected = "/".join(str(item) for item in self._body_types)
            raise ValueError(f"{self.type} must contain exactly one {expected}")
        body = bodies[0]
        if self.index is not None and body.index != self.index:
            raise ValueError(f"{self.type} body index must equal parent index")
        if self.bbox is not None and body.bbox is not None and body.bbox != self.bbox:
            raise ValueError(f"{self.type} body bbox must equal parent bbox")
        return self


ImageChildBlock: TypeAlias = Annotated[
    Union[ImageBodyBlock, ImageAnnotationBlock],
    Field(discriminator="type"),
]


class ImageBlock(_VisualBlockBase):
    type: Literal[BlockType.IMAGE]  # type: ignore[reportIncompatibleVariableOverride]
    content: list[ImageChildBlock]
    sub_type: str | None = None
    _body_types: ClassVar[tuple[str, ...]] = (BlockType.IMAGE_BODY,)


TableChildBlock: TypeAlias = Annotated[
    Union[TableBodyBlock, TableAnnotationBlock],
    Field(discriminator="type"),
]


class TableBlock(_VisualBlockBase):
    type: Literal[BlockType.TABLE]  # type: ignore[reportIncompatibleVariableOverride]
    content: list[TableChildBlock]
    continues_prev: bool | None = None
    cell_merge: list[Literal[0, 1]] | None = None
    _body_types: ClassVar[tuple[str, ...]] = (BlockType.TABLE_BODY,)


ChartChildBlock: TypeAlias = Annotated[
    Union[ChartBodyBlock, ChartAnnotationBlock],
    Field(discriminator="type"),
]


class ChartBlock(_VisualBlockBase):
    type: Literal[BlockType.CHART]  # type: ignore[reportIncompatibleVariableOverride]
    content: list[ChartChildBlock]
    sub_type: str | None = None
    _body_types: ClassVar[tuple[str, ...]] = (BlockType.CHART_BODY,)


CodeChildBlock: TypeAlias = Annotated[
    Union[CodeBodyBlock, AlgorithmBodyBlock, CodeAnnotationBlock],
    Field(discriminator="type"),
]


class CodeBlock(_VisualBlockBase):
    type: Literal[BlockType.CODE]  # type: ignore[reportIncompatibleVariableOverride]
    content: list[CodeChildBlock]
    sub_type: Literal[BlockType.CODE, RAW_ALGORITHM]
    guess_lang: str | None = None
    _body_types: ClassVar[tuple[str, ...]] = (BlockType.CODE_BODY, BlockType.ALGORITHM_BODY)

    @model_validator(mode="after")
    def _validate_language(self) -> CodeBlock:
        """代码块要求语言，算法块则禁止携带代码语言猜测结果。"""
        body = next(child for child in self.content if child.type in self._body_types)
        if self.sub_type == BlockType.CODE:
            if body.type != BlockType.CODE_BODY:
                raise ValueError("code block must contain code_body")
            if not isinstance(self.guess_lang, str) or not self.guess_lang.strip():
                raise ValueError("code block must contain a non-empty guess_lang")
        else:
            if body.type != BlockType.ALGORITHM_BODY:
                raise ValueError("algorithm block must contain algorithm_body")
            if self.guess_lang is not None:
                raise ValueError("algorithm block must not contain guess_lang")
        return self


ListBlock.model_rebuild()
IndexBlock.model_rebuild()


PageBlock: TypeAlias = Annotated[
    Union[
        TextBlock,
        RefTextBlock,
        DocTitleBlock,
        ParagraphTitleBlock,
        PageAuxTextBlock,
        PageFootnoteBlock,
        EquationBlock,
        ListBlock,
        IndexBlock,
        ImageBlock,
        TableBlock,
        ChartBlock,
        CodeBlock,
    ],
    Field(discriminator="type"),
]


Block: TypeAlias = Annotated[
    Union[
        TextBlock,
        RefTextBlock,
        DocTitleBlock,
        ParagraphTitleBlock,
        PageAuxTextBlock,
        PageFootnoteBlock,
        EquationBlock,
        ListBlock,
        IndexBlock,
        ImageBodyBlock,
        ImageAnnotationBlock,
        ImageBlock,
        TableBodyBlock,
        TableAnnotationBlock,
        TableBlock,
        ChartBodyBlock,
        ChartAnnotationBlock,
        ChartBlock,
        CodeBodyBlock,
        AlgorithmBodyBlock,
        CodeAnnotationBlock,
        CodeBlock,
    ],
    Field(discriminator="type"),
]

BLOCK_ADAPTER = TypeAdapter(Block)


def parse_block(value: Any) -> Block:
    """将字典或已有模型严格解析成对应的具体 Block 类型。"""
    return BLOCK_ADAPTER.validate_python(value)


def _iter_child_blocks(block: BlockBase) -> list[BlockBase]:
    """返回容器 block 的直接子块，叶子 block 返回空列表。"""
    content = getattr(block, "content", None)
    if not isinstance(content, list):
        return []
    return [child for child in content if isinstance(child, BlockBase)]


_RAW_INLINE_CONTENT_TYPES = {
    BlockType.TEXT,
    BlockType.REF_TEXT,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
    BlockType.PAGE_FOOTNOTE,
    BlockType.IMAGE_CAPTION,
    BlockType.IMAGE_FOOTNOTE,
    BlockType.TABLE_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.CHART_CAPTION,
    BlockType.CHART_FOOTNOTE,
    BlockType.CODE_CAPTION,
    BlockType.CODE_FOOTNOTE,
    RAW_ALGORITHM,
    RAW_CAPTION,
    RAW_FOOTNOTE,
    RAW_PHONETIC,
}


def _looks_like_raw_inline_span_list(content: list[Any]) -> bool:
    """区分 PDF 扁平 LIST/INDEX 的 Span 载荷与已经成树的文本子块。"""
    if not content:
        return False
    for item in content:
        if not isinstance(item, dict):
            return False
        span_type = item.get("type")
        span_content = item.get("content")
        if span_type in {"text", "equation_inline", "code_inline"} and isinstance(span_content, str):
            continue
        if span_type == "hyperlink" and isinstance(span_content, list) and isinstance(item.get("url"), str):
            continue
        return False
    return True


def _validate_raw_block_inline_content(block: dict[str, Any], *, location: str) -> None:
    """递归校验 raw block 的自然语言 content 已切换为 Span 列表。"""
    block_type = block.get("type")
    content = block.get("content")
    if block_type in _RAW_INLINE_CONTENT_TYPES:
        if not isinstance(content, list):
            raise ValueError(f"ModelJson inline content must be a span list: {location}, type={block_type}")
        try:
            parse_inline_spans(content)
        except ValueError as exc:
            raise ValueError(f"Invalid ModelJson inline spans: {location}, type={block_type}: {exc}") from exc
        return
    if block_type not in {BlockType.LIST, BlockType.INDEX} or not isinstance(content, list):
        return
    if _looks_like_raw_inline_span_list(content):
        try:
            parse_inline_spans(content)
        except ValueError as exc:
            raise ValueError(f"Invalid ModelJson inline spans: {location}, type={block_type}: {exc}") from exc
        return
    for child_index, child in enumerate(content):
        if isinstance(child, dict):
            _validate_raw_block_inline_content(child, location=f"{location}.content[{child_index}]")


class ModelJson(_StrictMiddleModel):
    """Analyze 返回的完整严格 Model JSON 对象。"""

    pages: list[list[dict[str, Any]]]
    page_index_map: list[int]
    file_suffix: FileSuffix
    effort: Literal["flash", "medium", "high", "xhigh"]
    parse_mode: Literal["txt", "ocr"]
    mineru_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_page_index_map(self) -> ModelJson:
        """校验显式抽页映射及每个 raw 文本块的 Span 契约。"""
        if self.page_index_map:
            if len(self.page_index_map) != len(self.pages):
                raise ValueError(f"page_index_map length mismatch: pages={len(self.pages)}, mapping={len(self.page_index_map)}")
            if any(page_idx < 0 for page_idx in self.page_index_map):
                raise ValueError("page_index_map values must be non-negative integers")
            if len(self.page_index_map) != len(set(self.page_index_map)):
                raise ValueError("page_index_map values must be unique")
            if any(current <= previous for previous, current in zip(self.page_index_map, self.page_index_map[1:])):
                raise ValueError("page_index_map values must preserve strictly increasing order")
        for page_index, page in enumerate(self.pages):
            for block_index, block in enumerate(page):
                if isinstance(block, dict):
                    _validate_raw_block_inline_content(block, location=f"pages[{page_index}][{block_index}]")
        return self

    @property
    def is_full_document(self) -> bool:
        """返回当前 Model JSON 是否表示整本文档解析。"""
        return not self.page_index_map

    @property
    def resolved_page_indices(self) -> list[int]:
        """返回显式抽页映射或整本文档的顺序页号副本。"""
        if self.is_full_document:
            return list(range(len(self.pages)))
        return list(self.page_index_map)


class PageInfo(_StrictMiddleModel):
    """一页的严格 Middle JSON 内容。"""

    page_idx: int = Field(ge=0)
    blocks: list[PageBlock] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_page_tree(self) -> PageInfo:
        """校验顶层 index 顺序，并禁止嵌套块携带跨块延续标记。"""
        indices: list[int] = []
        for block in self.blocks:
            if block.index is None:
                raise ValueError("top-level block index is required")
            indices.append(block.index)
        if len(indices) != len(set(indices)):
            raise ValueError("top-level block indices must be unique")
        if any(current <= previous for previous, current in zip(indices, indices[1:])):
            raise ValueError("top-level block indices must be strictly increasing")

        pending = [child for block in self.blocks for child in _iter_child_blocks(block)]
        while pending:
            child = pending.pop()
            if "continues_prev" in child.model_fields_set:
                raise ValueError("nested blocks must not contain continues_prev")
            pending.extend(_iter_child_blocks(child))
        return self


class MiddleJson(_StrictMiddleModel):
    """Analyze 返回的完整严格 Middle JSON 对象。"""

    pages: list[PageInfo]
    is_full_document: bool
    file_suffix: FileSuffix
    effort: Literal["flash", "medium", "high", "xhigh"]
    parse_mode: Literal["txt", "ocr"]
    mineru_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_document(self) -> MiddleJson:
        """校验页号唯一有序，并要求固定版式文档顶层 block 均具有 bbox。"""
        page_indices = [page.page_idx for page in self.pages]
        if len(page_indices) != len(set(page_indices)):
            raise ValueError("page_idx values must be unique")
        if any(current <= previous for previous, current in zip(page_indices, page_indices[1:])):
            raise ValueError("page_idx values must be strictly increasing")
        if self.file_suffix in {"pdf", "ofd"}:
            for page in self.pages:
                for block in page.blocks:
                    if block.bbox is None:
                        raise ValueError(
                            f"Fixed-layout top-level block requires bbox: "
                            f"file_suffix={self.file_suffix}, page_idx={page.page_idx}, index={block.index}"
                        )
        return self

    def export(
        self,
        output_dir: str | Path,
        *,
        json_name: str = "middle_json.json",
        overwrite: bool = False,
    ) -> MiddleJsonExportResult:
        """在副本上外置全部图片，并原子写出不含 base64 的 Middle JSON。"""
        return _export_middle_json(self, Path(output_dir), json_name=json_name, overwrite=overwrite)


@dataclass(frozen=True, slots=True)
class MiddleJsonExportResult:
    """Middle JSON 图片外置后的对象副本与实际文件路径。"""

    middle_json: MiddleJson
    json_path: Path
    image_paths: tuple[Path, ...]


def _iter_page_blocks(blocks: Sequence[BlockBase]) -> list[BlockBase]:
    """按深度优先顺序展开页面 block 树，供图片外置统一遍历。"""
    result: list[BlockBase] = []
    pending: list[BlockBase] = list(reversed(blocks))
    while pending:
        block = pending.pop()
        result.append(block)
        pending.extend(reversed(_iter_child_blocks(block)))
    return result


def _register_export_file(files: dict[str, bytes], relative_path: str, payload: bytes) -> None:
    """登记待写文件，并在同名内容冲突时立即终止导出。"""
    existing = files.get(relative_path)
    if existing is not None and existing != payload:
        raise ValueError(f"Conflicting image payload for path: {relative_path}")
    files[relative_path] = payload


def _prepare_export_copy(middle_json: MiddleJson) -> tuple[MiddleJson, dict[str, bytes]]:
    """复制对象、解析直接及 HTML 图片，并回填副本中的相对路径。"""
    from .utils.image_payload import (
        INLINE_IMAGE_DATA_URI_RE,
        parse_image_data_uri_strict,
    )

    exported = middle_json.model_copy(deep=True)
    image_files: dict[str, bytes] = {}
    for page in exported.pages:
        for block in _iter_page_blocks(page.blocks):
            data_uri = getattr(block, "image_base64", None)
            if data_uri is not None:
                if block.index is None:
                    raise ValueError(f"Image carrier requires index: page_idx={page.page_idx}, type={block.type}")
                image_bytes, extension = parse_image_data_uri_strict(data_uri)
                relative_path = f"images/page_{page.page_idx}_{block.type}_{block.index}.{extension}"
                _register_export_file(image_files, relative_path, image_bytes)
                block.image_path = relative_path  # type: ignore[attr-defined]
                block.image_base64 = None  # type: ignore[attr-defined]

            content = getattr(block, "content", None)
            if not isinstance(content, str) or "data:image/" not in content:
                continue
            if block.index is None:
                raise ValueError(f"HTML image carrier requires index: page_idx={page.page_idx}, type={block.type}")
            ordinal = 0

            def _replace_data_uri(match: Any) -> str:
                """将当前 HTML data URI 登记为 sidecar，并返回确定性相对路径。"""
                nonlocal ordinal
                ordinal += 1
                image_bytes, extension = parse_image_data_uri_strict(match.group(0))
                relative_path = f"images/page_{page.page_idx}_{block.type}_{block.index}_{ordinal}.{extension}"
                _register_export_file(image_files, relative_path, image_bytes)
                return relative_path

            block.content = INLINE_IMAGE_DATA_URI_RE.sub(_replace_data_uri, content)  # type: ignore[assignment]
    return exported, image_files


def _resolve_export_target(output_root: Path, relative_path: str) -> Path:
    """校验导出相对路径并确保解析后的目标仍位于文档输出目录内。"""
    from .utils.image_payload import validate_image_sidecar_path

    safe_path = validate_image_sidecar_path(relative_path)
    if output_root.is_symlink():
        raise ValueError(f"Export directory must not be a symlink: {output_root}")
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"Export directory must be a directory: {output_root}")
    root = output_root.resolve()
    target = root / safe_path
    current_parent = root
    for path_part in Path(safe_path).parts[:-1]:
        current_parent /= path_part
        if current_parent.is_symlink():
            raise ValueError(f"Export path contains a symlink: {relative_path}")
        if current_parent.exists() and not current_parent.is_dir():
            raise ValueError(f"Export path parent is not a directory: {relative_path}")
    try:
        target.parent.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Export path escapes output directory: {relative_path}") from exc
    return target


def _commit_export_files(files: dict[Path, bytes], *, overwrite: bool) -> None:
    """预检冲突后以临时文件提交，并在提交失败时恢复已有文件。"""
    pending: dict[Path, bytes] = {}
    originals: dict[Path, bytes | None] = {}
    for target, payload in files.items():
        if target.is_symlink():
            raise ValueError(f"Export target must not be a symlink: {target}")
        if target.exists():
            if not target.is_file():
                raise ValueError(f"Export target is not a regular file: {target}")
            current = target.read_bytes()
            if current == payload:
                continue
            if not overwrite:
                raise FileExistsError(f"Export target already exists with different content: {target}")
            originals[target] = current
        else:
            originals[target] = None
        pending[target] = payload

    temp_paths: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for target, payload in pending.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                mode="wb",
                prefix=".mineru-export-",
                dir=target.parent,
                delete=False,
            ) as temp_file:
                temp_file.write(payload)
                temp_paths[target] = Path(temp_file.name)
        for target, temp_path in temp_paths.items():
            os.replace(temp_path, target)
            committed.append(target)
    except Exception:
        for temp_path in temp_paths.values():
            temp_path.unlink(missing_ok=True)
        for target in reversed(committed):
            original = originals[target]
            if original is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(original)
        raise


def _validate_export_path_relationships(relative_paths: list[str]) -> None:
    """拒绝任一导出文件占用另一文件的父目录，避免提交阶段才产生冲突。"""
    path_parts = {relative_path: Path(relative_path).parts for relative_path in relative_paths}
    for relative_path, parts in path_parts.items():
        for other_path, other_parts in path_parts.items():
            if relative_path == other_path or len(parts) >= len(other_parts):
                continue
            if other_parts[: len(parts)] == parts:
                raise ValueError(f"Export file path conflicts with a required directory: {relative_path} -> {other_path}")


def _export_middle_json(
    middle_json: MiddleJson,
    output_dir: Path,
    *,
    json_name: str,
    overwrite: bool,
) -> MiddleJsonExportResult:
    """构造完整导出事务，保证 JSON 与图片使用同一份规范化对象副本。"""
    from .utils.image_payload import validate_image_sidecar_path

    exported, image_files = _prepare_export_copy(middle_json)
    json_text = exported.to_json(
        exclude_block_fields={"image_base64"},
    )
    if "data:image/" in json_text.lower():
        raise ValueError("Exported Middle JSON still contains an inline image data URI")
    json_bytes = json_text.encode("utf-8")
    safe_json_name = validate_image_sidecar_path(json_name)
    relative_files = dict(image_files)
    if safe_json_name in relative_files:
        raise ValueError(f"JSON path conflicts with an exported image: {safe_json_name}")
    relative_files[safe_json_name] = json_bytes
    _validate_export_path_relationships(list(relative_files))
    absolute_files = {
        _resolve_export_target(output_dir, relative_path): payload for relative_path, payload in relative_files.items()
    }
    _commit_export_files(absolute_files, overwrite=overwrite)
    json_path = _resolve_export_target(output_dir, safe_json_name)
    image_paths = tuple(_resolve_export_target(output_dir, relative_path) for relative_path in sorted(image_files))
    return MiddleJsonExportResult(
        middle_json=exported,
        json_path=json_path,
        image_paths=image_paths,
    )
