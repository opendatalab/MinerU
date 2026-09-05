# Copyright (c) Opendatalab. All rights reserved.
"""MinerU 产品档位与 DocGale 共享文档类型的稳定入口。"""

from __future__ import annotations
from typing import Iterable, Literal
from docgale.schema import (
    RawBlockType,
    RAW_ALGORITHM,
    RAW_CAPTION,
    RAW_FOOTNOTE,
    RAW_FORMULA_NUMBER,
    RAW_PHONETIC,
    RAW_ONLY_BLOCK_TYPES,
    FileSuffix,
    FILE_SUFFIXES,
    BlockType,
    ContentType,
    ContentTypeV2,
    BlockTypes,
    PageBlockTypes,
    BLOCK_TYPES,
    PAGE_BLOCK_TYPES,
    PAGE_AUXILIARY_BLOCK_TYPES,
    MERGE_TRANSPARENT_BLOCK_TYPES,
    VISUAL_RELATION_IGNORED_TYPES,
    VISUAL_MAIN_TYPES,
    VISUAL_TYPE_MAPPING,
    BBox,
    IntBBox,
    InlineStyle,
    INLINE_STYLE_ORDER,
    TextSpan,
    EquationInlineSpan,
    CodeInlineSpan,
    NonLinkInlineSpan,
    HyperlinkSpan,
    InlineSpan,
    INLINE_SPAN_ADAPTER,
    INLINE_SPAN_LIST_ADAPTER,
    parse_inline_span,
    parse_inline_spans,
    BlockBase,
    StringContentBlock,
    InlineContentBlock,
    ContinuableTextBlockBase,
    TextBlock,
    RefTextBlock,
    TitleBlockBase,
    DocTitleBlock,
    ParagraphTitleBlock,
    PageAuxTextBlock,
    PageFootnoteBlock,
    ImagePayloadBlock,
    ImagePayloadContentBlock,
    EquationBlock,
    ImageBodyBlock,
    TableBodyBlock,
    ChartBodyBlock,
    CodeBodyBlock,
    AlgorithmBodyBlock,
    ImageAnnotationBlock,
    TableAnnotationBlock,
    ChartAnnotationBlock,
    CodeAnnotationBlock,
    ListChildBlock,
    ListBlock,
    IndexChildBlock,
    IndexBlock,
    ImageChildBlock,
    ImageBlock,
    TableChildBlock,
    TableBlock,
    ChartChildBlock,
    ChartBlock,
    CodeChildBlock,
    CodeBlock,
    PageBlock,
    Block,
    BLOCK_ADAPTER,
    parse_block,
    Producer,
    DocumentModel,
    ModelJson,
    PageInfo,
    MiddleJson,
)

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


__all__ = [
    "RawBlockType",
    "RAW_ALGORITHM",
    "RAW_CAPTION",
    "RAW_FOOTNOTE",
    "RAW_FORMULA_NUMBER",
    "RAW_PHONETIC",
    "RAW_ONLY_BLOCK_TYPES",
    "FileSuffix",
    "FILE_SUFFIXES",
    "BlockType",
    "ContentType",
    "ContentTypeV2",
    "BlockTypes",
    "PageBlockTypes",
    "BLOCK_TYPES",
    "PAGE_BLOCK_TYPES",
    "PAGE_AUXILIARY_BLOCK_TYPES",
    "MERGE_TRANSPARENT_BLOCK_TYPES",
    "VISUAL_RELATION_IGNORED_TYPES",
    "VISUAL_MAIN_TYPES",
    "VISUAL_TYPE_MAPPING",
    "BBox",
    "IntBBox",
    "InlineStyle",
    "INLINE_STYLE_ORDER",
    "TextSpan",
    "EquationInlineSpan",
    "CodeInlineSpan",
    "NonLinkInlineSpan",
    "HyperlinkSpan",
    "InlineSpan",
    "INLINE_SPAN_ADAPTER",
    "INLINE_SPAN_LIST_ADAPTER",
    "parse_inline_span",
    "parse_inline_spans",
    "BlockBase",
    "StringContentBlock",
    "InlineContentBlock",
    "ContinuableTextBlockBase",
    "TextBlock",
    "RefTextBlock",
    "TitleBlockBase",
    "DocTitleBlock",
    "ParagraphTitleBlock",
    "PageAuxTextBlock",
    "PageFootnoteBlock",
    "ImagePayloadBlock",
    "ImagePayloadContentBlock",
    "EquationBlock",
    "ImageBodyBlock",
    "TableBodyBlock",
    "ChartBodyBlock",
    "CodeBodyBlock",
    "AlgorithmBodyBlock",
    "ImageAnnotationBlock",
    "TableAnnotationBlock",
    "ChartAnnotationBlock",
    "CodeAnnotationBlock",
    "ListChildBlock",
    "ListBlock",
    "IndexChildBlock",
    "IndexBlock",
    "ImageChildBlock",
    "ImageBlock",
    "TableChildBlock",
    "TableBlock",
    "ChartChildBlock",
    "ChartBlock",
    "CodeChildBlock",
    "CodeBlock",
    "PageBlock",
    "Block",
    "BLOCK_ADAPTER",
    "parse_block",
    "Producer",
    "DocumentModel",
    "ModelJson",
    "PageInfo",
    "MiddleJson",
    "Tier",
    "TIERS",
    "TIER_ORDER",
    "ServerTier",
    "SERVER_TIERS",
    "TIERS_BY_SERVER_TIER",
    "DeploymentTier",
    "DEPLOYMENT_TIERS",
    "DEFAULT_QUALITY_TIER_SELECTION_ORDER",
    "QUALITY_TIERS",
    "CACHED_TIER_SELECTION_ORDER",
    "PARSING_RULE_TIER_SELECTION_ORDER",
    "validate_tier",
    "select_default_quality_tier",
    "select_highest_cached_tier",
    "select_parsing_rule_tier",
]
