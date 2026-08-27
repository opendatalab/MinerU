# Copyright (c) Opendatalab. All rights reserved.
"""校验并解码版本化 MinerU HTML 机器语义契约。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import html
from typing import Literal, TypeAlias

from lxml import etree  # type: ignore[reportMissingImports]

from ....types import PAGE_BLOCK_TYPES, RAW_ALGORITHM, BlockType, VISUAL_TYPE_MAPPING
from .._shared.markup import MarkupProjector, MarkupStylesheet, extract_formula
from .._shared.markup.projector import BLOCK_TAGS, local_name
from .resources import HtmlResourceContext


MineruHtmlRenderMode: TypeAlias = Literal["default", "full"]

_MINERU_HTML_VERSION = "1"
_VISUAL_TYPES = frozenset(VISUAL_TYPE_MAPPING)
_SIMPLE_TEXT_TYPES = frozenset(
    {
        BlockType.TEXT,
        BlockType.REF_TEXT,
        BlockType.DOC_TITLE,
        BlockType.PARAGRAPH_TITLE,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.ASIDE_TEXT,
        BlockType.PAGE_FOOTNOTE,
    }
)
_PAGE_AUXILIARY_TYPES = frozenset(
    {
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.ASIDE_TEXT,
    }
)
_LIST_LEAF_TYPES = frozenset({BlockType.TEXT, BlockType.REF_TEXT})
_INDEX_LEAF_TYPES = frozenset({BlockType.TEXT, BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE})
_OWNED_VISUAL_IMAGE_TOKENS = frozenset(
    {"mineru-chart-image", "mineru-flowchart-fallback", "mineru-image", "mineru-table-image"}
)


@dataclass(frozen=True, slots=True)
class MineruHtmlBlockSpec:
    """保存一个已完整校验但尚未加载资源的顶层机器 block。"""

    wrapper: etree._Element
    content_root: etree._Element
    block_type: BlockType
    page_idx: int
    block_index: int | None


@dataclass(frozen=True, slots=True)
class MineruHtmlWirePlan:
    """保存一次版本化 MinerU HTML 的只读验证结果。"""

    root: etree._Element
    render_mode: MineruHtmlRenderMode
    blocks: tuple[MineruHtmlBlockSpec, ...]


@dataclass(frozen=True, slots=True)
class MineruHtmlWireInspection:
    """区分无 marker、有效 marker 和需通用回退的非法 marker。"""

    plan: MineruHtmlWirePlan | None
    fallback_reason: str | None = None


class _WireValidationError(ValueError):
    """携带不包含正文或属性值的稳定 marker 失败原因。"""

    def __init__(self, reason: str) -> None:
        """保存固定诊断码。"""
        super().__init__(reason)
        self.reason = reason


class _ExactAnchorResolver:
    """把 renderer 生成的 DOM id 恢复为 wire 中保存的原始 anchor。"""

    def __init__(self, plan: MineruHtmlWirePlan) -> None:
        """预扫描标题和页面脚注的 id、标签及原始 anchor。"""
        self._targets: dict[str, str] = {}
        self._heading_anchors: dict[etree._Element, str] = {}
        self._heading_labels: dict[str, str] = {}
        self._note_anchors: dict[etree._Element, str] = {}
        for spec in plan.blocks:
            anchor = (spec.wrapper.get("data-anchor") or "").strip()
            if not anchor:
                continue
            self._targets.setdefault(anchor, anchor)
            for candidate in [spec.content_root, *spec.content_root.iterdescendants()]:
                identity = (candidate.get("id") or "").strip()
                if identity:
                    self._targets.setdefault(identity, anchor)
            if spec.block_type in {BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE}:
                self._heading_anchors[spec.content_root] = anchor
                self._heading_labels[anchor] = " ".join(spec.content_root.itertext()).strip()
            elif spec.block_type == BlockType.PAGE_FOOTNOTE:
                self._note_anchors[spec.content_root] = anchor

    def resolve_fragment(self, fragment: str) -> str | None:
        """把 renderer id 或原始 anchor 统一还原为内部 fragment。"""
        identity = fragment.removeprefix("#").strip()
        return f"#{anchor}" if (anchor := self._targets.get(identity)) else None

    def heading_anchor(self, heading: etree._Element) -> str | None:
        """返回精确标题节点的原始 anchor。"""
        return self._heading_anchors.get(heading)

    def heading_label(self, anchor: str) -> str | None:
        """返回原始 anchor 对应的标题可见文本。"""
        return self._heading_labels.get(anchor)

    def note_anchor(self, note: etree._Element) -> str | None:
        """返回精确页面脚注节点的原始 anchor。"""
        return self._note_anchors.get(note)


def inspect_mineru_html_wire(body: etree._Element) -> MineruHtmlWireInspection:
    """发现并事务式校验 MinerU HTML marker，失败只返回固定诊断码。"""
    roots = [
        element
        for element in body.iter()
        if isinstance(element.tag, str) and element.get("data-mineru-html-version") is not None
    ]
    if not roots:
        return MineruHtmlWireInspection(None)
    if len(roots) != 1:
        return MineruHtmlWireInspection(None, "multiple_version_roots")
    root = roots[0]
    if (root.get("data-mineru-html-version") or "").strip() != _MINERU_HTML_VERSION:
        return MineruHtmlWireInspection(None, "unsupported_version")
    try:
        return MineruHtmlWireInspection(_validate_wire_root(root))
    except _WireValidationError as exc:
        return MineruHtmlWireInspection(None, exc.reason)


def materialize_mineru_html_wire(
    plan: MineruHtmlWirePlan,
    resources: HtmlResourceContext,
) -> list[dict[str, object]]:
    """在结构验证成功后一次性加载资源并物化单页 raw blocks。"""
    resources.bind_anchors(_ExactAnchorResolver(plan))
    stylesheet = MarkupStylesheet()
    projector = MarkupProjector(plan.root, resources, stylesheet, single_document_title=True)
    blocks: list[dict[str, object]] = []
    for spec in plan.blocks:
        if spec.block_type in _SIMPLE_TEXT_TYPES:
            blocks.append(_materialize_text_block(spec, projector))
        elif spec.block_type == BlockType.EQUATION:
            blocks.append(_materialize_equation(spec, resources))
        elif spec.block_type == BlockType.LIST:
            blocks.append(_materialize_list(spec.content_root, projector))
        elif spec.block_type == BlockType.INDEX:
            blocks.append(_materialize_index(spec.content_root, projector))
        elif spec.block_type in _VISUAL_TYPES:
            blocks.extend(_materialize_visual(spec, resources, stylesheet, projector))
        else:  # pragma: no cover - 根验证已经穷尽公开 PageBlock 类型。
            raise AssertionError(f"unhandled wire block type: {spec.block_type}")
    return blocks


def _validate_wire_root(root: etree._Element) -> MineruHtmlWirePlan:
    """校验根版本、渲染模式、页面容器与顶层 block 完整性。"""
    if local_name(root) != "article" or "mineru-document" not in _class_tokens(root):
        raise _WireValidationError("invalid_root")
    _validate_element_only_content(root)
    mode = (root.get("data-render-mode") or "").strip()
    if mode not in {"default", "full"}:
        raise _WireValidationError("invalid_render_mode")
    wrappers: list[tuple[etree._Element, int | None]] = []
    if mode == "default":
        for child in _element_children(root):
            if local_name(child) != "div" or "mineru-block" not in _class_tokens(child):
                raise _WireValidationError("invalid_default_child")
            wrappers.append((child, None))
    else:
        for child in _element_children(root):
            if local_name(child) == "hr" and "mineru-page-break" in _class_tokens(child):
                continue
            if local_name(child) != "section" or "mineru-page" not in _class_tokens(child):
                raise _WireValidationError("invalid_full_child")
            _validate_element_only_content(child)
            section_page_idx = _non_negative_integer(child, "data-page-idx", required=True)
            for wrapper in _element_children(child):
                if local_name(wrapper) != "div" or "mineru-block" not in _class_tokens(wrapper):
                    raise _WireValidationError("invalid_page_child")
                wrappers.append((wrapper, section_page_idx))

    nested_wrappers = [
        element
        for element in root.iterdescendants()
        if isinstance(element.tag, str) and "mineru-block" in _class_tokens(element)
    ]
    if len(nested_wrappers) != len(wrappers):
        raise _WireValidationError("nested_block_wrapper")

    specs = tuple(_validate_top_block(wrapper, section_page_idx) for wrapper, section_page_idx in wrappers)
    return MineruHtmlWirePlan(root, mode, specs)  # type: ignore[arg-type]


def _validate_top_block(wrapper: etree._Element, section_page_idx: int | None) -> MineruHtmlBlockSpec:
    """校验单个顶层 wrapper 的公开类型、元数据与内部机器树。"""
    _validate_element_only_content(wrapper)
    block_type = _page_block_type(wrapper)
    page_idx = _non_negative_integer(wrapper, "data-page-idx", required=True)
    if section_page_idx is not None and page_idx != section_page_idx:
        raise _WireValidationError("page_index_mismatch")
    block_index = _non_negative_integer(wrapper, "data-block-index", required=False)
    children = _element_children(wrapper)
    if len(children) != 1:
        raise _WireValidationError("invalid_block_content_count")
    content_root = children[0]
    _validate_top_metadata(wrapper, block_type)
    if block_type in _SIMPLE_TEXT_TYPES:
        _validate_simple_content(content_root, block_type)
    elif block_type == BlockType.EQUATION:
        _validate_equation_content(content_root)
    elif block_type == BlockType.LIST:
        _validate_list_container(content_root, top_wrapper=wrapper)
    elif block_type == BlockType.INDEX:
        _validate_index_root(content_root, wrapper)
    elif block_type in _VISUAL_TYPES:
        _validate_visual_content(content_root, wrapper, block_type)
    else:
        raise _WireValidationError("unsupported_block_type")
    return MineruHtmlBlockSpec(wrapper, content_root, block_type, page_idx, block_index)


def _validate_top_metadata(wrapper: etree._Element, block_type: BlockType) -> None:
    """校验 subtype、语言、anchor 和 level 只出现在合法父类型上。"""
    sub_type = (wrapper.get("data-block-sub-type") or "").strip()
    guess_lang = (wrapper.get("data-guess-lang") or "").strip()
    anchor = (wrapper.get("data-anchor") or "").strip()
    level = _optional_integer(wrapper, "data-level")

    if block_type == BlockType.CODE:
        if sub_type not in {BlockType.CODE, RAW_ALGORITHM}:
            raise _WireValidationError("invalid_code_sub_type")
        if sub_type == BlockType.CODE and not guess_lang:
            raise _WireValidationError("missing_code_language")
        if sub_type == RAW_ALGORITHM and guess_lang:
            raise _WireValidationError("algorithm_language_mismatch")
    elif block_type == BlockType.LIST:
        if sub_type and sub_type not in _LIST_LEAF_TYPES:
            raise _WireValidationError("invalid_list_sub_type")
        if guess_lang:
            raise _WireValidationError("unexpected_guess_lang")
    elif block_type in {BlockType.IMAGE, BlockType.CHART}:
        if guess_lang:
            raise _WireValidationError("unexpected_guess_lang")
    elif sub_type or guess_lang:
        raise _WireValidationError("unexpected_block_metadata")

    if block_type == BlockType.DOC_TITLE:
        if level != 1:
            raise _WireValidationError("invalid_title_level")
    elif block_type == BlockType.PARAGRAPH_TITLE:
        if level is None or not 2 <= level <= 6:
            raise _WireValidationError("invalid_title_level")
    elif level is not None:
        raise _WireValidationError("unexpected_title_level")

    if anchor and block_type not in {BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE, BlockType.PAGE_FOOTNOTE}:
        raise _WireValidationError("unexpected_anchor")


def _validate_simple_content(content_root: etree._Element, block_type: BlockType) -> None:
    """校验文本、标题、页面辅助块只含允许的内部公式 marker。"""
    expected_tags: dict[BlockType, frozenset[str]] = {
        BlockType.TEXT: frozenset({"p"}),
        BlockType.REF_TEXT: frozenset({"p"}),
        BlockType.DOC_TITLE: frozenset({"h1"}),
        BlockType.PARAGRAPH_TITLE: frozenset({"h2", "h3", "h4", "h5", "h6"}),
        BlockType.PAGE_FOOTNOTE: frozenset({"div"}),
        **{block_type_value: frozenset({"div"}) for block_type_value in _PAGE_AUXILIARY_TYPES},
    }
    if local_name(content_root) not in expected_tags[block_type]:
        raise _WireValidationError("invalid_simple_shape")
    for marker in _descendant_markers(content_root):
        marker_type = (marker.get("data-block-type") or "").strip()
        if marker is content_root and marker_type == block_type:
            continue
        if marker_type != BlockType.EQUATION:
            raise _WireValidationError("invalid_nested_marker")
        _validate_formula_carrier(marker, expected_display="inline")
    _validate_inline_content_shape(content_root)


def _validate_equation_content(content_root: etree._Element) -> None:
    """校验行间公式为规范 carrier，或为无 LaTeX 的安全图片回退。"""
    if local_name(content_root) == "math":
        _validate_formula_carrier(content_root, expected_display="block")
        return
    if local_name(content_root) != "img" or "mineru-equation-image" not in _class_tokens(content_root):
        raise _WireValidationError("invalid_equation_shape")


def _validate_formula_carrier(
    element: etree._Element,
    *,
    expected_display: Literal["inline", "block"] | None,
) -> None:
    """校验公式 marker 含非空裸 LaTeX 和合法显示方式。"""
    if local_name(element) != "math" or (element.get("data-block-type") or "").strip() != BlockType.EQUATION:
        raise _WireValidationError("invalid_formula_marker")
    display = (element.get("data-formula-display") or "").strip()
    if display not in {"inline", "block"} or (expected_display is not None and display != expected_display):
        raise _WireValidationError("invalid_formula_display")
    formula = extract_formula(element)
    if formula is None or not formula.latex:
        raise _WireValidationError("missing_formula_latex")


def _validate_visual_content(content_root: etree._Element, wrapper: etree._Element, parent_type: BlockType) -> None:
    """校验视觉父块只有匹配 body，并拒绝跨父类型 annotation。"""
    if local_name(content_root) != "figure":
        raise _WireValidationError("invalid_visual_root")
    _validate_element_only_content(content_root)
    mapping = VISUAL_TYPE_MAPPING[parent_type]
    allowed_types = frozenset(mapping.values())
    children = _element_children(content_root)
    child_types = [(child.get("data-block-type") or "").strip() for child in children]
    if child_types.count(mapping["body"]) != 1:
        raise _WireValidationError("invalid_visual_body_count")
    if any(child_type not in allowed_types for child_type in child_types):
        raise _WireValidationError("annotation_parent_mismatch")
    parent_index = _non_negative_integer(wrapper, "data-block-index", required=False)
    for child, child_type in zip(children, child_types, strict=True):
        child_index = _non_negative_integer(child, "data-block-index", required=False)
        if child_type == mapping["body"]:
            if local_name(child) != "div":
                raise _WireValidationError("invalid_visual_body_shape")
            if parent_index is not None and child_index != parent_index:
                raise _WireValidationError("visual_body_index_mismatch")
            if parent_type == BlockType.CODE and (wrapper.get("data-block-sub-type") or "").strip() == BlockType.CODE:
                _validate_code_body_shape(child)
            for marker in _descendant_markers(child):
                if marker is child:
                    continue
                if (marker.get("data-block-type") or "").strip() != BlockType.EQUATION:
                    raise _WireValidationError("invalid_visual_nested_marker")
                _validate_formula_carrier(marker, expected_display="inline")
        else:
            if local_name(child) != "p":
                raise _WireValidationError("invalid_annotation_shape")
            for marker in _descendant_markers(child):
                if marker is child:
                    continue
                if (marker.get("data-block-type") or "").strip() != BlockType.EQUATION:
                    raise _WireValidationError("invalid_annotation_nested_marker")
                _validate_formula_carrier(marker, expected_display="inline")
            _validate_inline_content_shape(child)


def _validate_code_body_shape(body: etree._Element) -> None:
    """校验普通代码 body 只含规范 pre/code，避免精确物化静默丢弃编辑内容。"""
    _validate_element_only_content(body)
    children = _element_children(body)
    if len(children) != 1 or local_name(children[0]) != "pre":
        raise _WireValidationError("invalid_code_body_shape")
    pre = children[0]
    _validate_element_only_content(pre)
    code_children = _element_children(pre)
    if len(code_children) != 1 or local_name(code_children[0]) != "code":
        raise _WireValidationError("invalid_code_body_shape")
    if _element_children(code_children[0]):
        raise _WireValidationError("invalid_code_body_shape")


def _validate_list_container(container: etree._Element, *, top_wrapper: etree._Element | None = None) -> None:
    """递归校验列表容器、叶子类型和嵌套列表 marker。"""
    if local_name(container) not in {"ol", "ul"} or (container.get("data-block-type") or "").strip() != BlockType.LIST:
        raise _WireValidationError("invalid_list_root")
    _validate_element_only_content(container)
    _non_negative_integer(container, "data-block-index", required=False)
    sub_type = (container.get("data-block-sub-type") or "").strip()
    if sub_type and sub_type not in _LIST_LEAF_TYPES:
        raise _WireValidationError("invalid_nested_list_sub_type")
    if top_wrapper is not None:
        wrapper_sub_type = (top_wrapper.get("data-block-sub-type") or "").strip()
        if sub_type != wrapper_sub_type:
            raise _WireValidationError("list_sub_type_mismatch")
    for item in _element_children(container):
        if local_name(item) != "li":
            raise _WireValidationError("invalid_list_item")
        item_type = (item.get("data-block-type") or "").strip()
        nested_lists = [child for child in _element_children(item) if local_name(child) in {"ol", "ul"}]
        if item_type:
            if item_type not in _LIST_LEAF_TYPES:
                raise _WireValidationError("invalid_list_leaf_type")
            _non_negative_integer(item, "data-block-index", required=False)
            for marker in _descendant_markers(item):
                if marker is item or _is_within_any(marker, nested_lists):
                    continue
                marker_type = (marker.get("data-block-type") or "").strip()
                if marker_type == BlockType.LIST and local_name(marker) in {"ol", "ul"}:
                    continue
                if marker_type != BlockType.EQUATION:
                    raise _WireValidationError("invalid_list_nested_marker")
                _validate_formula_carrier(marker, expected_display="inline")
            _validate_inline_content_shape(item, excluded_roots=nested_lists)
        elif not nested_lists:
            raise _WireValidationError("markerless_list_leaf")
        for nested in nested_lists:
            _validate_list_container(nested)


def _validate_index_root(content_root: etree._Element, wrapper: etree._Element) -> None:
    """校验目录根 nav 与递归目录叶子元数据。"""
    if local_name(content_root) != "nav" or (content_root.get("data-block-type") or "").strip() != BlockType.INDEX:
        raise _WireValidationError("invalid_index_root")
    _validate_element_only_content(content_root)
    root_index = _non_negative_integer(content_root, "data-block-index", required=False)
    wrapper_index = _non_negative_integer(wrapper, "data-block-index", required=False)
    if wrapper_index is not None and root_index != wrapper_index:
        raise _WireValidationError("index_root_mismatch")
    lists = _element_children(content_root)
    if len(lists) != 1 or local_name(lists[0]) != "ul":
        raise _WireValidationError("invalid_index_list")
    _validate_index_list(lists[0], nested=False)


def _validate_index_list(container: etree._Element, *, nested: bool) -> None:
    """递归校验目录 li 类型、anchor、level 和嵌套 IndexBlock。"""
    _validate_element_only_content(container)
    if nested and (container.get("data-block-type") or "").strip() != BlockType.INDEX:
        raise _WireValidationError("missing_nested_index_marker")
    if nested:
        _non_negative_integer(container, "data-block-index", required=False)
    for item in _element_children(container):
        if local_name(item) != "li":
            raise _WireValidationError("invalid_index_item")
        item_type_value = (item.get("data-block-type") or "").strip()
        nested_lists = [child for child in _element_children(item) if local_name(child) == "ul"]
        if item_type_value:
            try:
                item_type = BlockType(item_type_value)
            except ValueError as exc:
                raise _WireValidationError("invalid_index_leaf_type") from exc
            if item_type not in _INDEX_LEAF_TYPES:
                raise _WireValidationError("invalid_index_leaf_type")
            _non_negative_integer(item, "data-block-index", required=False)
            _validate_index_leaf_metadata(item, item_type)
            for marker in _descendant_markers(item):
                if marker is item or _is_within_any(marker, nested_lists):
                    continue
                marker_type = (marker.get("data-block-type") or "").strip()
                if marker_type == BlockType.INDEX and local_name(marker) == "ul":
                    continue
                if marker_type != BlockType.EQUATION:
                    raise _WireValidationError("invalid_index_nested_marker")
                _validate_formula_carrier(marker, expected_display="inline")
            _validate_inline_content_shape(item, excluded_roots=nested_lists)
        elif not nested_lists:
            raise _WireValidationError("markerless_index_leaf")
        for child in nested_lists:
            _validate_index_list(child, nested=True)


def _validate_index_leaf_metadata(item: etree._Element, item_type: BlockType) -> None:
    """校验目录标题叶子的 anchor 与 level，不允许普通文本伪装标题。"""
    anchor = (item.get("data-anchor") or "").strip()
    level = _optional_integer(item, "data-level")
    if item_type == BlockType.TEXT:
        if anchor or level is not None:
            raise _WireValidationError("unexpected_index_text_metadata")
        return
    if not anchor:
        raise _WireValidationError("missing_index_anchor")
    if item_type == BlockType.DOC_TITLE and level != 1:
        raise _WireValidationError("invalid_index_level")
    if item_type == BlockType.PARAGRAPH_TITLE and (level is None or not 2 <= level <= 6):
        raise _WireValidationError("invalid_index_level")


def _materialize_text_block(spec: MineruHtmlBlockSpec, projector: MarkupProjector) -> dict[str, object]:
    """恢复文本、标题、页面辅助块和页面脚注 raw 字段。"""
    block: dict[str, object] = {
        "type": spec.block_type,
        "content": _project_inline_content(projector, spec.content_root),
    }
    if spec.block_type in {BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE}:
        block["level"] = int(spec.wrapper.get("data-level") or 1)
    if spec.block_type in {BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE, BlockType.PAGE_FOOTNOTE}:
        anchor = (spec.wrapper.get("data-anchor") or "").strip()
        if anchor:
            block["anchor"] = anchor
    return block


def _materialize_equation(spec: MineruHtmlBlockSpec, resources: HtmlResourceContext) -> dict[str, object]:
    """恢复行间公式裸 LaTeX，图片公式只保留经安全解析的载荷。"""
    formula = extract_formula(spec.content_root)
    block: dict[str, object] = {
        "type": BlockType.EQUATION,
        "content": formula.latex if formula is not None else "",
    }
    if formula is None:
        block.update(_resolved_image_payload(spec.content_root, resources))
    return block


def _materialize_visual(
    spec: MineruHtmlBlockSpec,
    resources: HtmlResourceContext,
    stylesheet: MarkupStylesheet,
    projector: MarkupProjector,
) -> list[dict[str, object]]:
    """按原始子节点顺序把 visual body、caption 和 footnote 展平成 raw blocks。"""
    mapping = VISUAL_TYPE_MAPPING[spec.block_type]
    output: list[dict[str, object]] = []
    for child in _element_children(spec.content_root):
        child_type = (child.get("data-block-type") or "").strip()
        if child_type == mapping["body"]:
            output.append(_materialize_visual_body(spec, child, resources, stylesheet))
            continue
        output.append(
            {
                "type": BlockType(child_type),
                "content": _project_inline_content(projector, child),
            }
        )
    return output


def _materialize_visual_body(
    spec: MineruHtmlBlockSpec,
    body: etree._Element,
    resources: HtmlResourceContext,
    stylesheet: MarkupStylesheet,
) -> dict[str, object]:
    """按父 visual 类型恢复主体内容、图片载荷和 subtype/lang 元数据。"""
    sub_type = (spec.wrapper.get("data-block-sub-type") or "").strip()
    if spec.block_type == BlockType.CODE:
        return _materialize_code_body(spec, body, resources, stylesheet)
    if spec.block_type == BlockType.TABLE:
        block = _materialize_table_body(body, resources, stylesheet)
    else:
        content = _flowchart_content(body) if spec.block_type == BlockType.IMAGE and sub_type == "flowchart" else None
        image_classes = (
            frozenset({"mineru-image", "mineru-flowchart-fallback"})
            if spec.block_type == BlockType.IMAGE
            else frozenset({"mineru-chart-image"})
        )
        block = {
            "type": spec.block_type,
            "content": content if content is not None else _project_visual_body_content(body, resources, stylesheet),
            **_resolved_image_payload(body, resources, allowed_classes=image_classes),
        }
    if sub_type:
        block["sub_type"] = sub_type
    return block


def _materialize_code_body(
    spec: MineruHtmlBlockSpec,
    body: etree._Element,
    resources: HtmlResourceContext,
    stylesheet: MarkupStylesheet,
) -> dict[str, object]:
    """恢复普通代码或 algorithm，并保持语言只属于普通代码。"""
    sub_type = (spec.wrapper.get("data-block-sub-type") or "").strip()
    if sub_type == BlockType.CODE:
        code = next(
            (element for element in body.iterdescendants() if isinstance(element.tag, str) and local_name(element) == "code"),
            None,
        )
        content = "".join(code.itertext()) if code is not None else ""
        block: dict[str, object] = {
            "type": BlockType.CODE,
            "content": content,
            "guess_lang": (spec.wrapper.get("data-guess-lang") or "").strip(),
        }
        return block
    algorithm = next(
        (
            element
            for element in body.iterdescendants()
            if isinstance(element.tag, str) and "mineru-algorithm" in _class_tokens(element)
        ),
        body,
    )
    projector = MarkupProjector(algorithm, resources, stylesheet)
    return {"type": RAW_ALGORITHM, "content": _project_inline_content(projector, algorithm)}


def _materialize_table_body(
    body: etree._Element,
    resources: HtmlResourceContext,
    stylesheet: MarkupStylesheet,
) -> dict[str, object]:
    """优先恢复结构表格，其次恢复空间文本或安全图片载荷。"""
    table = next(
        (element for element in body.iterdescendants() if isinstance(element.tag, str) and local_name(element) == "table"),
        None,
    )
    content = ""
    if table is not None:
        projected = MarkupProjector(table, resources, stylesheet).project_block(table)
        table_block = next((block for block in projected if block.get("type") == BlockType.TABLE), None)
        if table_block is not None:
            content = str(table_block.get("content") or "")
    if not content:
        text_fallback = next(
            (
                element
                for element in body.iterdescendants()
                if isinstance(element.tag, str) and bool(_class_tokens(element) & {"mineru-table-text", "mineru-raw-fallback"})
            ),
            None,
        )
        if text_fallback is not None:
            content = "".join(text_fallback.itertext())
    return {
        "type": BlockType.TABLE,
        "content": content,
        **_resolved_image_payload(body, resources, allowed_classes=frozenset({"mineru-table-image"})),
    }


def _materialize_list(container: etree._Element, projector: MarkupProjector) -> dict[str, object]:
    """递归恢复列表叶子精确类型，并让既有无坐标后处理生成规范 marker。"""
    children: list[dict[str, object]] = []
    classes = _class_tokens(container)
    for item in _element_children(container):
        item_type = (item.get("data-block-type") or "").strip()
        nested_lists = [child for child in _element_children(item) if local_name(child) in {"ol", "ul"}]
        if item_type:
            content_host = next(
                (
                    child
                    for child in item.iterdescendants()
                    if isinstance(child.tag, str) and "mineru-list-content" in _class_tokens(child)
                ),
                None,
            )
            content_element = content_host if content_host is not None else _without_nested_lists(item)
            content = _project_inline_content(projector, content_element)
            marker_host = next(
                (
                    child
                    for child in item.iterdescendants()
                    if isinstance(child.tag, str) and "mineru-list-marker" in _class_tokens(child)
                ),
                None,
            )
            marker = "".join(marker_host.itertext()).strip() if marker_host is not None else ""
            if "mineru-list--reference" in classes and marker:
                content = f"{marker} {content}".strip()
            elif "mineru-list--explicit" in classes and marker:
                content = f"{marker} {content}".strip()
            child_block: dict[str, object] = {"type": BlockType(item_type), "content": content}
            if (index := _non_negative_integer(item, "data-block-index", required=False)) is not None:
                child_block["index"] = index
            children.append(child_block)
        children.extend(_materialize_list(nested, projector) for nested in nested_lists)
    block: dict[str, object] = {
        "type": BlockType.LIST,
        "attribute": "ordered" if local_name(container) == "ol" else "unordered",
        "content": children,
    }
    if local_name(container) == "ol":
        block["start"] = _html_list_start(container)
    sub_type = (container.get("data-block-sub-type") or "").strip()
    if sub_type:
        block["sub_type"] = sub_type
    return block


def _materialize_index(content_root: etree._Element, projector: MarkupProjector) -> dict[str, object]:
    """恢复目录根及其精确标题/文本叶子。"""
    root_list = _element_children(content_root)[0]
    return {"type": BlockType.INDEX, "content": _materialize_index_children(root_list, projector)}


def _materialize_index_children(container: etree._Element, projector: MarkupProjector) -> list[dict[str, object]]:
    """递归恢复目录 li 与嵌套 IndexBlock。"""
    children: list[dict[str, object]] = []
    for item in _element_children(container):
        item_type = (item.get("data-block-type") or "").strip()
        nested_lists = [child for child in _element_children(item) if local_name(child) == "ul"]
        if item_type:
            content_host = next(
                (child for child in _element_children(item) if local_name(child) == "a"),
                None,
            )
            content_element = content_host if content_host is not None else _without_nested_lists(item)
            content = _project_inline_content(projector, content_element)
            block: dict[str, object] = {"type": BlockType(item_type), "content": content}
            if item_type in {BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE}:
                block["anchor"] = (item.get("data-anchor") or "").strip()
                block["level"] = int(item.get("data-level") or 1)
            if (index := _non_negative_integer(item, "data-block-index", required=False)) is not None:
                block["index"] = index
            children.append(block)
        for nested in nested_lists:
            nested_block: dict[str, object] = {
                "type": BlockType.INDEX,
                "content": _materialize_index_children(nested, projector),
            }
            if (index := _non_negative_integer(nested, "data-block-index", required=False)) is not None:
                nested_block["index"] = index
            children.append(nested_block)
    return children


def _project_visual_body_content(
    body: etree._Element,
    resources: HtmlResourceContext,
    stylesheet: MarkupStylesheet,
) -> str:
    """删除展示性图片/summary 后，用通用 projector 恢复 visual 的结构内容。"""
    clone = deepcopy(body)
    for element in list(clone.iterdescendants()):
        if not isinstance(element.tag, str):
            continue
        name = local_name(element)
        classes = _class_tokens(element)
        is_owned_image = name == "img" and bool(classes & _OWNED_VISUAL_IMAGE_TOKENS)
        if name == "summary" or "mineru-flowchart-canvas" in classes or is_owned_image:
            _drop_tree_preserve_tail(element)
    blocks = MarkupProjector(clone, resources, stylesheet).convert()
    parts: list[str] = []
    for block in blocks:
        block_type = block.get("type")
        content = str(block.get("content") or "")
        if block_type == BlockType.EQUATION and content:
            parts.append(f"<eq>{html.escape(content, quote=False)}</eq>")
        elif content:
            parts.append(content if block_type == BlockType.TABLE else html.unescape(content))
    return "\n".join(parts)


def _project_inline_content(projector: MarkupProjector, element: etree._Element) -> str:
    """恢复 projector 为内部标签安全转义的一层实体，避免二次 HTML escape。"""
    return html.unescape(projector.project_inline_content(element))


def _flowchart_content(body: etree._Element) -> str | None:
    """从 renderer 的受限 Mermaid 源码回退恢复标准 fence。"""
    source = next(
        (
            element
            for element in body.iterdescendants()
            if isinstance(element.tag, str) and "mineru-flowchart-source" in _class_tokens(element)
        ),
        None,
    )
    if source is None:
        return None
    value = "".join(source.itertext()).strip("\n")
    return f"```mermaid\n{value}\n```" if value else None


def _resolved_image_payload(
    element: etree._Element,
    resources: HtmlResourceContext,
    *,
    allowed_classes: frozenset[str] | None = None,
) -> dict[str, object]:
    """解析子树中首个图片来源，并仅返回统一安全载荷字段。"""
    candidates = [element] if local_name(element) == "img" else list(element.iterdescendants())
    image = next(
        (
            child
            for child in candidates
            if isinstance(child.tag, str)
            and local_name(child) == "img"
            and (allowed_classes is None or bool(_class_tokens(child) & allowed_classes))
        ),
        None,
    )
    if image is None:
        return {}
    resolved = resources.resolve_image(image.get("src") or "", alt=image.get("alt") or "")
    if resolved is None:
        return {}
    if resolved.image_base64:
        return {"image_base64": resolved.image_base64}
    if resolved.image_url:
        return {"image_url": resolved.image_url}
    return {}


def _without_nested_lists(element: etree._Element) -> etree._Element:
    """复制 li 并移除直属嵌套列表，避免叶子内容重复包含子列表。"""
    clone = deepcopy(element)
    for child in list(_element_children(clone)):
        if local_name(child) in {"ol", "ul"}:
            _drop_tree_preserve_tail(child)
    return clone


def _drop_tree_preserve_tail(element: etree._Element) -> None:
    """删除展示节点并把 tail 归还到相邻文本位置。"""
    parent = element.getparent()
    if parent is None:
        return
    tail = element.tail or ""
    previous = element.getprevious()
    if tail:
        if previous is not None:
            previous.tail = (previous.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
    parent.remove(element)


def _page_block_type(wrapper: etree._Element) -> BlockType:
    """把顶层 data-block-type 严格转换为公开 PageBlock 类型。"""
    value = (wrapper.get("data-block-type") or "").strip()
    try:
        block_type = BlockType(value)
    except ValueError as exc:
        raise _WireValidationError("invalid_block_type") from exc
    if block_type not in PAGE_BLOCK_TYPES:
        raise _WireValidationError("invalid_top_level_type")
    return block_type


def _element_children(element: etree._Element) -> list[etree._Element]:
    """返回元素直属的真实标签子节点。"""
    return [child for child in element if isinstance(child.tag, str)]


def _validate_element_only_content(element: etree._Element) -> None:
    """拒绝机器结构容器直属的可见文本，避免精确物化静默丢弃内容。"""
    if (element.text or "").strip() or any((child.tail or "").strip() for child in element):
        raise _WireValidationError("unexpected_wire_text")


def _validate_inline_content_shape(
    element: etree._Element,
    *,
    excluded_roots: list[etree._Element] | None = None,
) -> None:
    """拒绝行内物化区域中的块级或图片后代，确保校验失败时事务式回退。"""
    excluded = excluded_roots or []
    for candidate in element.iterdescendants():
        if not isinstance(candidate.tag, str) or (excluded and _is_within_any(candidate, excluded)):
            continue
        name = local_name(candidate)
        marker_type = (candidate.get("data-block-type") or "").strip()
        if name == "math" and marker_type == BlockType.EQUATION:
            continue
        if name in BLOCK_TAGS or name in {"image", "img"}:
            raise _WireValidationError("invalid_inline_child")


def _descendant_markers(element: etree._Element) -> list[etree._Element]:
    """返回自身及后代中声明 data-block-type 的元素。"""
    return [
        candidate
        for candidate in [element, *element.iterdescendants()]
        if isinstance(candidate.tag, str) and candidate.get("data-block-type") is not None
    ]


def _is_within_any(element: etree._Element, roots: list[etree._Element]) -> bool:
    """判断元素是否等于或位于任一嵌套语义根之内。"""
    ancestors = set(element.iterancestors())
    return any(root is element or root in ancestors for root in roots)


def _class_tokens(element: etree._Element) -> frozenset[str]:
    """按 HTML class 空白边界返回完整小写 token，不做 substring 推断。"""
    return frozenset((element.get("class") or "").casefold().split())


def _non_negative_integer(element: etree._Element, name: str, *, required: bool) -> int | None:
    """读取非负整数 data 属性，缺失或非法时按契约失败。"""
    value = element.get(name)
    if value is None:
        if required:
            raise _WireValidationError("missing_integer_attribute")
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise _WireValidationError("invalid_integer_attribute") from exc
    if parsed < 0 or str(parsed) != value.strip():
        raise _WireValidationError("invalid_integer_attribute")
    return parsed


def _optional_integer(element: etree._Element, name: str) -> int | None:
    """读取可选整数属性，非法文本不允许静默降级。"""
    value = element.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise _WireValidationError("invalid_integer_attribute") from exc


def _html_list_start(element: etree._Element) -> int:
    """读取 renderer 原生有序列表起始值，非法值保守回退为一。"""
    try:
        value = int(element.get("start") or 1)
    except ValueError:
        return 1
    return value if value >= 0 else 1


__all__ = [
    "MineruHtmlBlockSpec",
    "MineruHtmlWireInspection",
    "MineruHtmlWirePlan",
    "inspect_mineru_html_wire",
    "materialize_mineru_html_wire",
]
