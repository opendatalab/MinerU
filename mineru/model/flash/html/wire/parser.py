# Copyright (c) Opendatalab. All rights reserved.
"""把 MinerU HTML v1 固定 DOM 解析为无资源副作用的 typed plan。"""

from __future__ import annotations

from copy import deepcopy
from typing import cast
from urllib.parse import unquote

from lxml import etree  # type: ignore[reportMissingImports]

from .....types import PAGE_BLOCK_TYPES, RAW_ALGORITHM, BlockType, VISUAL_TYPE_MAPPING
from ..._shared.markup import extract_formula
from ..._shared.markup.projector import BLOCK_TAGS, local_name
from .contracts import (
    AnnotationWireSpec,
    CodeBodyWireSpec,
    EquationWireSpec,
    FlowchartBodyWireSpec,
    IndexBlockWireSpec,
    IndexLeafWireSpec,
    IndexWireSpec,
    ListBlockWireSpec,
    ListLeafWireSpec,
    ListWireSpec,
    MINERU_HTML_VERSION,
    MineruHtmlWirePlan,
    PageWireSpec,
    RichVisualBodyWireSpec,
    TableBodyWireSpec,
    TextWireSpec,
    VisualBodyWireSpec,
    VisualWireSpec,
    WireFallbackReason,
    WireRenderMode,
    WIRE_BLOCK_CLASS,
    WIRE_DOCUMENT_CLASS,
    WIRE_INDEX_CLASS,
    WIRE_LIST_CONTENT_CLASS,
    WIRE_LIST_MARKER_CLASS,
    WIRE_PAGE_BREAK_CLASS,
    WIRE_PAGE_CLASS,
    WIRE_VISUAL_BODY_CLASS,
)


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
_PAGE_AUXILIARY_TYPES = frozenset({BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER, BlockType.ASIDE_TEXT})
_LIST_LEAF_TYPES = frozenset({BlockType.TEXT, BlockType.REF_TEXT})
_INDEX_LEAF_TYPES = frozenset({BlockType.TEXT, BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE})
_OWNED_VISUAL_IMAGE_TOKENS = frozenset(
    {"mineru-chart-image", "mineru-flowchart-fallback", "mineru-image", "mineru-table-image"}
)


class NonCanonicalWire(ValueError):
    """表示当前 DOM 不是 renderer 能生成的 canonical v1 wire。"""


def parse_mineru_html_wire(body: etree._Element) -> tuple[MineruHtmlWirePlan | None, WireFallbackReason | None]:
    """发现并解析 canonical v1 wire，非法结构只返回统一回退原因。"""
    roots = [
        element
        for element in body.iter()
        if isinstance(element.tag, str) and element.get("data-mineru-html-version") is not None
    ]
    if not roots:
        return None, None
    if len(roots) != 1:
        return None, "non_canonical_wire"
    root = roots[0]
    if (root.get("data-mineru-html-version") or "").strip() != MINERU_HTML_VERSION:
        return None, "unsupported_version"
    try:
        _validate_wire_root_ownership(body, root)
        return _parse_wire_root(root), None
    except NonCanonicalWire:
        return None, "non_canonical_wire"


def _parse_wire_root(root: etree._Element) -> MineruHtmlWirePlan:
    """解析根、渲染模式、页面容器和全部顶层 block。"""
    if local_name(root) != "article" or _class_tokens(root) != {
        WIRE_DOCUMENT_CLASS,
        f"{WIRE_DOCUMENT_CLASS}--{(root.get('data-render-mode') or '').strip()}",
    }:
        raise NonCanonicalWire
    _validate_structural_text(root)
    mode_value = (root.get("data-render-mode") or "").strip()
    if mode_value not in {"default", "full"}:
        raise NonCanonicalWire
    mode = cast(WireRenderMode, mode_value)
    wrappers: list[tuple[etree._Element, int | None]] = []
    if mode == "default":
        for child in _element_children(root):
            if local_name(child) != "div" or _class_tokens(child) != {WIRE_BLOCK_CLASS}:
                raise NonCanonicalWire
            wrappers.append((child, None))
    else:
        for child in _element_children(root):
            if local_name(child) == "hr" and _class_tokens(child) == {WIRE_PAGE_BREAK_CLASS}:
                continue
            if local_name(child) != "section" or _class_tokens(child) != {WIRE_PAGE_CLASS}:
                raise NonCanonicalWire
            _validate_structural_text(child)
            page_idx = _non_negative_integer(child, "data-page-idx", required=True)
            for wrapper in _element_children(child):
                if local_name(wrapper) != "div" or _class_tokens(wrapper) != {WIRE_BLOCK_CLASS}:
                    raise NonCanonicalWire
                wrappers.append((wrapper, page_idx))
    nested_wrappers = [
        element
        for element in root.iterdescendants()
        if isinstance(element.tag, str) and WIRE_BLOCK_CLASS in _class_tokens(element)
    ]
    if len(nested_wrappers) != len(wrappers):
        raise NonCanonicalWire
    target_ids = _collect_title_target_ids(wrappers)
    blocks = tuple(_parse_top_block(wrapper, section_page_idx, target_ids) for wrapper, section_page_idx in wrappers)
    return MineruHtmlWirePlan(root, mode, blocks)


def _validate_wire_root_ownership(body: etree._Element, root: etree._Element) -> None:
    """要求 canonical wire 根独占从自身到 body 的可见内容路径。"""
    current = root
    while current is not body:
        parent = current.getparent()
        if parent is None or (parent.text or "").strip():
            raise NonCanonicalWire
        for sibling in parent:
            if sibling is current:
                if (sibling.tail or "").strip():
                    raise NonCanonicalWire
                continue
            if isinstance(sibling.tag, str) or (sibling.tail or "").strip():
                raise NonCanonicalWire
        current = parent


def _parse_top_block(
    wrapper: etree._Element,
    section_page_idx: int | None,
    target_ids: dict[str, frozenset[str]],
) -> PageWireSpec:
    """解析一个顶层 wrapper，并构造与 block 家族匹配的 typed spec。"""
    _validate_structural_text(wrapper)
    block_type = _page_block_type(wrapper)
    page_idx = _non_negative_integer(wrapper, "data-page-idx", required=True)
    if section_page_idx is not None and page_idx != section_page_idx:
        raise NonCanonicalWire
    block_index = _non_negative_integer(wrapper, "data-block-index", required=False)
    _validate_top_metadata(wrapper, block_type)
    roots = _element_children(wrapper)
    if len(roots) != 1:
        raise NonCanonicalWire
    content_root = roots[0]
    if block_type in _SIMPLE_TEXT_TYPES:
        _validate_simple_content(content_root, block_type)
        return TextWireSpec(wrapper, content_root, block_type, page_idx, block_index)
    if block_type == BlockType.EQUATION:
        _validate_equation_content(content_root)
        return EquationWireSpec(wrapper, content_root, page_idx, block_index)
    if block_type == BlockType.LIST:
        root = _parse_list_container(content_root, top_wrapper=wrapper)
        return ListBlockWireSpec(wrapper, page_idx, block_index, root)
    if block_type == BlockType.INDEX:
        root = _parse_index_root(content_root, wrapper, target_ids)
        return IndexBlockWireSpec(wrapper, page_idx, block_index, root)
    if block_type in VISUAL_TYPE_MAPPING:
        return _parse_visual_content(wrapper, content_root, block_type, page_idx, block_index)
    raise NonCanonicalWire


def _validate_top_metadata(wrapper: etree._Element, block_type: BlockType) -> None:
    """校验 subtype、语言、anchor 和 level 只出现在 renderer 定义的位置。"""
    sub_type = (wrapper.get("data-block-sub-type") or "").strip()
    guess_lang = (wrapper.get("data-guess-lang") or "").strip()
    anchor = (wrapper.get("data-anchor") or "").strip()
    level = _optional_integer(wrapper, "data-level")
    if block_type == BlockType.CODE:
        if sub_type not in {BlockType.CODE, RAW_ALGORITHM}:
            raise NonCanonicalWire
        if sub_type == BlockType.CODE and not guess_lang:
            raise NonCanonicalWire
        if sub_type == RAW_ALGORITHM and guess_lang:
            raise NonCanonicalWire
    elif block_type == BlockType.LIST:
        if sub_type and sub_type not in _LIST_LEAF_TYPES:
            raise NonCanonicalWire
        if guess_lang:
            raise NonCanonicalWire
    elif block_type in {BlockType.IMAGE, BlockType.CHART}:
        if guess_lang:
            raise NonCanonicalWire
    elif sub_type or guess_lang:
        raise NonCanonicalWire
    if block_type == BlockType.DOC_TITLE:
        if level != 1:
            raise NonCanonicalWire
    elif block_type == BlockType.PARAGRAPH_TITLE:
        if level is None or not 2 <= level <= 6:
            raise NonCanonicalWire
    elif level is not None:
        raise NonCanonicalWire
    if anchor and block_type not in {BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE, BlockType.PAGE_FOOTNOTE}:
        raise NonCanonicalWire


def _validate_simple_content(content_root: etree._Element, block_type: BlockType) -> None:
    """校验文本、标题、脚注和页面辅助 block 的固定行内容器。"""
    expected_tags: dict[BlockType, frozenset[str]] = {
        BlockType.TEXT: frozenset({"p"}),
        BlockType.REF_TEXT: frozenset({"p"}),
        BlockType.DOC_TITLE: frozenset({"h1"}),
        BlockType.PARAGRAPH_TITLE: frozenset({"h2", "h3", "h4", "h5", "h6"}),
        BlockType.PAGE_FOOTNOTE: frozenset({"div"}),
        **{value: frozenset({"div"}) for value in _PAGE_AUXILIARY_TYPES},
    }
    if local_name(content_root) not in expected_tags[block_type]:
        raise NonCanonicalWire
    _validate_inline_region(content_root)


def _validate_equation_content(content_root: etree._Element) -> None:
    """校验行间公式为 renderer math carrier 或公式图片。"""
    if local_name(content_root) == "math":
        _validate_formula_carrier(content_root, expected_display="block")
        return
    if local_name(content_root) != "img" or _class_tokens(content_root) != {"mineru-equation-image"}:
        raise NonCanonicalWire


def _validate_formula_carrier(element: etree._Element, *, expected_display: str) -> None:
    """校验公式 carrier 的类型、显示模式和非空 LaTeX。"""
    if local_name(element) != "math" or (element.get("data-block-type") or "").strip() != BlockType.EQUATION:
        raise NonCanonicalWire
    if (element.get("data-formula-display") or "").strip() != expected_display:
        raise NonCanonicalWire
    formula = extract_formula(element)
    if formula is None or not formula.latex:
        raise NonCanonicalWire


def _parse_visual_content(
    wrapper: etree._Element,
    content_root: etree._Element,
    parent_type: BlockType,
    page_idx: int,
    block_index: int | None,
) -> VisualWireSpec:
    """解析 figure 下唯一 body 与有序 annotation 子节点。"""
    if local_name(content_root) != "figure":
        raise NonCanonicalWire
    _validate_structural_text(content_root)
    mapping = VISUAL_TYPE_MAPPING[parent_type]
    sub_type = (wrapper.get("data-block-sub-type") or "").strip()
    body_type = BlockType.ALGORITHM_BODY if parent_type == BlockType.CODE and sub_type == RAW_ALGORITHM else mapping["body"]
    allowed_types = frozenset({body_type, mapping["caption"], mapping["footnote"]})
    children = _element_children(content_root)
    child_types = [(child.get("data-block-type") or "").strip() for child in children]
    if child_types.count(body_type) != 1 or any(value not in allowed_types for value in child_types):
        raise NonCanonicalWire
    guess_lang = (wrapper.get("data-guess-lang") or "").strip()
    parsed_children: list[VisualBodyWireSpec | AnnotationWireSpec] = []
    for child, child_type in zip(children, child_types, strict=True):
        child_index = _non_negative_integer(child, "data-block-index", required=False)
        if child_type == body_type:
            if local_name(child) != "div" or (block_index is not None and child_index != block_index):
                raise NonCanonicalWire
            parsed_children.append(_parse_visual_body(child, parent_type, sub_type))
            continue
        if local_name(child) != "p":
            raise NonCanonicalWire
        _validate_inline_region(child)
        parsed_children.append(AnnotationWireSpec(child, BlockType(child_type)))
    return VisualWireSpec(
        wrapper,
        content_root,
        parent_type,
        page_idx,
        block_index,
        sub_type,
        guess_lang,
        tuple(parsed_children),
    )


def _parse_visual_body(body: etree._Element, parent_type: BlockType, sub_type: str) -> VisualBodyWireSpec:
    """按父 visual 类型解析唯一 canonical body 载荷。"""
    expected_class = f"{WIRE_VISUAL_BODY_CLASS}--{'image' if parent_type == BlockType.IMAGE else str(parent_type)}"
    if _class_tokens(body) != {WIRE_VISUAL_BODY_CLASS, expected_class}:
        raise NonCanonicalWire
    if parent_type == BlockType.CODE:
        return _parse_code_body(body, sub_type)
    if parent_type == BlockType.TABLE:
        return _parse_table_body(body)
    if parent_type == BlockType.IMAGE and _looks_like_flowchart_body(body):
        return _parse_flowchart_body(body)
    return _parse_rich_visual_body(body, parent_type, sub_type)


def _parse_code_body(body: etree._Element, sub_type: str) -> CodeBodyWireSpec:
    """解析普通代码或 algorithm 的固定内容载体。"""
    _validate_structural_text(body)
    children = _element_children(body)
    if sub_type == BlockType.CODE:
        if len(children) != 1 or local_name(children[0]) != "pre":
            raise NonCanonicalWire
        pre = children[0]
        _validate_structural_text(pre)
        code_children = _element_children(pre)
        if len(code_children) != 1 or local_name(code_children[0]) != "code" or _element_children(code_children[0]):
            raise NonCanonicalWire
        return CodeBodyWireSpec(body, "code", code_children[0])
    if sub_type != RAW_ALGORITHM:
        raise NonCanonicalWire
    if not children:
        empty = etree.Element("div")
        return CodeBodyWireSpec(body, "algorithm", empty)
    if len(children) != 1 or local_name(children[0]) != "div" or _class_tokens(children[0]) != {"mineru-algorithm"}:
        raise NonCanonicalWire
    _validate_inline_region(children[0])
    return CodeBodyWireSpec(body, "algorithm", _clone_fragment(children[0]))


def _parse_table_body(body: etree._Element) -> TableBodyWireSpec:
    """解析结构表格、空间文本、图片或空 table body。"""
    _validate_structural_text(body)
    children = _element_children(body)
    if not children:
        return TableBodyWireSpec(body, "empty", None)
    if len(children) != 1:
        raise NonCanonicalWire
    child = children[0]
    name = local_name(child)
    classes = _class_tokens(child)
    if name == "table":
        return TableBodyWireSpec(body, "html", child)
    if name == "pre" and classes in ({"mineru-table-text"}, {"mineru-raw-fallback"}) and not _element_children(child):
        return TableBodyWireSpec(body, "text", child)
    if name == "img" and classes == {"mineru-table-image"}:
        return TableBodyWireSpec(body, "image", child)
    raise NonCanonicalWire


def _looks_like_flowchart_body(body: etree._Element) -> bool:
    """判断 body 是否使用 renderer 的 flowchart 固定外壳。"""
    children = _element_children(body)
    return bool(children and local_name(children[0]) == "div" and "mineru-flowchart" in _class_tokens(children[0]))


def _parse_flowchart_body(body: etree._Element) -> FlowchartBodyWireSpec:
    """解析 flowchart canvas、可选 raster 和源码 details。"""
    _validate_structural_text(body)
    children = _element_children(body)
    if len(children) != 2:
        raise NonCanonicalWire
    display, details = children
    display_classes = _class_tokens(display)
    if local_name(display) != "div" or "mineru-flowchart" not in display_classes:
        raise NonCanonicalWire
    _validate_structural_text(display)
    display_children = _element_children(display)
    if not 1 <= len(display_children) <= 2:
        raise NonCanonicalWire
    canvas = display_children[0]
    if local_name(canvas) != "div" or _class_tokens(canvas) != {"mineru-flowchart-canvas"}:
        raise NonCanonicalWire
    _validate_structural_text(canvas)
    if _element_children(canvas):
        raise NonCanonicalWire
    fallback_image = None
    if len(display_children) == 2:
        fallback_image = display_children[1]
        if local_name(fallback_image) != "img" or _class_tokens(fallback_image) != {"mineru-flowchart-fallback"}:
            raise NonCanonicalWire
    if local_name(details) != "details" or _class_tokens(details) != {"mineru-details", "mineru-flowchart-details"}:
        raise NonCanonicalWire
    _validate_structural_text(details)
    details_children = _element_children(details)
    if len(details_children) != 2:
        raise NonCanonicalWire
    summary, source = details_children
    if (
        local_name(summary) != "summary"
        or _element_children(summary)
        or " ".join(summary.itertext()).strip() != "flowchart source"
    ):
        raise NonCanonicalWire
    if local_name(source) != "pre" or _class_tokens(source) != {"mineru-flowchart-source"}:
        raise NonCanonicalWire
    _validate_structural_text(source)
    code_children = _element_children(source)
    if len(code_children) != 1 or local_name(code_children[0]) != "code" or _element_children(code_children[0]):
        raise NonCanonicalWire
    return FlowchartBodyWireSpec(body, code_children[0], fallback_image)


def _parse_rich_visual_body(body: etree._Element, parent_type: BlockType, sub_type: str) -> RichVisualBodyWireSpec:
    """区分 renderer-owned 主图与开放但受 sanitizer 约束的富内容 carrier。"""
    allowed_token = "mineru-image" if parent_type == BlockType.IMAGE else "mineru-chart-image"
    children = _element_children(body)
    primary_image = (
        children[0] if children and local_name(children[0]) == "img" and allowed_token in _class_tokens(children[0]) else None
    )
    if primary_image is not None and _class_tokens(primary_image) != {allowed_token}:
        raise NonCanonicalWire
    for element in body.iterdescendants():
        if not isinstance(element.tag, str) or element is primary_image:
            continue
        if _class_tokens(element) & _OWNED_VISUAL_IMAGE_TOKENS:
            raise NonCanonicalWire
    if primary_image is None:
        return RichVisualBodyWireSpec(body, parent_type, sub_type, None, _clone_fragment(body))
    if (body.text or "").strip() or (primary_image.tail or "").strip():
        raise NonCanonicalWire
    remaining = children[1:]
    if not remaining:
        return RichVisualBodyWireSpec(body, parent_type, sub_type, primary_image, None)
    if len(remaining) != 1:
        raise NonCanonicalWire
    details = remaining[0]
    if local_name(details) != "details" or _class_tokens(details) != {"mineru-details"}:
        raise NonCanonicalWire
    if (details.text or "").strip() or (details.tail or "").strip():
        raise NonCanonicalWire
    details_children = _element_children(details)
    if not details_children or local_name(details_children[0]) != "summary" or _element_children(details_children[0]):
        raise NonCanonicalWire
    expected_summary = sub_type or ("image content" if parent_type == BlockType.IMAGE else "chart content")
    if " ".join(details_children[0].itertext()).strip() != expected_summary:
        raise NonCanonicalWire
    fragment = _clone_fragment(details, after_child=details_children[0])
    return RichVisualBodyWireSpec(body, parent_type, sub_type, primary_image, fragment)


def _parse_list_container(container: etree._Element, *, top_wrapper: etree._Element | None = None) -> ListWireSpec:
    """递归解析 renderer 生成的列表 carrier、叶子和嵌套列表。"""
    if local_name(container) not in {"ol", "ul"} or (container.get("data-block-type") or "").strip() != BlockType.LIST:
        raise NonCanonicalWire
    _validate_structural_text(container)
    block_index = _non_negative_integer(container, "data-block-index", required=False)
    sub_type = (container.get("data-block-sub-type") or "").strip()
    if sub_type and sub_type not in _LIST_LEAF_TYPES:
        raise NonCanonicalWire
    if top_wrapper is not None and sub_type != (top_wrapper.get("data-block-sub-type") or "").strip():
        raise NonCanonicalWire
    classes = _class_tokens(container)
    if "mineru-list" not in classes or len(classes) != 2:
        raise NonCanonicalWire
    children: list[ListLeafWireSpec | ListWireSpec] = []
    for item in _element_children(container):
        if local_name(item) != "li":
            raise NonCanonicalWire
        item_type = (item.get("data-block-type") or "").strip()
        nested_lists = [child for child in _element_children(item) if local_name(child) in {"ol", "ul"}]
        if item_type:
            if item_type not in _LIST_LEAF_TYPES:
                raise NonCanonicalWire
            leaf = _parse_list_leaf(item, BlockType(item_type), nested_lists)
            children.append(leaf)
        elif any(child not in nested_lists for child in _element_children(item)) or not nested_lists:
            raise NonCanonicalWire
        children.extend(_parse_list_container(nested) for nested in nested_lists)
    start = _canonical_list_start(container)
    return ListWireSpec(container, block_index, local_name(container) == "ol", start, sub_type, classes, tuple(children))


def _parse_list_leaf(
    item: etree._Element,
    block_type: BlockType,
    nested_lists: list[etree._Element],
) -> ListLeafWireSpec:
    """解析一个列表叶子的唯一 marker/content carrier。"""
    block_index = _non_negative_integer(item, "data-block-index", required=False)
    candidates = [child for child in _element_children(item) if child not in nested_lists]
    content_carriers = [child for child in candidates if _class_tokens(child) == {WIRE_LIST_CONTENT_CLASS}]
    marker_carriers = [child for child in candidates if _class_tokens(child) == {WIRE_LIST_MARKER_CLASS}]
    if len(content_carriers) > 1 or len(marker_carriers) > 1:
        raise NonCanonicalWire
    allowed = [*content_carriers, *marker_carriers, *nested_lists]
    if any(child not in allowed for child in _element_children(item)):
        raise NonCanonicalWire
    if content_carriers:
        _validate_structural_text(item)
        content_element = content_carriers[0]
        if local_name(content_element) != "span":
            raise NonCanonicalWire
        _validate_inline_region(content_element)
    else:
        if marker_carriers or (item.text or "").strip() or any((child.tail or "").strip() for child in item):
            raise NonCanonicalWire
        content_element = None
    marker = ""
    if marker_carriers:
        marker_element = marker_carriers[0]
        if local_name(marker_element) != "span" or _element_children(marker_element):
            raise NonCanonicalWire
        marker = "".join(marker_element.itertext()).strip()
    return ListLeafWireSpec(block_type, block_index, content_element, marker)


def _parse_index_root(
    content_root: etree._Element,
    wrapper: etree._Element,
    target_ids: dict[str, frozenset[str]],
) -> IndexWireSpec:
    """解析目录根与唯一直属 ul。"""
    if local_name(content_root) != "nav" or _class_tokens(content_root) != {WIRE_INDEX_CLASS}:
        raise NonCanonicalWire
    if (content_root.get("data-block-type") or "").strip() != BlockType.INDEX:
        raise NonCanonicalWire
    _validate_structural_text(content_root)
    root_index = _non_negative_integer(content_root, "data-block-index", required=False)
    wrapper_index = _non_negative_integer(wrapper, "data-block-index", required=False)
    if wrapper_index is not None and root_index != wrapper_index:
        raise NonCanonicalWire
    lists = _element_children(content_root)
    if len(lists) != 1 or local_name(lists[0]) != "ul":
        raise NonCanonicalWire
    return _parse_index_list(lists[0], nested=False, target_ids=target_ids, block_index=root_index)


def _parse_index_list(
    container: etree._Element,
    *,
    nested: bool,
    target_ids: dict[str, frozenset[str]],
    block_index: int | None = None,
) -> IndexWireSpec:
    """递归解析目录叶子、linked carrier 和嵌套 IndexBlock。"""
    _validate_structural_text(container)
    if nested:
        if (container.get("data-block-type") or "").strip() != BlockType.INDEX:
            raise NonCanonicalWire
        block_index = _non_negative_integer(container, "data-block-index", required=False)
    children: list[IndexLeafWireSpec | IndexWireSpec] = []
    for item in _element_children(container):
        if local_name(item) != "li":
            raise NonCanonicalWire
        item_type_value = (item.get("data-block-type") or "").strip()
        nested_lists = [child for child in _element_children(item) if local_name(child) == "ul"]
        if item_type_value:
            try:
                item_type = BlockType(item_type_value)
            except ValueError as exc:
                raise NonCanonicalWire from exc
            if item_type not in _INDEX_LEAF_TYPES:
                raise NonCanonicalWire
            children.append(_parse_index_leaf(item, item_type, nested_lists, target_ids))
        elif any(child not in nested_lists for child in _element_children(item)) or not nested_lists:
            raise NonCanonicalWire
        children.extend(_parse_index_list(nested_list, nested=True, target_ids=target_ids) for nested_list in nested_lists)
    return IndexWireSpec(container, block_index, tuple(children))


def _parse_index_leaf(
    item: etree._Element,
    item_type: BlockType,
    nested_lists: list[etree._Element],
    target_ids: dict[str, frozenset[str]],
) -> IndexLeafWireSpec:
    """解析 linked/unlinked 目录叶子并封闭 anchor 外结构。"""
    block_index = _non_negative_integer(item, "data-block-index", required=False)
    anchor = (item.get("data-anchor") or "").strip()
    level = _optional_integer(item, "data-level")
    _validate_index_leaf_metadata(item_type, anchor, level)
    direct_content = [child for child in _element_children(item) if child not in nested_lists]
    linked = [
        child
        for child in direct_content
        if _is_canonical_index_link(child, item_type=item_type, anchor=anchor, target_ids=target_ids)
    ]
    if linked:
        if len(linked) != 1 or len(direct_content) != 1:
            raise NonCanonicalWire
        _validate_structural_text(item)
        content_element = linked[0]
        _validate_inline_region(content_element)
    else:
        content_element = _clone_fragment(item, excluded_children=nested_lists)
        _validate_inline_region(content_element)
    return IndexLeafWireSpec(item_type, block_index, content_element, anchor, level)


def _collect_title_target_ids(
    wrappers: list[tuple[etree._Element, int | None]],
) -> dict[str, frozenset[str]]:
    """预收集 renderer 标题 id，供目录 linked carrier 做确定性判定。"""
    collected: dict[str, set[str]] = {}
    for wrapper, _ in wrappers:
        block_type = (wrapper.get("data-block-type") or "").strip()
        if block_type not in {BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE}:
            continue
        anchor = (wrapper.get("data-anchor") or "").strip()
        children = _element_children(wrapper)
        if not anchor or len(children) != 1:
            continue
        identities = {
            identity
            for element in [children[0], *children[0].iterdescendants()]
            if (identity := (element.get("id") or "").strip())
        }
        if identities:
            collected.setdefault(anchor, set()).update(identities)
    return {anchor: frozenset(identities) for anchor, identities in collected.items()}


def _is_canonical_index_link(
    element: etree._Element,
    *,
    item_type: BlockType,
    anchor: str,
    target_ids: dict[str, frozenset[str]],
) -> bool:
    """判断直属 anchor 是否为 renderer 生成的目录目标外壳。"""
    if item_type not in {BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE} or local_name(element) != "a":
        return False
    href = (element.get("href") or "").strip()
    if not href.startswith("#"):
        return False
    return unquote(href[1:]).strip() in target_ids.get(anchor, frozenset())


def _validate_index_leaf_metadata(item_type: BlockType, anchor: str, level: int | None) -> None:
    """校验目录标题叶子的 anchor/level 与普通文本互斥。"""
    if item_type == BlockType.TEXT:
        if anchor or level is not None:
            raise NonCanonicalWire
        return
    if not anchor:
        raise NonCanonicalWire
    if item_type == BlockType.DOC_TITLE and level != 1:
        raise NonCanonicalWire
    if item_type == BlockType.PARAGRAPH_TITLE and (level is None or not 2 <= level <= 6):
        raise NonCanonicalWire


def _validate_inline_region(element: etree._Element) -> None:
    """校验 canonical 行内区域只含行内节点和可信公式 carrier。"""
    for candidate in element.iterdescendants():
        if not isinstance(candidate.tag, str):
            continue
        name = local_name(candidate)
        marker_type = (candidate.get("data-block-type") or "").strip()
        if name == "math" and marker_type == BlockType.EQUATION:
            _validate_formula_carrier(candidate, expected_display="inline")
            continue
        if name in BLOCK_TAGS or name in {"image", "img"}:
            raise NonCanonicalWire


def _clone_fragment(
    element: etree._Element,
    *,
    after_child: etree._Element | None = None,
    excluded_children: list[etree._Element] | None = None,
) -> etree._Element:
    """复制一个不含 renderer 外壳的富内容片段供 materializer 使用。"""
    fragment = etree.Element("div")
    excluded = excluded_children or []
    children = _element_children(element)
    start = 0
    if after_child is None:
        fragment.text = element.text
    else:
        try:
            start = children.index(after_child) + 1
        except ValueError as exc:
            raise NonCanonicalWire from exc
        fragment.text = after_child.tail
    for child in children[start:]:
        if child in excluded:
            continue
        fragment.append(deepcopy(child))
    return fragment


def _validate_structural_text(element: etree._Element) -> None:
    """拒绝 renderer 结构容器直属的非空文本和 tail。"""
    if (element.text or "").strip() or any((child.tail or "").strip() for child in element):
        raise NonCanonicalWire


def _page_block_type(wrapper: etree._Element) -> BlockType:
    """把顶层 data-block-type 转换为公开 PageBlock 类型。"""
    try:
        block_type = BlockType((wrapper.get("data-block-type") or "").strip())
    except ValueError as exc:
        raise NonCanonicalWire from exc
    if block_type not in PAGE_BLOCK_TYPES:
        raise NonCanonicalWire
    return block_type


def _element_children(element: etree._Element) -> list[etree._Element]:
    """返回元素直属的真实标签子节点。"""
    return [child for child in element if isinstance(child.tag, str)]


def _class_tokens(element: etree._Element) -> frozenset[str]:
    """按 HTML class 空白边界返回完整小写 token。"""
    return frozenset((element.get("class") or "").casefold().split())


def _non_negative_integer(element: etree._Element, name: str, *, required: bool) -> int | None:
    """读取 canonical 非负整数 data 属性。"""
    value = element.get(name)
    if value is None:
        if required:
            raise NonCanonicalWire
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise NonCanonicalWire from exc
    if parsed < 0 or str(parsed) != value.strip():
        raise NonCanonicalWire
    return parsed


def _optional_integer(element: etree._Element, name: str) -> int | None:
    """读取可选整数属性并拒绝非法文本。"""
    value = element.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise NonCanonicalWire from exc


def _canonical_list_start(element: etree._Element) -> int:
    """读取 renderer 生成的合法有序列表起始值。"""
    value = element.get("start")
    if value is None:
        return 1
    try:
        parsed = int(value)
    except ValueError as exc:
        raise NonCanonicalWire from exc
    if parsed < 0 or str(parsed) != value.strip():
        raise NonCanonicalWire
    return parsed


__all__ = ["NonCanonicalWire", "parse_mineru_html_wire"]
