# Copyright (c) Opendatalab. All rights reserved.
"""把已验证的 MinerU HTML v1 typed plan 物化为 raw model-list。"""

from __future__ import annotations

from copy import deepcopy
import html

from lxml import etree  # type: ignore[reportMissingImports]

from .....types import RAW_ALGORITHM, BlockType
from ..._shared.markup import MarkupProjector, MarkupStylesheet, extract_formula
from ..._shared.markup.projector import BLOCK_TAGS, local_name
from ..._shared.spans import text_spans
from ..resources import HtmlResourceContext
from .contracts import (
    AnnotationWireSpec,
    CodeBodyWireSpec,
    EquationWireSpec,
    FlowchartBodyWireSpec,
    IndexBlockWireSpec,
    IndexLeafWireSpec,
    IndexWireSpec,
    ListBlockWireSpec,
    ListWireSpec,
    MineruHtmlWirePlan,
    RichVisualBodyWireSpec,
    TableBodyWireSpec,
    TextWireSpec,
    VisualWireSpec,
)


class ExactAnchorResolver:
    """把 renderer DOM id 恢复为 typed plan 中保存的原始 anchor。"""

    def __init__(self, plan: MineruHtmlWirePlan) -> None:
        """预扫描标题和页面脚注的 id、文本及原始 anchor。"""
        self._targets: dict[str, str] = {}
        self._heading_anchors: dict[etree._Element, str] = {}
        self._heading_labels: dict[str, str] = {}
        self._note_anchors: dict[etree._Element, str] = {}
        for spec in plan.blocks:
            if not isinstance(spec, TextWireSpec):
                continue
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
        """返回原始 anchor 对应的标题文本。"""
        return self._heading_labels.get(anchor)

    def note_anchor(self, note: etree._Element) -> str | None:
        """返回精确页面脚注节点的原始 anchor。"""
        return self._note_anchors.get(note)


def materialize_mineru_html_wire(
    plan: MineruHtmlWirePlan,
    resources: HtmlResourceContext,
) -> list[dict[str, object]]:
    """在整棵 canonical 树验证成功后一次性解析资源并生成 raw blocks。"""
    resources.bind_anchors(ExactAnchorResolver(plan))
    stylesheet = MarkupStylesheet()
    projector = MarkupProjector(plan.root, resources, stylesheet, single_document_title=True)
    blocks: list[dict[str, object]] = []
    for spec in plan.blocks:
        if isinstance(spec, TextWireSpec):
            blocks.append(_materialize_text(spec, projector))
        elif isinstance(spec, EquationWireSpec):
            blocks.append(_materialize_equation(spec, resources))
        elif isinstance(spec, VisualWireSpec):
            blocks.extend(_materialize_visual(spec, resources, projector))
        elif isinstance(spec, ListBlockWireSpec):
            blocks.append(_materialize_list(spec.root, projector))
        elif isinstance(spec, IndexBlockWireSpec):
            blocks.append(_materialize_index(spec.root, projector))
        else:  # pragma: no cover - PageWireSpec 已穷尽。
            raise AssertionError(f"unsupported wire spec: {type(spec).__name__}")
    return blocks


def _materialize_text(spec: TextWireSpec, projector: MarkupProjector) -> dict[str, object]:
    """恢复文本、标题、页面辅助块和页面脚注。"""
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


def _materialize_equation(spec: EquationWireSpec, resources: HtmlResourceContext) -> dict[str, object]:
    """恢复行间公式裸 LaTeX 或 renderer-owned 公式图片。"""
    formula = extract_formula(spec.content_root)
    block: dict[str, object] = {
        "type": BlockType.EQUATION,
        "content": formula.latex if formula is not None else "",
    }
    if formula is None:
        block.update(_resolve_image_payload(spec.content_root, resources))
    return block


def _materialize_visual(
    spec: VisualWireSpec,
    resources: HtmlResourceContext,
    projector: MarkupProjector,
) -> list[dict[str, object]]:
    """按 renderer DOM 顺序恢复 visual body 与 annotations。"""
    blocks: list[dict[str, object]] = []
    for child in spec.children:
        if isinstance(child, AnnotationWireSpec):
            blocks.append(
                {
                    "type": child.block_type,
                    "content": _project_inline_content(projector, child.element),
                }
            )
            continue
        block = _materialize_visual_body(child, resources, projector)
        if spec.sub_type:
            block["sub_type"] = spec.sub_type
        if isinstance(child, CodeBodyWireSpec) and child.kind == "code":
            block["guess_lang"] = spec.guess_lang
        blocks.append(block)
    return blocks


def _materialize_visual_body(
    spec: RichVisualBodyWireSpec | FlowchartBodyWireSpec | TableBodyWireSpec | CodeBodyWireSpec,
    resources: HtmlResourceContext,
    projector: MarkupProjector,
) -> dict[str, object]:
    """从 typed body spec 恢复唯一载荷，不重新判断 DOM 形状。"""
    if isinstance(spec, RichVisualBodyWireSpec):
        block: dict[str, object] = {
            "type": spec.parent_type,
            "content": _restore_content(spec.content_fragment, projector),
        }
        if spec.primary_image is not None:
            block.update(_resolve_image_payload(spec.primary_image, resources))
        return block
    if isinstance(spec, FlowchartBodyWireSpec):
        block = {
            "type": BlockType.IMAGE,
            "content": _flowchart_content(spec.source_element),
        }
        if spec.fallback_image is not None:
            block.update(_resolve_image_payload(spec.fallback_image, resources))
        return block
    if isinstance(spec, TableBodyWireSpec):
        return _materialize_table(spec, resources, projector)
    return _materialize_code(spec, projector)


def _materialize_table(
    spec: TableBodyWireSpec,
    resources: HtmlResourceContext,
    projector: MarkupProjector,
) -> dict[str, object]:
    """恢复已判别的结构表格、文本、图片或空载荷。"""
    block: dict[str, object] = {"type": BlockType.TABLE, "content": ""}
    if spec.kind == "html" and spec.payload_element is not None:
        fragment = etree.Element("div")
        fragment.append(deepcopy(spec.payload_element))
        block["content"] = _restore_content(fragment, projector)
    elif spec.kind == "text" and spec.payload_element is not None:
        block["content"] = "".join(spec.payload_element.itertext())
    elif spec.kind == "image" and spec.payload_element is not None:
        block.update(_resolve_image_payload(spec.payload_element, resources))
    return block


def _materialize_code(spec: CodeBodyWireSpec, projector: MarkupProjector) -> dict[str, object]:
    """恢复普通代码文本或 algorithm 行内语义。"""
    if spec.kind == "code":
        return {
            "type": BlockType.CODE,
            "content": "".join(spec.content_element.itertext()),
        }
    return {
        "type": RAW_ALGORITHM,
        "content": _project_inline_content(projector, spec.content_element),
    }


def _materialize_list(spec: ListWireSpec, projector: MarkupProjector) -> dict[str, object]:
    """递归恢复列表叶子、marker 和嵌套 ListBlock。"""
    children: list[dict[str, object]] = []
    for child in spec.children:
        if isinstance(child, ListWireSpec):
            children.append(_materialize_list(child, projector))
            continue
        content = _project_inline_content(projector, child.content_element) if child.content_element is not None else []
        if child.marker and ({"mineru-list--reference", "mineru-list--explicit"} & spec.classes):
            content = [*text_spans(f"{child.marker} "), *content]
        block: dict[str, object] = {"type": child.block_type, "content": content}
        if child.block_index is not None:
            block["index"] = child.block_index
        children.append(block)
    block = {
        "type": BlockType.LIST,
        "attribute": "ordered" if spec.ordered else "unordered",
        "content": children,
    }
    if spec.ordered:
        block["start"] = spec.start
    if spec.sub_type:
        block["sub_type"] = spec.sub_type
    return block


def _materialize_index(spec: IndexWireSpec, projector: MarkupProjector) -> dict[str, object]:
    """递归恢复目录叶子和嵌套 IndexBlock。"""
    children: list[dict[str, object]] = []
    for child in spec.children:
        if isinstance(child, IndexWireSpec):
            nested = _materialize_index(child, projector)
            if child.block_index is not None:
                nested["index"] = child.block_index
            children.append(nested)
            continue
        children.append(_materialize_index_leaf(child, projector))
    return {"type": BlockType.INDEX, "content": children}


def _materialize_index_leaf(spec: IndexLeafWireSpec, projector: MarkupProjector) -> dict[str, object]:
    """恢复目录叶子的内容和标题元数据。"""
    content = _project_inline_content(projector, spec.content_element) if spec.content_element is not None else []
    block: dict[str, object] = {"type": spec.block_type, "content": content}
    if spec.block_type in {BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE}:
        block["anchor"] = spec.anchor
        block["level"] = spec.level or 1
    if spec.block_index is not None:
        block["index"] = spec.block_index
    return block


def _restore_content(fragment: etree._Element | None, projector: MarkupProjector) -> str:
    """按 inline 或安全富 HTML 语义恢复 visual body content。"""
    if fragment is None:
        return ""
    if not _contains_rich_markup(fragment):
        return "".join(fragment.itertext()).strip()
    clone = deepcopy(fragment)
    for carrier in list(clone.iterdescendants()):
        if not isinstance(carrier.tag, str):
            continue
        if local_name(carrier) != "math" or (carrier.get("data-block-type") or "").strip() != BlockType.EQUATION:
            continue
        formula = extract_formula(carrier)
        if formula is None or not formula.latex:
            continue
        replacement = etree.Element("eq")
        replacement.text = formula.latex
        replacement.tail = carrier.tail
        parent = carrier.getparent()
        if parent is not None:
            parent.replace(carrier, replacement)
    return _serialize_fragment(clone)


def _contains_rich_markup(fragment: etree._Element) -> bool:
    """判断片段是否需要以结构化 HTML 而非内部 inline 字符串保存。"""
    return any(
        isinstance(element.tag, str) and (local_name(element) in BLOCK_TAGS or local_name(element) in {"image", "img"})
        for element in fragment.iterdescendants()
    )


def _serialize_fragment(fragment: etree._Element) -> str:
    """把 synthetic fragment 的内容序列化为规范化安全 HTML。"""
    parts = [html.escape(fragment.text or "", quote=False)]
    parts.extend(
        etree.tostring(child, encoding="unicode", method="html", with_tail=True)
        for child in fragment
        if isinstance(child.tag, str)
    )
    return "".join(parts).strip()


def _project_inline_content(projector: MarkupProjector, element: etree._Element) -> list[dict[str, object]]:
    """直接恢复 projector 生成的结构化 Span。"""
    return projector.project_inline_content(element)


def _flowchart_content(source_element: etree._Element) -> str:
    """把 canonical Mermaid 源码恢复为标准 fence。"""
    value = "".join(source_element.itertext()).strip("\n")
    return f"```mermaid\n{value}\n```" if value else ""


def _resolve_image_payload(element: etree._Element, resources: HtmlResourceContext) -> dict[str, object]:
    """解析已由 canonical parser 选定的单个 renderer-owned 图片。"""
    resolved = resources.resolve_image(element.get("src") or "", alt=element.get("alt") or "")
    if resolved is None:
        return {}
    if resolved.image_base64:
        return {"image_base64": resolved.image_base64}
    if resolved.image_url:
        return {"image_url": resolved.image_url}
    return {}


__all__ = ["ExactAnchorResolver", "materialize_mineru_html_wire"]
