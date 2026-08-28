# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到单正文 EPUB 3.3 的静态 XHTML renderer。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from functools import lru_cache
from importlib import resources
import re
from urllib.parse import quote, unquote, urlsplit
from uuid import NAMESPACE_URL, uuid5

from bs4 import BeautifulSoup, NavigableString, Tag
from bs4.element import Comment, Doctype, ProcessingInstruction
from latex2mathml.converter import convert as latex_to_mathml
from lxml import etree

from ....backend.postprocess.inline import inline_plain_text, join_inline_spans, normalize_inline_spans
from ....types import (
    PAGE_AUXILIARY_BLOCK_TYPES,
    RAW_ALGORITHM,
    AlgorithmBodyBlock,
    BlockBase,
    BlockType,
    ChartAnnotationBlock,
    ChartBlock,
    ChartBodyBlock,
    CodeAnnotationBlock,
    CodeBlock,
    CodeBodyBlock,
    CodeInlineSpan,
    DocTitleBlock,
    EquationBlock,
    EquationInlineSpan,
    HyperlinkSpan,
    ImageAnnotationBlock,
    ImageBlock,
    ImageBodyBlock,
    IndexBlock,
    InlineSpan,
    ListBlock,
    MiddleJson,
    PageFootnoteBlock,
    ParagraphTitleBlock,
    RefTextBlock,
    TableAnnotationBlock,
    TableBlock,
    TableBodyBlock,
    TextBlock,
    TextSpan,
    TitleBlockBase,
)
from ...contracts import AssetResolver, EpubRenderOptions
from ..common.index import strip_index_page_tail
from ..common.list_items import ListItem, parse_list_item_marker, reference_list_needs_bullets
from ..common.planner import PlannedBlock, build_render_plan
from .assets import EpubAssetRegistry
from .package import EpubMetadata, NavigationItem, build_epub_package

_EPUB_NS = "http://www.idpf.org/2007/ops"
_MATHML_NS = "http://www.w3.org/1998/Math/MathML"
_XHTML_NS = "http://www.w3.org/1999/xhtml"
_XML_NS = "http://www.w3.org/XML/1998/namespace"
_STYLE_RESOURCE_NAME = "mineru.css"
_INVALID_XML_TEXT_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]")
_MARKUP_TOKEN_RE = re.compile(
    r"<\s*(?P<closing>/)?\s*(?P<name>[A-Za-z][A-Za-z0-9:-]*)\b(?P<attrs>[^>]*)>",
    re.DOTALL,
)
_SAFE_LANGUAGE_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,31}\Z")
_ALLOWED_MARKUP_TAGS = {
    "a",
    "b",
    "blockquote",
    "br",
    "caption",
    "code",
    "col",
    "colgroup",
    "details",
    "div",
    "em",
    "eq",
    "i",
    "img",
    "kbd",
    "li",
    "mark",
    "ol",
    "p",
    "pre",
    "s",
    "span",
    "strong",
    "sub",
    "summary",
    "summary",
    "sup",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "u",
    "ul",
}
_DROP_CONTENT_TAGS = {
    "audio",
    "button",
    "canvas",
    "embed",
    "form",
    "head",
    "iframe",
    "input",
    "math",
    "noscript",
    "object",
    "script",
    "select",
    "style",
    "svg",
    "template",
    "textarea",
    "video",
}
_SOURCE_MARKUP_TAGS = _ALLOWED_MARKUP_TAGS | _DROP_CONTENT_TAGS
_VOID_MARKUP_TAGS = {"br", "col", "img"}
_PHRASING_MARKUP_TAGS = {
    "a",
    "b",
    "code",
    "em",
    "i",
    "kbd",
    "mark",
    "p",
    "s",
    "span",
    "strong",
    "sub",
    "sup",
    "u",
}
_BLOCK_MARKUP_TAGS = {"blockquote", "details", "div", "ol", "p", "pre", "table", "ul"}
_TABLE_PARENT_RULES = {
    "caption": {"table"},
    "col": {"colgroup"},
    "colgroup": {"table"},
    "tbody": {"table"},
    "td": {"tr"},
    "tfoot": {"table"},
    "th": {"tr"},
    "thead": {"table"},
    "tr": {"table", "tbody", "tfoot", "thead"},
}


@dataclass(frozen=True, slots=True)
class _TitleTarget:
    """保存正文标题的可见文本、层级与 XHTML 目标。"""

    title: str
    level: int
    target_id: str


class _AnchorRegistry:
    """为正文标题和页面脚注分配文档级唯一 XHTML id。"""

    def __init__(self, middle_json: MiddleJson) -> None:
        """按页面与 block 顺序建立目标、来源 anchor 和标题索引。"""
        self._block_targets: dict[tuple[int, int, str], str] = {}
        self._anchor_targets: dict[str, str] = {}
        self._footnote_targets: set[str] = set()
        self.title_targets: list[_TitleTarget] = []
        used_ids: set[str] = {"content-start"}
        heading_position = 0
        footnote_position = 0
        for page in middle_json.pages:
            for block in page.blocks:
                if isinstance(block, TitleBlockBase):
                    visible = inline_plain_text(block.content).strip()
                    if not visible:
                        continue
                    heading_position += 1
                    target_id = _allocate_target_id(
                        block.anchor,
                        fallback=f"heading-{heading_position}",
                        used_ids=used_ids,
                    )
                    self.title_targets.append(_TitleTarget(visible, block.level, target_id))
                elif isinstance(block, PageFootnoteBlock):
                    visible = inline_plain_text(block.content).strip()
                    if not visible:
                        continue
                    footnote_position += 1
                    target_id = _allocate_target_id(
                        block.anchor,
                        fallback=f"footnote-{footnote_position}",
                        used_ids=used_ids,
                    )
                    self._footnote_targets.add(target_id)
                else:
                    continue
                assert block.index is not None
                self._block_targets[(page.page_idx, block.index, str(block.type))] = target_id
                anchor_key = _anchor_key(block.anchor)
                if anchor_key and anchor_key not in self._anchor_targets:
                    self._anchor_targets[anchor_key] = target_id

    def target_for_block(self, page_idx: int, block: BlockBase) -> str | None:
        """按来源页、index 和类型返回标题或脚注的唯一目标。"""
        if block.index is None:
            return None
        return self._block_targets.get((page_idx, block.index, str(block.type)))

    def target_for_anchor(self, anchor: str | None) -> str | None:
        """按 producer anchor 返回首次匹配的正文目标。"""
        return self._anchor_targets.get(_anchor_key(anchor))

    def is_footnote_target(self, target_id: str) -> bool:
        """判断目标是否对应页面脚注，以便标注 noteref 语义。"""
        return target_id in self._footnote_targets


class _EpubXhtmlRenderer:
    """维护单个 EPUB 正文的锚点、素材与 MathML 状态。"""

    def __init__(
        self,
        middle_json: MiddleJson,
        *,
        metadata: EpubMetadata,
        assets: EpubAssetRegistry,
        anchors: _AnchorRegistry,
    ) -> None:
        """保存严格输入和已规范化的调用状态。"""
        self.middle_json = middle_json
        self.metadata = metadata
        self.assets = assets
        self.anchors = anchors
        self.has_mathml = False

    def render(self) -> bytes:
        """把完整 render plan 写成一个无脚本 XHTML content document。"""
        root = etree.Element(
            _xhtml("html"),
            nsmap={None: _XHTML_NS, "epub": _EPUB_NS},
            attrib={f"{{{_XML_NS}}}lang": self.metadata.language, "lang": self.metadata.language},
        )
        head = etree.SubElement(root, _xhtml("head"))
        etree.SubElement(head, _xhtml("meta"), charset="utf-8")
        title = etree.SubElement(head, _xhtml("title"))
        title.text = self.metadata.title
        etree.SubElement(
            head,
            _xhtml("link"),
            rel="stylesheet",
            href="../styles/mineru.css",
            type="text/css",
        )
        body = etree.SubElement(root, _xhtml("body"), attrib={"class": "mineru-epub-body"})
        article = etree.SubElement(
            body,
            _xhtml("article"),
            id="content-start",
            attrib={"class": "mineru-document"},
        )
        self._render_pages(article, build_render_plan(self.middle_json))
        return etree.tostring(
            root,
            encoding="utf-8",
            xml_declaration=True,
            doctype="<!DOCTYPE html>",
        )

    def _render_pages(self, parent: etree._Element, pages: list[list[PlannedBlock]]) -> None:
        """把默认计划展平到单个连续阅读容器。"""
        for page in pages:
            for planned in page:
                rendered = self._render_planned_block(planned)
                if rendered is not None:
                    parent.append(rendered)

    def _render_planned_block(self, planned: PlannedBlock) -> etree._Element | None:
        """过滤计划块、分派具体类型并追加稳定来源属性。"""
        if planned.removed:
            return None
        block = planned.block
        if block.type in PAGE_AUXILIARY_BLOCK_TYPES:
            return None
        content = self._render_block_content(planned)
        if content is None:
            return None
        wrapper = etree.Element(
            _xhtml("div"),
            attrib={
                "class": "mineru-block",
                "data-page-idx": str(planned.page_idx),
                "data-block-type": str(block.type),
            },
        )
        if block.index is not None:
            wrapper.set("data-block-index", str(block.index))
        wrapper.append(content)
        return wrapper

    def _render_block_content(self, planned: PlannedBlock) -> etree._Element | None:
        """把一个具体 PageBlock 映射为静态 XHTML 元素。"""
        block = planned.block
        if isinstance(block, (TextBlock, RefTextBlock)):
            content = join_inline_spans(planned.text_contents or [block.content])
            paragraph = etree.Element(
                _xhtml("p"),
                attrib={"class": "mineru-ref-text" if isinstance(block, RefTextBlock) else "mineru-text"},
            )
            self._append_inline_spans(paragraph, content)
            return paragraph if _has_visible_content(paragraph) else None
        if isinstance(block, (DocTitleBlock, ParagraphTitleBlock)):
            return self._render_title(planned.page_idx, block)
        if isinstance(block, PageFootnoteBlock):
            return self._render_page_footnote(planned.page_idx, block)
        if isinstance(block, EquationBlock):
            return self._render_equation(block)
        if isinstance(block, ListBlock):
            return self._render_list(block)
        if isinstance(block, IndexBlock):
            return self._render_index(block)
        if isinstance(block, ImageBlock):
            return self._render_image_block(block)
        if isinstance(block, TableBlock):
            return self._render_table_block(block)
        if isinstance(block, ChartBlock):
            return self._render_chart_block(block)
        if isinstance(block, CodeBlock):
            return self._render_code_block(block)
        raise TypeError(f"Unsupported PageBlock type: {type(block).__name__}")

    def _render_title(self, page_idx: int, block: TitleBlockBase) -> etree._Element | None:
        """渲染带文档级唯一 id 的 h1-h6 标题。"""
        if not inline_plain_text(block.content).strip():
            return None
        level = min(max(block.level, 1), 6)
        heading = etree.Element(_xhtml(f"h{level}"), attrib={"class": f"mineru-heading mineru-heading--{level}"})
        target_id = self.anchors.target_for_block(page_idx, block)
        if target_id:
            heading.set("id", target_id)
        self._append_inline_spans(heading, block.content)
        return heading

    def _render_page_footnote(self, page_idx: int, block: PageFootnoteBlock) -> etree._Element | None:
        """把页面脚注保留为 EPUB footnote aside。"""
        footnote = etree.Element(
            _xhtml("aside"),
            attrib={"class": "mineru-page-footnote", f"{{{_EPUB_NS}}}type": "footnote", "role": "doc-footnote"},
        )
        target_id = self.anchors.target_for_block(page_idx, block)
        if target_id:
            footnote.set("id", target_id)
        self._append_inline_spans(footnote, block.content)
        return footnote if _has_visible_content(footnote) else None

    def _render_equation(self, block: EquationBlock) -> etree._Element | None:
        """优先渲染行间 MathML，空公式时才尝试包内图片。"""
        container = etree.Element(_xhtml("div"), attrib={"class": "mineru-equation"})
        if block.content.strip():
            self._append_math(container, block.content, display="block")
        elif source := self.assets.resolve_block(block):
            etree.SubElement(container, _xhtml("img"), src=source, alt="formula")
        return container if _has_visible_content(container) else None

    def _render_list(self, block: ListBlock) -> etree._Element | None:
        """按共享 marker 语义递归渲染原生有序、无序或显式 marker 列表。"""
        parsed_leaves = [
            parse_list_item_marker(child.content)
            for child in block.content
            if not isinstance(child, ListBlock) and inline_plain_text(child.content).strip()
        ]
        add_reference_bullets = reference_list_needs_bullets(block)
        container_tag, list_type, class_name = _classify_list(parsed_leaves, add_reference_bullets)
        container = etree.Element(_xhtml(container_tag), attrib={"class": f"mineru-list {class_name}"})
        if list_type:
            container.set("type", list_type)
        if container_tag == "ol" and parsed_leaves and parsed_leaves[0].value not in (None, 1):
            container.set("start", str(parsed_leaves[0].value))
        expected_value: int | None = None
        last_item: etree._Element | None = None
        for child in block.content:
            if isinstance(child, ListBlock):
                nested = self._render_list(child)
                if nested is None:
                    continue
                if last_item is None:
                    last_item = etree.SubElement(container, _xhtml("li"), attrib={"class": "mineru-list-item--orphan"})
                last_item.append(nested)
                continue
            parsed = parse_list_item_marker(child.content)
            item_content, marker = _list_item_content(
                parsed,
                add_reference_bullets,
                explicit_markers=class_name == "mineru-list--explicit",
            )
            item = etree.SubElement(container, _xhtml("li"))
            if class_name == "mineru-list--explicit":
                item.set("class", "mineru-list-item--explicit")
            if container_tag == "ol" and parsed.kind == "ordered" and parsed.value is not None:
                if expected_value is None:
                    expected_value = parsed.value
                if parsed.value != expected_value:
                    item.set("value", str(parsed.value))
                expected_value = parsed.value + 1
            if marker or class_name == "mineru-list--explicit":
                marker_element = etree.SubElement(item, _xhtml("span"), attrib={"class": "mineru-list-marker"})
                marker_element.text = marker or ""
            content_element = etree.SubElement(item, _xhtml("span"), attrib={"class": "mineru-list-content"})
            self._append_inline_spans(content_element, item_content)
            last_item = item
        return container if len(container) else None

    def _render_index(self, block: IndexBlock) -> etree._Element | None:
        """把源目录保留为正文内导航，并只链接到真实正文目标。"""
        navigation = etree.Element(_xhtml("nav"), attrib={"class": "mineru-index", "aria-label": "Table of contents"})
        listing = etree.SubElement(navigation, _xhtml("ul"))
        self._append_index_children(listing, block)
        return navigation if len(listing) else None

    def _append_index_children(self, parent: etree._Element, block: IndexBlock) -> None:
        """递归渲染 IndexBlock，并把孤立嵌套目录提升到当前层级。"""
        last_item: etree._Element | None = None
        for child in block.content:
            if isinstance(child, IndexBlock):
                nested = etree.Element(_xhtml("ul"))
                self._append_index_children(nested, child)
                if not len(nested):
                    continue
                if last_item is None:
                    last_item = etree.SubElement(parent, _xhtml("li"), attrib={"class": "mineru-list-item--orphan"})
                last_item.append(nested)
                continue
            content = strip_index_page_tail(child.content)
            if not inline_plain_text(content).strip():
                continue
            item = etree.SubElement(parent, _xhtml("li"))
            target = self.anchors.target_for_anchor(child.anchor) if isinstance(child, TitleBlockBase) else None
            inline_parent = item
            if target:
                inline_parent = etree.SubElement(item, _xhtml("a"), href=f"#{quote(target, safe='-._~')}")
            self._append_inline_spans(inline_parent, content)
            last_item = item

    def _render_image_block(self, block: ImageBlock) -> etree._Element | None:
        """按子块顺序渲染图片主体及其标题、脚注。"""
        figure = etree.Element(_xhtml("figure"), attrib={"class": "mineru-figure mineru-figure--image"})
        for child in block.content:
            rendered = (
                self._render_image_body(block, child) if isinstance(child, ImageBodyBlock) else self._render_annotation(child)
            )
            if rendered is not None:
                figure.append(rendered)
        return figure if len(figure) else None

    def _render_image_body(self, parent: ImageBlock, block: ImageBodyBlock) -> etree._Element | None:
        """渲染包内图片，并在缺图时保留已有结构或可见文字。"""
        container = etree.Element(_xhtml("div"), attrib={"class": "mineru-visual-body mineru-visual-body--image"})
        source = self.assets.resolve_block(block)
        if source:
            alt = _plain_content_text(block.content) or parent.sub_type or "image"
            etree.SubElement(container, _xhtml("img"), src=source, alt=alt, attrib={"class": "mineru-image"})
        if block.content.strip():
            content = etree.Element(_xhtml("div"), attrib={"class": "mineru-image-content"})
            self._append_rich_or_text(content, block.content)
            if _has_visible_content(content):
                container.append(content)
        return container if _has_visible_content(container) else None

    def _render_table_block(self, block: TableBlock) -> etree._Element | None:
        """按子块顺序渲染结构表格、图片回退及说明。"""
        figure = etree.Element(_xhtml("figure"), attrib={"class": "mineru-figure mineru-figure--table"})
        for child in block.content:
            rendered = self._render_table_body(child) if isinstance(child, TableBodyBlock) else self._render_annotation(child)
            if rendered is not None:
                figure.append(rendered)
        return figure if len(figure) else None

    def _render_table_body(self, block: TableBodyBlock) -> etree._Element | None:
        """优先输出安全结构内容，无内容时尝试整体表格图片。"""
        container = etree.Element(_xhtml("div"), attrib={"class": "mineru-visual-body mineru-visual-body--table"})
        if block.content.strip():
            if _is_supported_markup(block.content):
                self._append_markup(container, block.content)
            else:
                pre = etree.SubElement(container, _xhtml("pre"), attrib={"class": "mineru-table-text"})
                pre.text = _normalize_xml_text(block.content)
        if not _has_visible_content(container) and (source := self.assets.resolve_block(block)):
            etree.SubElement(container, _xhtml("img"), src=source, alt="table", attrib={"class": "mineru-table-image"})
        return container if _has_visible_content(container) else None

    def _render_chart_block(self, block: ChartBlock) -> etree._Element | None:
        """按子块顺序渲染图表图片、结构内容及说明。"""
        figure = etree.Element(_xhtml("figure"), attrib={"class": "mineru-figure mineru-figure--chart"})
        for child in block.content:
            rendered = (
                self._render_chart_body(block, child) if isinstance(child, ChartBodyBlock) else self._render_annotation(child)
            )
            if rendered is not None:
                figure.append(rendered)
        return figure if len(figure) else None

    def _render_chart_body(self, parent: ChartBlock, block: ChartBodyBlock) -> etree._Element | None:
        """渲染包内图表图片，并始终保留并存结构内容。"""
        container = etree.Element(_xhtml("div"), attrib={"class": "mineru-visual-body mineru-visual-body--chart"})
        if source := self.assets.resolve_block(block):
            etree.SubElement(
                container,
                _xhtml("img"),
                src=source,
                alt=parent.sub_type or "chart",
                attrib={"class": "mineru-chart-image"},
            )
        if block.content.strip():
            content = etree.Element(_xhtml("div"), attrib={"class": "mineru-chart-content"})
            self._append_rich_or_text(content, block.content, preformatted=True)
            if _has_visible_content(content):
                container.append(content)
        return container if _has_visible_content(container) else None

    def _render_code_block(self, block: CodeBlock) -> etree._Element | None:
        """按子块顺序渲染静态代码、算法及其说明。"""
        figure = etree.Element(_xhtml("figure"), attrib={"class": "mineru-figure mineru-figure--code"})
        for child in block.content:
            if isinstance(child, (CodeBodyBlock, AlgorithmBodyBlock)):
                rendered = self._render_code_body(block, child)
            else:
                rendered = self._render_annotation(child)
            if rendered is not None:
                figure.append(rendered)
        return figure if len(figure) else None

    def _render_code_body(self, parent: CodeBlock, block: CodeBodyBlock | AlgorithmBodyBlock) -> etree._Element:
        """代码使用 pre/code，算法使用保留换行的结构化 Span。"""
        container = etree.Element(_xhtml("div"), attrib={"class": "mineru-visual-body mineru-visual-body--code"})
        if parent.sub_type == BlockType.CODE:
            if not isinstance(block, CodeBodyBlock):
                raise TypeError("code subtype requires CodeBodyBlock")
            pre = etree.SubElement(container, _xhtml("pre"), attrib={"class": "mineru-code"})
            code = etree.SubElement(pre, _xhtml("code"))
            language = _normalize_code_language(parent.guess_lang)
            if language:
                code.set("class", f"language-{language}")
            code.text = _normalize_xml_text(block.content)
            return container
        if parent.sub_type == RAW_ALGORITHM:
            if not isinstance(block, AlgorithmBodyBlock):
                raise TypeError("algorithm subtype requires AlgorithmBodyBlock")
            algorithm = etree.SubElement(container, _xhtml("div"), attrib={"class": "mineru-algorithm"})
            self._append_inline_spans(algorithm, block.content, preserve_newlines=True, separate_adjacent_math=True)
            return container
        raise ValueError(f"Unsupported code subtype: {parent.sub_type}")

    def _render_annotation(
        self,
        block: ImageAnnotationBlock | TableAnnotationBlock | ChartAnnotationBlock | CodeAnnotationBlock,
    ) -> etree._Element | None:
        """按 caption 或 footnote 语义渲染视觉说明。"""
        role = "mineru-caption" if str(block.type).endswith("caption") else "mineru-footnote"
        annotation = etree.Element(
            _xhtml("p"),
            attrib={"class": f"{role} {role}--{str(block.type).replace('_', '-')}"},
        )
        self._append_inline_spans(annotation, block.content)
        return annotation if _has_visible_content(annotation) else None

    def _append_inline_spans(
        self,
        parent: etree._Element,
        spans: list[InlineSpan],
        *,
        preserve_newlines: bool = False,
        separate_adjacent_math: bool = False,
    ) -> None:
        """按结构化 Span 顺序向 XHTML mixed content 追加安全节点。"""
        previous_was_math = False
        for span in spans:
            current_is_math = isinstance(span, EquationInlineSpan)
            if separate_adjacent_math and previous_was_math and current_is_math:
                _append_text(parent, " ", preserve_newlines=True)
            self._append_inline_span(parent, span, preserve_newlines=preserve_newlines)
            previous_was_math = current_is_math

    def _append_inline_span(self, parent: etree._Element, span: InlineSpan, *, preserve_newlines: bool) -> None:
        """把单个 Text/Code/Equation/Hyperlink Span 追加到父节点。"""
        if isinstance(span, TextSpan):
            target = _append_text_style_container(parent, span)
            _append_text(target, span.content, preserve_newlines=preserve_newlines)
            return
        if isinstance(span, CodeInlineSpan):
            code = etree.SubElement(parent, _xhtml("code"))
            _append_text(code, span.content, preserve_newlines=preserve_newlines)
            return
        if isinstance(span, EquationInlineSpan):
            self._append_math(parent, span.content, display="inline")
            return
        if isinstance(span, HyperlinkSpan):
            href, target_id = self._resolve_link(span.url)
            link_parent = parent
            if href:
                link_parent = etree.SubElement(parent, _xhtml("a"), href=href)
                if target_id and self.anchors.is_footnote_target(target_id):
                    link_parent.set(f"{{{_EPUB_NS}}}type", "noteref")
                    link_parent.set("role", "doc-noteref")
            self._append_inline_spans(link_parent, list(span.content), preserve_newlines=preserve_newlines)
            return
        raise TypeError(f"Unsupported inline span: {type(span).__name__}")

    def _append_math(self, parent: etree._Element, latex: str, *, display: str) -> None:
        """追加 Presentation MathML，并在转换失败时显示原始 LaTeX。"""
        normalized = latex.strip()
        if not normalized:
            return
        try:
            markup = latex_to_mathml(normalized, display=display)
            parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True, recover=False, huge_tree=False)
            math = etree.fromstring(markup.encode("utf-8"), parser=parser)
            if math.tag != f"{{{_MATHML_NS}}}math":
                raise ValueError("latex2mathml did not return a MathML root")
        except Exception:
            fallback = etree.SubElement(
                parent,
                _xhtml("code"),
                attrib={"class": f"mineru-latex-fallback mineru-latex-fallback--{display}"},
            )
            fallback.text = _normalize_xml_text(normalized)
            return
        parent.append(math)
        self.has_mathml = True

    def _append_rich_or_text(self, parent: etree._Element, content: str, *, preformatted: bool = False) -> None:
        """识别安全富 HTML，否则按普通文本或预格式文本输出。"""
        if _is_supported_markup(content):
            self._append_markup(parent, content)
            return
        if preformatted:
            pre = etree.SubElement(parent, _xhtml("pre"))
            pre.text = _normalize_xml_text(content)
        else:
            _append_text(parent, content)

    def _append_markup(self, parent: etree._Element, markup: str) -> None:
        """通过 EPUB 专用 allowlist 把不可信 HTML 转为安全 XHTML 节点。"""
        soup = BeautifulSoup(_normalize_xml_text(markup), "html.parser")
        for child in list(soup.contents):
            self._append_soup_node(parent, child)

    def _append_soup_node(self, parent: etree._Element, node: object) -> None:
        """递归复制一个 BeautifulSoup 节点，仅创建允许的 XHTML 结构。"""
        if isinstance(node, (Comment, Doctype, ProcessingInstruction)):
            return
        if isinstance(node, NavigableString):
            _append_text(parent, str(node), preserve_newlines=True)
            return
        if not isinstance(node, Tag):
            return
        name = (node.name or "").lower()
        if name in _DROP_CONTENT_TAGS:
            return
        if name not in _ALLOWED_MARKUP_TAGS:
            for child in list(node.children):
                self._append_soup_node(parent, child)
            return
        parent_name = etree.QName(parent).localname
        if name in _TABLE_PARENT_RULES and parent_name not in _TABLE_PARENT_RULES[name]:
            for child in list(node.children):
                self._append_soup_node(parent, child)
            return
        if parent_name in _PHRASING_MARKUP_TAGS and name in _BLOCK_MARKUP_TAGS:
            for child in list(node.children):
                self._append_soup_node(parent, child)
            return
        if name == "a" and parent_name == "a":
            for child in list(node.children):
                self._append_soup_node(parent, child)
            return
        if name == "li" and parent_name not in {"ol", "ul"}:
            if parent_name in _PHRASING_MARKUP_TAGS:
                for child in list(node.children):
                    self._append_soup_node(parent, child)
                return
            listing = etree.SubElement(parent, _xhtml("ul"))
            item = etree.SubElement(listing, _xhtml("li"), attrib=_safe_markup_attributes(name, node))
            for child in list(node.children):
                self._append_soup_node(item, child)
            return
        if name in {"ol", "ul"}:
            listing = etree.SubElement(parent, _xhtml(name), attrib=_safe_markup_attributes(name, node))
            for child in list(node.children):
                if isinstance(child, NavigableString) and not str(child).strip():
                    continue
                if isinstance(child, Tag) and (child.name or "").lower() == "li":
                    self._append_soup_node(listing, child)
                    continue
                item = etree.SubElement(listing, _xhtml("li"))
                self._append_soup_node(item, child)
            return
        if name == "eq":
            self._append_math(parent, node.get_text(), display="inline")
            return
        if name == "img":
            source = self.assets.resolve_embedded_source(_attribute_text(node.get("src")))
            alt = _attribute_text(node.get("alt"))
            if source:
                image = etree.SubElement(parent, _xhtml("img"), src=source, alt=_normalize_xml_text(alt))
                title = _attribute_text(node.get("title"))
                if title:
                    image.set("title", _normalize_xml_text(title))
            elif alt:
                _append_text(parent, alt)
            return
        if name == "a":
            href, target_id = self._resolve_link(_attribute_text(node.get("href")))
            target_parent = parent
            if href:
                target_parent = etree.SubElement(parent, _xhtml("a"), href=href)
                title = _attribute_text(node.get("title"))
                if title:
                    target_parent.set("title", _normalize_xml_text(title))
                if target_id and self.anchors.is_footnote_target(target_id):
                    target_parent.set(f"{{{_EPUB_NS}}}type", "noteref")
                    target_parent.set("role", "doc-noteref")
            for child in list(node.children):
                self._append_soup_node(target_parent, child)
            return
        attributes = _safe_markup_attributes(name, node)
        if name == "colgroup" and any(isinstance(child, Tag) and child.name == "col" for child in node.children):
            attributes.pop("span", None)
        element = etree.SubElement(parent, _xhtml(name), attrib=attributes)
        if name not in _VOID_MARKUP_TAGS:
            for child in list(node.children):
                self._append_soup_node(element, child)

    def _resolve_link(self, url: str) -> tuple[str | None, str | None]:
        """保留安全外链或已登记 fragment，删除无包内目标的相对链接。"""
        normalized = _normalize_xml_text(url).strip()
        if not normalized or normalized.startswith(("//", "\\")):
            return None, None
        if normalized.startswith("#"):
            target = self.anchors.target_for_anchor(unquote(normalized[1:]))
            if target:
                return f"#{quote(target, safe='-._~')}", target
            return None, None
        try:
            parsed = urlsplit(normalized)
            _ = parsed.port
        except ValueError:
            return None, None
        scheme = parsed.scheme.casefold()
        if scheme in {"http", "https"}:
            if not parsed.netloc or parsed.hostname is None or parsed.username is not None or parsed.password is not None:
                return None, None
        elif scheme in {"mailto", "tel"}:
            if not parsed.path:
                return None, None
        else:
            return None, None
        return quote(normalized, safe="/:#?&=%@+~,;!$'*-._"), None


def render_epub(
    middle_json: MiddleJson,
    *,
    title: str | None = None,
    authors: tuple[str, ...] = (),
    language: str = "und",
    identifier: str | None = None,
    modified_at: datetime | None = None,
    asset_resolver: AssetResolver | None = None,
) -> bytes:
    """把严格 MiddleJson 无副作用地渲染为单正文 EPUB 3.3 字节。"""
    if not isinstance(middle_json, MiddleJson):
        raise TypeError("render_epub expects a MiddleJson instance")
    options = EpubRenderOptions(
        title=title,
        authors=authors,
        language=language,
        identifier=identifier,
        modified_at=modified_at,
        asset_resolver=asset_resolver,
    )
    resolved_title = _resolve_document_title(middle_json, options.title)
    resolved_authors = tuple(_normalize_xml_text(author).strip() for author in options.authors)
    resolved_language = options.language.strip()
    resolved_identifier = (
        options.identifier.strip()
        if options.identifier
        else _stable_identifier(
            middle_json,
            title=resolved_title,
            authors=resolved_authors,
            language=resolved_language,
        )
    )
    resolved_modified = (options.modified_at or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)
    if resolved_modified.year < 1000:
        raise ValueError("modified_at UTC year must use four digits")
    metadata = EpubMetadata(
        title=resolved_title,
        authors=resolved_authors,
        language=resolved_language,
        identifier=_normalize_xml_text(resolved_identifier),
        modified_at=resolved_modified,
    )
    anchors = _AnchorRegistry(middle_json)
    assets = EpubAssetRegistry(options.asset_resolver)
    renderer = _EpubXhtmlRenderer(
        middle_json,
        metadata=metadata,
        assets=assets,
        anchors=anchors,
    )
    content_xhtml = renderer.render()
    navigation = _build_navigation(middle_json, anchors, resolved_title)
    return build_epub_package(
        metadata=metadata,
        content_xhtml=content_xhtml,
        navigation=navigation,
        stylesheet=_load_epub_stylesheet(),
        assets=assets.assets,
        has_mathml=renderer.has_mathml,
    )


def _build_navigation(middle_json: MiddleJson, anchors: _AnchorRegistry, document_title: str) -> list[NavigationItem]:
    """优先使用有效 IndexBlock，否则按标题层级或正文起点生成 toc。"""
    for page in middle_json.pages:
        for block in page.blocks:
            if not isinstance(block, IndexBlock):
                continue
            items = _navigation_from_index(block, anchors)
            if items:
                return items
    if anchors.title_targets:
        roots: list[NavigationItem] = []
        stack: list[tuple[int, NavigationItem]] = []
        for target in anchors.title_targets:
            item = NavigationItem(
                title=target.title,
                href=f"text/content.xhtml#{quote(target.target_id, safe='-._~')}",
            )
            while stack and stack[-1][0] >= target.level:
                stack.pop()
            if stack:
                stack[-1][1].children.append(item)
            else:
                roots.append(item)
            stack.append((target.level, item))
        return roots
    return [NavigationItem(title=document_title, href="text/content.xhtml#content-start")]


def _navigation_from_index(block: IndexBlock, anchors: _AnchorRegistry) -> list[NavigationItem]:
    """从一个 IndexBlock 提取仅包含真实标题目标的层级导航。"""
    result: list[NavigationItem] = []
    last_item: NavigationItem | None = None
    for child in block.content:
        if isinstance(child, IndexBlock):
            nested = _navigation_from_index(child, anchors)
            if last_item is not None:
                last_item.children.extend(nested)
            else:
                result.extend(nested)
            continue
        if not isinstance(child, TitleBlockBase):
            continue
        target = anchors.target_for_anchor(child.anchor)
        title = inline_plain_text(strip_index_page_tail(child.content)).strip()
        if not target or not title:
            continue
        last_item = NavigationItem(
            title=title,
            href=f"text/content.xhtml#{quote(target, safe='-._~')}",
        )
        result.append(last_item)
    return result


def _classify_list(items: list[ListItem], add_reference_bullets: bool) -> tuple[str, str | None, str]:
    """根据直属 marker 选择原生列表类型或显式 marker 模式。"""
    if add_reference_bullets:
        return "ul", None, "mineru-list--reference"
    if items and all(item.kind == "unordered" for item in items):
        return "ul", None, "mineru-list--unordered"
    if items and all(item.kind == "ordered" for item in items):
        styles = {item.ordered_style for item in items}
        if len(styles) == 1:
            list_type = {
                "lower-alpha": "a",
                "upper-alpha": "A",
                "lower-roman": "i",
                "upper-roman": "I",
            }.get(next(iter(styles)) or "")
            return "ol", list_type, "mineru-list--ordered"
    if items and all(item.kind == "none" for item in items):
        return "ul", None, "mineru-list--unmarked"
    return "ul", None, "mineru-list--explicit"


def _list_item_content(
    item: ListItem,
    add_reference_bullets: bool,
    *,
    explicit_markers: bool,
) -> tuple[list[InlineSpan], str | None]:
    """决定列表项应剥离、保留还是显式显示源 marker。"""
    if add_reference_bullets:
        if item.kind == "unordered":
            return item.body, None
        prefix = f"{item.leading}{item.marker or ''}{item.separator}"
        original = normalize_inline_spans([TextSpan(type="text", content=prefix), *item.body]) if prefix else item.body
        return original, None
    if explicit_markers:
        return item.body, item.marker
    if item.kind in {"unordered", "ordered"}:
        return item.body, None
    return item.body, item.marker


def _append_text_style_container(parent: etree._Element, span: TextSpan) -> etree._Element:
    """按固定样式顺序创建 TextSpan 的 XHTML 包装节点。"""
    target = parent
    if _needs_whitespace_preservation(span.content):
        target = etree.SubElement(target, _xhtml("span"), attrib={"class": "mineru-preserve-whitespace"})
    wrappers: list[tuple[str, dict[str, str]]] = []
    if "emphasis" in span.styles:
        wrappers.append(("span", {"class": "mineru-text-emphasis"}))
    if "strikethrough" in span.styles:
        wrappers.append(("s", {}))
    if "italic" in span.styles:
        wrappers.append(("em", {}))
    if "bold" in span.styles:
        wrappers.append(("strong", {}))
    if "underline" in span.styles:
        wrappers.append(("u", {}))
    if "superscript" in span.styles:
        wrappers.append(("sup", {}))
    elif "subscript" in span.styles:
        wrappers.append(("sub", {}))
    for tag, attributes in wrappers:
        target = etree.SubElement(target, _xhtml(tag), attrib=attributes)
    return target


def _append_text(parent: etree._Element, content: str, *, preserve_newlines: bool = False) -> None:
    """向 mixed content 追加安全文本，并按需把换行转换为 br。"""
    normalized = _normalize_xml_text(content).replace("\r\n", "\n").replace("\r", "\n")
    if preserve_newlines:
        _append_raw_text(parent, normalized)
        return
    parts = normalized.split("\n")
    for position, part in enumerate(parts):
        if position:
            parent.append(etree.Element(_xhtml("br")))
        _append_raw_text(parent, part)


def _append_raw_text(parent: etree._Element, content: str) -> None:
    """在不破坏既有子节点 tail 的前提下追加一段普通文本。"""
    if not content:
        return
    if len(parent):
        child = parent[-1]
        child.tail = f"{child.tail or ''}{content}"
    else:
        parent.text = f"{parent.text or ''}{content}"


def _safe_markup_attributes(name: str, tag: Tag) -> dict[str, str]:
    """只保留表格和列表语义需要的有界属性。"""
    attributes: dict[str, str] = {}
    if name in {"td", "th"}:
        for attribute in ("colspan", "rowspan"):
            if value := _bounded_integer(_attribute_text(tag.get(attribute)), minimum=1, maximum=1000):
                attributes[attribute] = value
        scope = _attribute_text(tag.get("scope"))
        if name == "th" and scope in {"col", "colgroup", "row", "rowgroup"}:
            attributes["scope"] = scope
    elif name in {"col", "colgroup"}:
        if value := _bounded_integer(_attribute_text(tag.get("span")), minimum=1, maximum=1000):
            attributes["span"] = value
    elif name == "ol":
        if value := _bounded_integer(_attribute_text(tag.get("start")), minimum=-1_000_000, maximum=1_000_000):
            attributes["start"] = value
    elif name == "li":
        if value := _bounded_integer(_attribute_text(tag.get("value")), minimum=-1_000_000, maximum=1_000_000):
            attributes["value"] = value
    return attributes


def _bounded_integer(value: str, *, minimum: int, maximum: int) -> str | None:
    """把十进制属性约束到 EPUB renderer 支持的闭区间。"""
    if re.fullmatch(r"[+-]?\d+", value) is None:
        return None
    number = int(value)
    return str(number) if minimum <= number <= maximum else None


def _attribute_text(value: object) -> str:
    """把 BeautifulSoup 属性值稳定转换为普通字符串。"""
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _is_supported_markup(content: str) -> bool:
    """仅把白名单或需整段删除的活动标签识别为富 HTML。"""
    if "<" not in content or ">" not in content:
        return False
    tokens = list(_MARKUP_TOKEN_RE.finditer(content))
    closing_names = {
        match.group("name").lower()
        for match in tokens
        if match.group("closing") and match.group("name").lower() in _SOURCE_MARKUP_TAGS
    }
    for match in tokens:
        if match.group("closing"):
            continue
        name = match.group("name").lower()
        if name not in _SOURCE_MARKUP_TAGS:
            continue
        if name in _VOID_MARKUP_TAGS or name in closing_names:
            return True
        if name in {"img", "embed"} and re.search(r"\bsrc\s*=", match.group("attrs"), re.IGNORECASE):
            return True
    return False


def _resolve_document_title(middle_json: MiddleJson, explicit_title: str | None) -> str:
    """按显式值、首个文档标题和固定回退值解析书名。"""
    if explicit_title:
        return _normalize_xml_text(explicit_title).strip()
    for page in middle_json.pages:
        for block in page.blocks:
            if isinstance(block, DocTitleBlock):
                title = _normalize_xml_text(inline_plain_text(block.content)).strip()
                if title:
                    return title
    return "MinerU Document"


def _stable_identifier(middle_json: MiddleJson, *, title: str, authors: tuple[str, ...], language: str) -> str:
    """由规范化 MiddleJson 和不随渲染时间变化的元数据生成稳定 UUID URN。"""
    seed = json.dumps(
        {
            "middle_json": middle_json.model_dump(mode="json"),
            "title": title,
            "authors": authors,
            "language": language,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(seed).hexdigest()
    return f"urn:uuid:{uuid5(NAMESPACE_URL, digest)}"


def _allocate_target_id(anchor: str | None, *, fallback: str, used_ids: set[str]) -> str:
    """从 producer anchor 或固定回退值分配无空白且不碰撞的 id。"""
    base = _safe_id_base(_anchor_key(anchor)) or fallback
    candidate = base
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def _safe_id_base(value: str) -> str:
    """把 anchor 归一化为适合 XHTML fragment 的稳定 id 基值。"""
    normalized = _normalize_xml_text(value).strip()
    normalized = re.sub(r"\s+", "-", normalized)
    normalized = re.sub(r"[^\w.:-]+", "-", normalized, flags=re.UNICODE).strip("-")
    return normalized


def _anchor_key(anchor: str | None) -> str:
    """保留 producer anchor 身份，仅去除首尾空白。"""
    return (anchor or "").strip()


def _plain_content_text(content: str) -> str:
    """从 body 内容提取图片 alt 所需的可见纯文本。"""
    if not content:
        return ""
    if _is_supported_markup(content):
        return BeautifulSoup(content, "html.parser").get_text(" ", strip=True)
    return _normalize_xml_text(content).strip()


def _normalize_code_language(language: str | None) -> str | None:
    """把代码语言限制为不会构造危险 class token 的短名称。"""
    normalized = (language or "").strip().lower().replace("_", "-")
    return normalized if _SAFE_LANGUAGE_RE.fullmatch(normalized) else None


def _normalize_xml_text(content: str) -> str:
    """替换 XML 1.0 禁止的控制字符和孤立 surrogate。"""
    return _INVALID_XML_TEXT_RE.sub("\ufffd", content)


def _needs_whitespace_preservation(content: str) -> bool:
    """判断文本是否含有 XHTML 默认会折叠的有效空白。"""
    return bool(content and (content != content.strip(" \t\n") or "  " in content or "\t" in content or "\n" in content))


def _has_visible_content(element: etree._Element) -> bool:
    """判断元素是否包含可见文本或媒体、结构子节点。"""
    if element.text and element.text.strip():
        return True
    if len(element):
        return True
    return False


def _xhtml(tag: str) -> str:
    """返回 XHTML namespace 下的 Clark notation 标签名。"""
    return f"{{{_XHTML_NS}}}{tag}"


@lru_cache(maxsize=1)
def _load_epub_stylesheet() -> bytes:
    """读取随包分发的静态 EPUB 样式表并缓存字节。"""
    root = resources.files("mineru").joinpath("resources", "epub")
    return root.joinpath(_STYLE_RESOURCE_NAME).read_bytes()


__all__ = ["render_epub"]
