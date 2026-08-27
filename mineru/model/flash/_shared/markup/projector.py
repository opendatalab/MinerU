# Copyright (c) Opendatalab. All rights reserved.
"""把静态 XHTML/HTML DOM 投影为 MinerU raw blocks。"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from lxml import etree  # type: ignore[reportMissingImports]

from .....types import RAW_ALGORITHM, BlockType, VISUAL_TYPE_MAPPING
from ..hyperlink import render_inline_hyperlink
from ..names import local_name
from .formula import FormulaExtraction, extract_formula
from .styles import MarkupStylesheet, TextStyle


BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "main",
        "math",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "summary",
        "svg",
        "table",
        "ul",
    }
)
SKIPPED_TAGS = frozenset(
    {
        "audio",
        "button",
        "canvas",
        "embed",
        "form",
        "head",
        "iframe",
        "input",
        "noscript",
        "object",
        "script",
        "select",
        "style",
        "template",
        "textarea",
        "video",
    }
)
_WHITESPACE_RE = re.compile(r"[\t\r\n\f ]+")
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
_MAX_TABLE_SPAN = 1_000
_CAPTION_TOKENS = frozenset(
    {
        "caption",
        "figure-caption",
        "image-caption",
        "table-caption",
        "chart-caption",
        "code-caption",
        "mineru-caption",
    }
)
_FOOTNOTE_TOKENS = frozenset(
    {
        "footnote",
        "figure-footnote",
        "image-footnote",
        "table-footnote",
        "chart-footnote",
        "code-footnote",
        "mineru-footnote",
    }
)
_VISUAL_ELEMENT_TAGS = frozenset({"img", "image", "pre", "svg", "table"})
_LIST_PAGE_BLOCK_TAGS = frozenset({"figure", "image", "img", "math", "pre", "svg", "table"})
_InlineProjectionSegment: TypeAlias = str | dict[str, object]


def clean_text_node(value: str | None) -> str:
    """折叠普通标记文档文本节点中的排版空白。"""
    return _WHITESPACE_RE.sub(" ", value) if value else ""


def visible_text(element: etree._Element) -> str:
    """提取元素折叠空白后的可见纯文本。"""
    return _WHITESPACE_RE.sub(" ", html.unescape("".join(element.itertext()))).strip()


def _semantic_tokens(element: etree._Element) -> frozenset[str]:
    """按 class/id 的完整空白 token 返回小写集合，不执行任意 substring 匹配。"""
    value = f"{element.get('class') or ''} {element.get('id') or ''}".casefold()
    return frozenset(value.split())


def _raw_visual_type(value: object) -> BlockType | None:
    """把 raw visual 主体或 algorithm 规范为统一父块类型。"""
    if value in {BlockType.IMAGE, BlockType.TABLE, BlockType.CHART}:
        return BlockType(value)
    if value in {BlockType.CODE, RAW_ALGORITHM}:
        return BlockType.CODE
    return None


def _append_inline_segment(
    segments: list[_InlineProjectionSegment],
    segment: _InlineProjectionSegment,
) -> None:
    """追加行内投影片段，并合并相邻文本以保持稳定 block 粒度。"""
    if isinstance(segment, str) and segments and isinstance(segments[-1], str):
        segments[-1] += segment
    elif not isinstance(segment, str) or segment:
        segments.append(segment)


def entity_text(element: etree._Element) -> str:
    """把 lxml 保留的安全命名实体恢复为可见文本。"""
    name = getattr(element, "name", "")
    return html.unescape(f"&{name};") if name else ""


def bounded_table_span(value: str) -> str | None:
    """规范化有界表格跨度，避免异常整数放大渲染网格。"""
    if not value.isdigit():
        return None
    normalized = value.lstrip("0")
    if not normalized or len(normalized) > len(str(_MAX_TABLE_SPAN)):
        return None
    span = int(normalized)
    return str(span) if span <= _MAX_TABLE_SPAN else None


def visible_raw_text_with_style(
    element: etree._Element,
    stylesheet: MarkupStylesheet,
    style: TextStyle,
    visibility_hidden: bool,
) -> str:
    """递归提取遵守整树隐藏和继承 visibility 的原始文本。"""
    parts: list[str] = [] if visibility_hidden else [element.text or ""]
    for child in element:
        if isinstance(child.tag, str):
            resolved = stylesheet.resolve(child, style, visibility_hidden)
            if not resolved.subtree_hidden:
                parts.append(
                    visible_raw_text_with_style(
                        child,
                        stylesheet,
                        resolved.text,
                        resolved.visibility_hidden,
                    )
                )
        elif not visibility_hidden:
            parts.append(entity_text(child))
        if not visibility_hidden:
            parts.append(child.tail or "")
    return "".join(parts)


@dataclass(frozen=True, slots=True)
class ResolvedMarkupImage:
    """保存标记文档图片解析后的互斥载荷和说明文本。"""

    image_base64: str | None = None
    image_url: str | None = None
    alt: str = ""


class MarkupContext(Protocol):
    """定义 projector 向具体容器请求链接、图片和 anchor 的边界。"""

    def resolve_link(self, href: str) -> str | None:
        """解析一个安全链接目标。"""

    def resolve_image(self, source: str, *, alt: str = "") -> ResolvedMarkupImage | None:
        """解析图片为 data URI、远程 URL 或可见降级文本。"""

    def heading_anchor(self, heading: etree._Element) -> str | None:
        """返回标题对应的规范 anchor。"""

    def heading_label(self, anchor: str) -> str | None:
        """返回规范 anchor 对应的标题标签。"""

    def note_anchor(self, note: etree._Element) -> str | None:
        """返回脚注节点对应的规范 anchor。"""


class MarkupProjector:
    """按 DOM 顺序把一个静态内容根节点投影为统一 raw blocks。"""

    def __init__(
        self,
        root: etree._Element,
        context: MarkupContext,
        stylesheet: MarkupStylesheet,
        *,
        single_document_title: bool = False,
        document_title_emitted: bool = False,
    ) -> None:
        """绑定 DOM、格式适配器、有限 CSS 和标题策略。"""
        self.root = root
        self.context = context
        self.stylesheet = stylesheet
        self.single_document_title = single_document_title
        self.document_title_emitted = document_title_emitted

    def convert(self) -> list[dict[str, object]]:
        """转换内容根节点的子树并返回按 DOM 顺序排列的 raw blocks。"""
        resolved = self.stylesheet.resolve(self.root, TextStyle())
        if resolved.subtree_hidden:
            return []
        name = local_name(self.root)
        if name == "figure" or (name in {"aside", "div", "section"} and self._has_contextual_visual_annotation(self.root)):
            return self._parse_figure(self.root, resolved.text, resolved.visibility_hidden)
        return self._parse_container_contents(self.root, resolved.text, resolved.visibility_hidden)

    def convert_svg(self) -> list[dict[str, object]]:
        """把 standalone SVG 根节点尽力转换为文本和静态图片。"""
        resolved = self.stylesheet.resolve(self.root, TextStyle())
        return [] if resolved.subtree_hidden else self._parse_svg(self.root, resolved.text, resolved.visibility_hidden)

    def project_block(self, element: etree._Element) -> list[dict[str, object]]:
        """把一个已知块元素按默认继承样式投影，供版本化 HTML 解码复用。"""
        return self._parse_block(element, TextStyle())

    def project_inline_content(self, element: etree._Element) -> str:
        """把一个已知行内容器恢复为统一富文本字符串。"""
        resolved = self.stylesheet.resolve(element, TextStyle())
        if resolved.subtree_hidden:
            return ""
        content, extras = self._render_inline_children(element, resolved.text, resolved.visibility_hidden)
        if extras:
            raise ValueError("inline projection produced unexpected block content")
        return content.strip()

    def _parse_container_contents(
        self,
        element: etree._Element,
        style: TextStyle,
        visibility_hidden: bool = False,
    ) -> list[dict[str, object]]:
        """把连续行内内容和块级子元素按源顺序拆成 raw blocks。"""
        blocks: list[dict[str, object]] = []
        inline_parts: list[str] = [] if visibility_hidden else [self._render_text(element.text, style)]

        def flush_inline() -> None:
            """把当前连续行内片段写为普通正文 block。"""
            content = "".join(inline_parts).strip()
            inline_parts.clear()
            if content:
                blocks.append({"type": BlockType.TEXT, "content": content})

        for child in element:
            if not isinstance(child.tag, str):
                if not visibility_hidden:
                    inline_parts.append(self._render_text(entity_text(child), style))
                    inline_parts.append(self._render_text(child.tail, style))
                continue
            name = local_name(child)
            if name in BLOCK_TAGS:
                flush_inline()
                blocks.extend(self._parse_block(child, style, visibility_hidden))
            else:
                for segment in self._render_inline_element_ordered(child, style, visibility_hidden):
                    if isinstance(segment, str):
                        inline_parts.append(segment)
                    else:
                        flush_inline()
                        blocks.append(segment)
            if not visibility_hidden:
                inline_parts.append(self._render_text(child.tail, style))
        flush_inline()
        return blocks

    def _parse_block(
        self,
        element: etree._Element,
        inherited: TextStyle,
        inherited_visibility_hidden: bool = False,
    ) -> list[dict[str, object]]:
        """把一个块级元素分派到对应 raw block 转换逻辑。"""
        resolved = self.stylesheet.resolve(element, inherited, inherited_visibility_hidden)
        if resolved.subtree_hidden:
            return []
        name = local_name(element)
        if name in SKIPPED_TAGS or name == "hr":
            return []
        if self.context.note_anchor(element) is not None:
            return self._parse_note_element(element, resolved.text, resolved.visibility_hidden)
        if name in {"h1", "h2", "h3", "h4", "h5", "h6", "p"}:
            return self._parse_textual_block(element, name, resolved.text, resolved.visibility_hidden)
        if name in {"ul", "ol"}:
            list_block, extras = self._parse_list(element, resolved.text, resolved.visibility_hidden)
            return ([list_block] if list_block is not None else []) + extras
        if name == "table":
            return self._parse_table(element, resolved.text, resolved.visibility_hidden)
        if name == "pre":
            content = self._visible_raw_text(element, resolved.text, resolved.visibility_hidden)
            language = self._code_language_hint(element)
            block: dict[str, object] = {"type": BlockType.CODE, "content": content}
            if language:
                block["guess_lang"] = language
            return [block] if content.strip() else []
        if name == "math":
            if resolved.visibility_hidden:
                return []
            formula = self._formula_extraction(element)
            if formula is not None:
                return [{"type": BlockType.EQUATION, "content": formula.latex}]
            fallback = self._visible_plain_text(element, resolved.text, resolved.visibility_hidden)
            return [{"type": BlockType.TEXT, "content": html.escape(fallback, quote=False)}] if fallback else []
        if name == "figure":
            return self._parse_figure(element, resolved.text, resolved.visibility_hidden)
        if name in {"aside", "div", "section"} and self._has_contextual_visual_annotation(element):
            return self._parse_figure(element, resolved.text, resolved.visibility_hidden)
        if name == "svg":
            return self._parse_svg(element, resolved.text, resolved.visibility_hidden)
        return self._parse_container_contents(element, resolved.text, resolved.visibility_hidden)

    def _parse_textual_block(
        self,
        element: etree._Element,
        name: str,
        style: TextStyle,
        visibility_hidden: bool,
    ) -> list[dict[str, object]]:
        """转换标题或段落，并旁路其中的视觉 blocks。"""
        blocks: list[dict[str, object]] = []
        text_emitted = False
        for segment in self._render_inline_children_ordered(element, style, visibility_hidden):
            if not isinstance(segment, str):
                blocks.append(segment)
                continue
            content = segment.strip()
            if not content:
                continue
            if text_emitted:
                blocks.append({"type": BlockType.TEXT, "content": content})
                continue
            if name == "h1" and (not self.single_document_title or not self.document_title_emitted):
                block: dict[str, object] = {"type": BlockType.DOC_TITLE, "level": 1, "content": content.strip()}
                self.document_title_emitted = True
            elif name.startswith("h"):
                level = min(max(int(name[1:]), 2), 6)
                block = {
                    "type": BlockType.PARAGRAPH_TITLE,
                    "level": level,
                    "is_numbered_style": False,
                    "content": content.strip(),
                }
            else:
                block = {"type": BlockType.TEXT, "content": content.strip()}
            if name.startswith("h") and (anchor := self.context.heading_anchor(element)):
                block["anchor"] = anchor
            blocks.append(block)
            text_emitted = True
        return blocks

    def _parse_note_element(
        self,
        element: etree._Element,
        style: TextStyle,
        visibility_hidden: bool = False,
    ) -> list[dict[str, object]]:
        """逐块转换单条脚注，并只给首个文本脚注挂载 anchor。"""
        blocks = self._parse_container_contents(element, style, visibility_hidden)
        anchor = self.context.note_anchor(element)
        anchor_attached = False
        for block in blocks:
            if block.get("type") != BlockType.TEXT or not str(block.get("content") or "").strip():
                continue
            block["type"] = BlockType.PAGE_FOOTNOTE
            if anchor is not None and not anchor_attached:
                block["anchor"] = anchor
                anchor_attached = True
        return blocks

    def _render_inline_children(
        self,
        element: etree._Element,
        style: TextStyle,
        visibility_hidden: bool = False,
    ) -> tuple[str, list[dict[str, object]]]:
        """渲染元素的连续行内内容，并旁路其中的视觉 blocks。"""
        segments = self._render_inline_children_ordered(element, style, visibility_hidden)
        return (
            "".join(segment for segment in segments if isinstance(segment, str)),
            [segment for segment in segments if not isinstance(segment, str)],
        )

    def _render_inline_children_ordered(
        self,
        element: etree._Element,
        style: TextStyle,
        visibility_hidden: bool = False,
    ) -> list[_InlineProjectionSegment]:
        """按 DOM 顺序返回连续文本与旁路 block，保留 inline visual 前后边界。"""
        segments: list[_InlineProjectionSegment] = []
        if not visibility_hidden:
            _append_inline_segment(segments, self._render_text(element.text, style))
        for child in element:
            if not isinstance(child.tag, str):
                if not visibility_hidden:
                    _append_inline_segment(segments, self._render_text(entity_text(child), style))
                    _append_inline_segment(segments, self._render_text(child.tail, style))
                continue
            for segment in self._render_inline_element_ordered(child, style, visibility_hidden):
                _append_inline_segment(segments, segment)
            if not visibility_hidden:
                _append_inline_segment(segments, self._render_text(child.tail, style))
        return segments

    def _render_inline_element(
        self,
        element: etree._Element,
        inherited: TextStyle,
        inherited_visibility_hidden: bool = False,
    ) -> tuple[str, list[dict[str, object]]]:
        """把一个行内元素转换为内部富文本协议和可选视觉块。"""
        segments = self._render_inline_element_ordered(element, inherited, inherited_visibility_hidden)
        return (
            "".join(segment for segment in segments if isinstance(segment, str)),
            [segment for segment in segments if not isinstance(segment, str)],
        )

    def _render_inline_element_ordered(
        self,
        element: etree._Element,
        inherited: TextStyle,
        inherited_visibility_hidden: bool = False,
    ) -> list[_InlineProjectionSegment]:
        """递归投影单个行内元素，并在嵌套 visual 位置保留顺序分段。"""
        resolved = self.stylesheet.resolve(element, inherited, inherited_visibility_hidden)
        if resolved.subtree_hidden:
            return []
        name = local_name(element)
        if name in SKIPPED_TAGS:
            return []
        if name == "br":
            return [] if resolved.visibility_hidden else ["\n"]
        if name in {"img", "image"}:
            return [] if resolved.visibility_hidden else self._image_blocks(element)
        if name == "math":
            if resolved.visibility_hidden:
                return []
            formula = self._formula_extraction(element)
            if formula is not None:
                if formula.display == "block":
                    return [{"type": BlockType.EQUATION, "content": formula.latex}]
                return [f"<eq>{html.escape(formula.latex, quote=False)}</eq>"]
            fallback = self._visible_plain_text(element, resolved.text, resolved.visibility_hidden)
            return [html.escape(fallback, quote=False)] if fallback else []
        if name == "code":
            if resolved.visibility_hidden:
                return []
            code = self._visible_raw_text(element, resolved.text, resolved.visibility_hidden)
            return [f"<code>{html.escape(code, quote=False)}</code>"] if code else []
        if name in BLOCK_TAGS:
            return self._parse_block(element, inherited, inherited_visibility_hidden)
        segments = self._render_inline_children_ordered(element, resolved.text, resolved.visibility_hidden)
        if name == "a":
            href = element.get("href") or element.get(_XLINK_HREF) or ""
            target = self.context.resolve_link(href)
            if target:
                return [
                    render_inline_hyperlink(segment, target) if isinstance(segment, str) and segment.strip() else segment
                    for segment in segments
                ]
        return segments

    @staticmethod
    def _render_text(value: str | None, style: TextStyle) -> str:
        """折叠并转义文本节点，再按现有行内协议应用文字样式。"""
        text = clean_text_node(value)
        if not text:
            return ""
        escaped = html.escape(text, quote=False)
        names = style.names()
        return f'<text style="{",".join(names)}">{escaped}</text>' if names else escaped

    def _visible_raw_text(
        self,
        element: etree._Element,
        style: TextStyle,
        visibility_hidden: bool = False,
    ) -> str:
        """递归提取可见原始文本，并允许后代显式恢复 visibility。"""
        return visible_raw_text_with_style(element, self.stylesheet, style, visibility_hidden)

    def _visible_plain_text(
        self,
        element: etree._Element,
        style: TextStyle,
        visibility_hidden: bool = False,
    ) -> str:
        """返回折叠空白并还原实体后的可见纯文本。"""
        value = self._visible_raw_text(element, style, visibility_hidden)
        return _WHITESPACE_RE.sub(" ", html.unescape(value)).strip()

    def _image_blocks(
        self,
        element: etree._Element,
        *,
        caption: str | None = None,
        emit_alt_caption: bool = True,
    ) -> list[dict[str, object]]:
        """把可解析图片转换为 image block，并用 caption/alt 补说明。"""
        source = element.get("src") or element.get("href") or element.get(_XLINK_HREF) or ""
        requested_alt = (caption or element.get("alt") or element.get("title") or "").strip()
        resolved = self.context.resolve_image(source, alt=requested_alt)
        alt = (resolved.alt if resolved is not None else requested_alt).strip()
        if resolved is None or not (resolved.image_base64 or resolved.image_url):
            return [{"type": BlockType.TEXT, "content": html.escape(alt, quote=False)}] if alt else []
        block: dict[str, object] = {"type": BlockType.IMAGE, "content": ""}
        if resolved.image_base64:
            block["image_base64"] = resolved.image_base64
        if resolved.image_url:
            block["image_url"] = resolved.image_url
        blocks: list[dict[str, object]] = [block]
        annotation = (caption or (alt if emit_alt_caption else "")).strip()
        if annotation:
            blocks.append({"type": BlockType.IMAGE_CAPTION, "content": html.escape(annotation, quote=False)})
        return blocks

    def _parse_figure(
        self,
        element: etree._Element,
        style: TextStyle,
        visibility_hidden: bool = False,
    ) -> list[dict[str, object]]:
        """按标准标签或完整 token 解析 visual 主体、caption 与 footnote。"""
        annotations = [
            (child, kind)
            for child in element
            if isinstance(child.tag, str) and (kind := self._visual_annotation_kind(child)) is not None
        ]
        annotation_elements = {child for child, _ in annotations}
        mineru_figure = "mineru-figure" in (element.get("class") or "").casefold().split()
        blocks, visual_blocks_by_child = self._parse_figure_contents(
            element,
            style,
            visibility_hidden,
            annotation_elements=annotation_elements,
            emit_alt_caption=not mineru_figure and not annotations,
        )

        annotations_by_visual: dict[int, list[dict[str, object]]] = {}
        unbound_annotations: list[dict[str, object]] = []
        for annotation, kind in annotations:
            resolved = self.stylesheet.resolve(annotation, style, visibility_hidden)
            if resolved.subtree_hidden:
                continue
            target = self._figure_annotation_target(annotation, element, visual_blocks_by_child)
            visual_type = _raw_visual_type(target.get("type")) if target is not None else None
            annotation_type = VISUAL_TYPE_MAPPING[visual_type][kind] if visual_type is not None else BlockType.TEXT
            annotation_blocks: list[dict[str, object]] = []
            for segment in self._render_inline_children_ordered(annotation, resolved.text, resolved.visibility_hidden):
                if isinstance(segment, str):
                    if content := segment.strip():
                        annotation_blocks.append({"type": annotation_type, "content": content})
                    continue
                if visual_type is not None and segment.get("type") == BlockType.TEXT:
                    segment = {**segment, "type": annotation_type}
                annotation_blocks.append(segment)
            if target is not None and visual_type is not None:
                annotations_by_visual.setdefault(id(target), []).extend(annotation_blocks)
            else:
                unbound_annotations.extend(annotation_blocks)

        output: list[dict[str, object]] = []
        for block in blocks:
            output.append(block)
            output.extend(annotations_by_visual.get(id(block), ()))
        output.extend(unbound_annotations)
        return output

    def _parse_figure_contents(
        self,
        element: etree._Element,
        style: TextStyle,
        visibility_hidden: bool,
        *,
        annotation_elements: set[etree._Element],
        emit_alt_caption: bool,
    ) -> tuple[list[dict[str, object]], dict[etree._Element, list[dict[str, object]]]]:
        """按 DOM 顺序缓冲 figure 文本，并在 visual extras 前后切分正文 block。"""
        blocks: list[dict[str, object]] = []
        visual_blocks_by_child: dict[etree._Element, list[dict[str, object]]] = {}
        inline_parts: list[str] = [] if visibility_hidden else [self._render_text(element.text, style)]

        def flush_inline() -> None:
            """把 figure 当前连续文本写为普通正文 block。"""
            content = "".join(inline_parts).strip()
            inline_parts.clear()
            if content:
                blocks.append({"type": BlockType.TEXT, "content": content})

        for child in element:
            if not isinstance(child.tag, str):
                if not visibility_hidden:
                    inline_parts.append(self._render_text(entity_text(child), style))
                    inline_parts.append(self._render_text(child.tail, style))
                continue
            if child in annotation_elements:
                if not visibility_hidden:
                    inline_parts.append(self._render_text(child.tail, style))
                continue

            first_child_block = len(blocks)
            name = local_name(child)
            if name in {"img", "image"}:
                flush_inline()
                child_style = self.stylesheet.resolve(child, style, visibility_hidden)
                if not child_style.subtree_hidden and not child_style.visibility_hidden:
                    blocks.extend(self._image_blocks(child, emit_alt_caption=emit_alt_caption))
            elif name in BLOCK_TAGS:
                flush_inline()
                blocks.extend(self._parse_block(child, style, visibility_hidden))
            else:
                for segment in self._render_inline_element_ordered(child, style, visibility_hidden):
                    if isinstance(segment, str):
                        inline_parts.append(segment)
                    else:
                        flush_inline()
                        blocks.append(segment)
            if not visibility_hidden:
                inline_parts.append(self._render_text(child.tail, style))
            child_visuals = [block for block in blocks[first_child_block:] if _raw_visual_type(block.get("type")) is not None]
            if child_visuals:
                visual_blocks_by_child[child] = child_visuals
        flush_inline()
        return blocks, visual_blocks_by_child

    @staticmethod
    def _figure_annotation_target(
        annotation: etree._Element,
        figure: etree._Element,
        visual_blocks_by_child: dict[etree._Element, list[dict[str, object]]],
    ) -> dict[str, object] | None:
        """把 annotation 绑定到最近前序 visual；没有前序时使用最近后序 visual。"""
        children = [child for child in figure if isinstance(child.tag, str)]
        position = children.index(annotation)
        for child in reversed(children[:position]):
            if visuals := visual_blocks_by_child.get(child):
                return visuals[-1]
        for child in children[position + 1 :]:
            if visuals := visual_blocks_by_child.get(child):
                return visuals[0]
        return None

    def _has_contextual_visual_annotation(self, element: etree._Element) -> bool:
        """仅在直属完整 token annotation 与 visual 后代并存时启用非标准容器解析。"""
        children = [child for child in element if isinstance(child.tag, str)]
        if not any(self._visual_annotation_kind(child) is not None for child in children):
            return False
        return any(
            local_name(candidate) in _VISUAL_ELEMENT_TAGS
            for child in children
            if self._visual_annotation_kind(child) is None
            for candidate in [child, *child.iterdescendants()]
            if isinstance(candidate.tag, str)
        )

    @staticmethod
    def _visual_annotation_kind(element: etree._Element) -> str | None:
        """用标准标签、role 或完整 class/id token 返回 caption/footnote 角色。"""
        if local_name(element) == "figcaption":
            return "caption"
        tokens = _semantic_tokens(element)
        roles = frozenset((element.get("role") or "").casefold().split())
        if tokens & _CAPTION_TOKENS or roles & {"caption", "doc-subtitle"}:
            return "caption"
        if tokens & _FOOTNOTE_TOKENS or roles & {"doc-footnote", "note"}:
            return "footnote"
        return None

    def _parse_svg(
        self,
        element: etree._Element,
        style: TextStyle,
        visibility_hidden: bool = False,
    ) -> list[dict[str, object]]:
        """从 SVG 尽力提取 title/desc/text 和静态 image。"""
        blocks: list[dict[str, object]] = []
        texts: list[str] = []

        def visit(parent: etree._Element, inherited: TextStyle, inherited_visibility_hidden: bool) -> None:
            """按 SVG 树顺序访问候选节点，并允许可见后代恢复输出。"""
            for child in parent:
                if not isinstance(child.tag, str):
                    continue
                resolved = self.stylesheet.resolve(child, inherited, inherited_visibility_hidden)
                if resolved.subtree_hidden:
                    continue
                name = local_name(child)
                if name in {"title", "desc", "text"}:
                    value = self._visible_plain_text(child, resolved.text, resolved.visibility_hidden)
                    if value and value not in texts:
                        texts.append(value)
                elif name == "image":
                    if not resolved.visibility_hidden:
                        blocks.extend(self._image_blocks(child))
                else:
                    visit(child, resolved.text, resolved.visibility_hidden)

        visit(element, style, visibility_hidden)
        if texts:
            blocks.insert(0, {"type": BlockType.TEXT, "content": html.escape("\n".join(texts), quote=False)})
        return blocks

    def _parse_table(
        self,
        table: etree._Element,
        style: TextStyle,
        visibility_hidden: bool = False,
    ) -> list[dict[str, object]]:
        """重建白名单化 HTML 表格，并把 caption 投影为表格说明。"""
        markup = self._serialize_table_node(table, style, visibility_hidden)
        if not markup:
            return []
        blocks: list[dict[str, object]] = [{"type": BlockType.TABLE, "content": markup}]
        caption_element = next(
            (child for child in table if isinstance(child.tag, str) and local_name(child) == "caption"),
            None,
        )
        if caption_element is not None:
            caption_style = self.stylesheet.resolve(caption_element, style, visibility_hidden)
            if not caption_style.subtree_hidden:
                caption = self._visible_plain_text(caption_element, caption_style.text, caption_style.visibility_hidden)
                if caption:
                    blocks.append({"type": BlockType.TABLE_CAPTION, "content": html.escape(caption, quote=False)})
        return blocks

    def _serialize_table_node(
        self,
        element: etree._Element,
        inherited: TextStyle,
        inherited_visibility_hidden: bool = False,
        *,
        row_link_target: str | None = None,
    ) -> str:
        """递归序列化安全表格结构、行内样式、链接、公式和图片。"""
        resolved = self.stylesheet.resolve(element, inherited, inherited_visibility_hidden)
        if resolved.subtree_hidden:
            return ""
        name = local_name(element)
        if name == "caption" or name in SKIPPED_TAGS:
            return ""
        allowed = {
            "a",
            "b",
            "br",
            "code",
            "col",
            "colgroup",
            "em",
            "i",
            "img",
            "math",
            "p",
            "s",
            "span",
            "strong",
            "sub",
            "sup",
            "table",
            "tbody",
            "td",
            "tfoot",
            "th",
            "thead",
            "tr",
            "u",
        }
        if name not in allowed:
            return self._serialize_table_children(
                element,
                resolved.text,
                resolved.visibility_hidden,
                row_link_target=row_link_target,
            )
        if name == "br":
            return "" if resolved.visibility_hidden else "<br>"
        if name == "math":
            if resolved.visibility_hidden:
                return ""
            formula = self._formula_extraction(element)
            if formula is not None:
                return f"<eq>{html.escape(formula.latex, quote=False)}</eq>"
            fallback = self._visible_plain_text(element, resolved.text, resolved.visibility_hidden)
            return html.escape(fallback, quote=False)
        if name == "img":
            if resolved.visibility_hidden:
                return ""
            source = element.get("src") or ""
            alt_text = (element.get("alt") or "").strip()
            image = self.context.resolve_image(source, alt=alt_text)
            if image is None:
                return html.escape(alt_text, quote=False)
            image_source = image.image_base64 or image.image_url
            alt = html.escape(image.alt or alt_text, quote=True)
            return f'<img src="{html.escape(image_source, quote=True)}" alt="{alt}">' if image_source else alt
        if name == "tr":
            row_link_target = self._toc_table_row_target(element)
        attributes: list[str] = []
        if name in {"td", "th"}:
            for attribute in ("colspan", "rowspan", "scope"):
                value = (element.get(attribute) or "").strip()
                if attribute == "scope" and value in {"col", "colgroup", "row", "rowgroup"}:
                    attributes.append(f'{attribute}="{value}"')
                elif span := bounded_table_span(value):
                    attributes.append(f'{attribute}="{span}"')
        elif name in {"col", "colgroup"}:
            if span := bounded_table_span((element.get("span") or "").strip()):
                attributes.append(f'span="{span}"')
        if name == "a":
            target = self.context.resolve_link(element.get("href") or "")
            if target:
                attributes.append(f'href="{html.escape(target, quote=True)}"')
        inner = self._serialize_table_children(
            element,
            resolved.text,
            resolved.visibility_hidden,
            row_link_target=row_link_target,
        )
        if resolved.visibility_hidden and not inner:
            return ""
        if name in {"td", "th"} and row_link_target and self._table_cell_can_inherit_toc_link(element):
            inner = f'<a href="{html.escape(row_link_target, quote=True)}">{inner}</a>'
        attrs = f" {' '.join(attributes)}" if attributes else ""
        return f"<{name}{attrs}>{inner}</{name}>"

    def _serialize_table_children(
        self,
        element: etree._Element,
        style: TextStyle,
        visibility_hidden: bool = False,
        *,
        row_link_target: str | None = None,
    ) -> str:
        """序列化表格节点的文本、子元素和 tail。"""
        parts = [] if visibility_hidden else [self._render_table_text(element.text, style)]
        for child in element:
            if isinstance(child.tag, str):
                parts.append(self._serialize_table_node(child, style, visibility_hidden, row_link_target=row_link_target))
            elif not visibility_hidden:
                parts.append(self._render_table_text(entity_text(child), style))
            if not visibility_hidden:
                parts.append(self._render_table_text(child.tail, style))
        return "".join(parts)

    def _toc_table_row_target(self, row: etree._Element) -> str | None:
        """为严格匹配单一目标标题的目录表格行返回内部链接。"""
        links = [
            element
            for element in row.iter()
            if isinstance(element.tag, str) and local_name(element) == "a" and (element.get("href") or "").strip()
        ]
        if not links:
            return None
        resolved_targets: list[str] = []
        for link in links:
            target = self.context.resolve_link(link.get("href") or "")
            if target is None or not target.startswith("#"):
                return None
            resolved_targets.append(target)
        if len(set(resolved_targets)) != 1:
            return None
        target = resolved_targets[0]
        title = self.context.heading_label(target[1:])
        if title is None:
            return None
        cells = [child for child in row if isinstance(child.tag, str) and local_name(child) in {"td", "th"}]
        row_label = " ".join(value for cell in cells if (value := visible_text(cell)))
        normalized_row = _WHITESPACE_RE.sub(" ", html.unescape(row_label)).strip().casefold()
        normalized_title = _WHITESPACE_RE.sub(" ", html.unescape(title)).strip().casefold()
        return target if normalized_row and normalized_row == normalized_title else None

    @staticmethod
    def _table_cell_can_inherit_toc_link(cell: etree._Element) -> bool:
        """只允许纯文本与行内样式单元格继承目录行的唯一内部链接。"""
        if not visible_text(cell):
            return False
        allowed_inline = {"b", "br", "code", "em", "i", "s", "span", "strong", "sub", "sup", "u"}
        return all(isinstance(child.tag, str) and local_name(child) in allowed_inline for child in cell.iterdescendants())

    @staticmethod
    def _render_table_text(value: str | None, style: TextStyle) -> str:
        """把表格文字转义后包装为 renderer 支持的安全 HTML 样式标签。"""
        rendered = html.escape(clean_text_node(value), quote=False)
        if not rendered:
            return ""
        for enabled, tag in (
            (style.bold, "strong"),
            (style.italic, "em"),
            (style.underline, "u"),
            (style.strikethrough, "s"),
            (style.superscript, "sup"),
            (style.subscript, "sub"),
        ):
            if enabled:
                rendered = f"<{tag}>{rendered}</{tag}>"
        return rendered

    def _parse_list(
        self,
        element: etree._Element,
        style: TextStyle,
        visibility_hidden: bool = False,
    ) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
        """解析有序/无序列表，并投影为连续阿拉伯编号结构。"""
        if self._list_contains_page_blocks(element):
            return self._parse_list_with_page_blocks(element, style, visibility_hidden)
        ordered = local_name(element) == "ol"
        items = [child for child in element if isinstance(child.tag, str) and local_name(child) == "li"]
        if not items:
            return None, []
        children: list[dict[str, object]] = []
        extras: list[dict[str, object]] = []
        for item in items:
            item_style = self.stylesheet.resolve(item, style, visibility_hidden)
            if item_style.subtree_hidden:
                continue
            if self.context.note_anchor(item) is not None:
                extras.extend(self._parse_note_element(item, item_style.text, item_style.visibility_hidden))
                continue
            content_parts: list[str] = [] if item_style.visibility_hidden else [self._render_text(item.text, item_style.text)]
            nested_lists: list[dict[str, object]] = []
            for child in item:
                if not isinstance(child.tag, str):
                    if not item_style.visibility_hidden:
                        content_parts.append(self._render_text(entity_text(child), item_style.text))
                        content_parts.append(self._render_text(child.tail, item_style.text))
                    continue
                name = local_name(child)
                if name in {"ul", "ol"}:
                    nested_style = self.stylesheet.resolve(child, item_style.text, item_style.visibility_hidden)
                    if not nested_style.subtree_hidden:
                        nested, nested_extras = self._parse_list(child, nested_style.text, nested_style.visibility_hidden)
                        if nested is not None:
                            nested_lists.append(nested)
                        extras.extend(nested_extras)
                elif name in {"table", "figure", "svg"}:
                    extras.extend(self._parse_block(child, item_style.text, item_style.visibility_hidden))
                elif name in BLOCK_TAGS:
                    child_style = self.stylesheet.resolve(child, item_style.text, item_style.visibility_hidden)
                    if not child_style.subtree_hidden:
                        if self.context.note_anchor(child) is not None:
                            extras.extend(self._parse_note_element(child, child_style.text, child_style.visibility_hidden))
                        else:
                            rendered, child_extras = self._render_inline_children(
                                child,
                                child_style.text,
                                child_style.visibility_hidden,
                            )
                            if rendered.strip():
                                content_parts.append(rendered)
                            extras.extend(child_extras)
                else:
                    rendered, child_extras = self._render_inline_element(child, item_style.text, item_style.visibility_hidden)
                    content_parts.append(rendered)
                    extras.extend(child_extras)
                if not item_style.visibility_hidden:
                    content_parts.append(self._render_text(child.tail, item_style.text))
            content = "".join(content_parts).strip()
            if content:
                children.append({"type": BlockType.TEXT, "content": content})
            children.extend(nested_lists)
        if not children:
            return None, extras
        block: dict[str, object] = {
            "type": BlockType.LIST,
            "attribute": "ordered" if ordered else "unordered",
            "content": children,
        }
        if ordered:
            block["start"] = self._ordered_list_start(element)
        return block, extras

    @staticmethod
    def _list_contains_page_blocks(element: etree._Element) -> bool:
        """判断列表是否含可能提升为页面兄弟的 visual、code 或公式子树。"""
        return any(
            isinstance(candidate.tag, str) and local_name(candidate) in _LIST_PAGE_BLOCK_TAGS
            for candidate in element.iterdescendants()
        )

    def _parse_list_with_page_blocks(
        self,
        element: etree._Element,
        style: TextStyle,
        visibility_hidden: bool,
    ) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
        """把含 visual 的列表切成有序 list/text/page block 片段，保持 DOM 阅读顺序。"""
        ordered = local_name(element) == "ol"
        list_start = self._ordered_list_start(element) if ordered else 1
        items = [child for child in element if isinstance(child.tag, str) and local_name(child) == "li"]
        pending_children: list[dict[str, object]] = []
        pending_start = list_start
        output: list[dict[str, object]] = []
        visible_item_ordinal = 0
        has_page_blocks = False

        def flush_pending() -> None:
            """把当前连续列表项写为一个顶层 list block。"""
            nonlocal pending_children
            if not pending_children:
                return
            output.append(self._build_raw_list_block(pending_children, ordered=ordered, start=pending_start))
            pending_children = []

        for item in items:
            item_style = self.stylesheet.resolve(item, style, visibility_hidden)
            if item_style.subtree_hidden:
                continue
            if self.context.note_anchor(item) is not None:
                flush_pending()
                output.extend(self._parse_note_element(item, item_style.text, item_style.visibility_hidden))
                has_page_blocks = True
                continue

            segments = self._normalize_list_item_segments(
                self._render_inline_children_ordered(item, item_style.text, item_style.visibility_hidden)
            )
            page_positions = [
                index
                for index, segment in enumerate(segments)
                if not isinstance(segment, str) and segment.get("type") != BlockType.LIST
            ]
            if not page_positions:
                item_children = self._list_item_children(segments)
                if item_children:
                    if not pending_children:
                        pending_start = list_start + visible_item_ordinal
                    pending_children.extend(item_children)
                    visible_item_ordinal += 1
                continue

            first_page_position = page_positions[0]
            prefix_children = self._list_item_children(segments[:first_page_position])
            if not pending_children:
                pending_start = list_start + visible_item_ordinal
            pending_children.extend(prefix_children or [{"type": BlockType.TEXT, "content": ""}])
            flush_pending()

            for segment in segments[first_page_position:]:
                if isinstance(segment, str):
                    content = segment.strip()
                    if content:
                        output.append({"type": BlockType.TEXT, "content": content})
                else:
                    output.append(segment)
            visible_item_ordinal += 1
            has_page_blocks = True

        if not has_page_blocks:
            return (
                self._build_raw_list_block(pending_children, ordered=ordered, start=list_start) if pending_children else None,
                [],
            )
        flush_pending()
        return None, output

    @staticmethod
    def _normalize_list_item_segments(
        segments: list[_InlineProjectionSegment],
    ) -> list[_InlineProjectionSegment]:
        """把列表内部普通 text block 还原为文本片段，保留 visual/list 页面边界。"""
        normalized: list[_InlineProjectionSegment] = []
        for segment in segments:
            if isinstance(segment, dict) and segment.get("type") == BlockType.TEXT:
                _append_inline_segment(normalized, str(segment.get("content") or ""))
            else:
                _append_inline_segment(normalized, segment)
        return normalized

    @staticmethod
    def _list_item_children(segments: list[_InlineProjectionSegment]) -> list[dict[str, object]]:
        """把无页面 visual 的列表片段收敛为一个文本叶子及其嵌套列表。"""
        content = "".join(segment for segment in segments if isinstance(segment, str)).strip()
        children = [{"type": BlockType.TEXT, "content": content}] if content else []
        children.extend(segment for segment in segments if isinstance(segment, dict) and segment.get("type") == BlockType.LIST)
        return children

    @staticmethod
    def _build_raw_list_block(
        children: list[dict[str, object]],
        *,
        ordered: bool,
        start: int,
    ) -> dict[str, object]:
        """构造一段可由既有无坐标后处理编号的 raw list block。"""
        block: dict[str, object] = {
            "type": BlockType.LIST,
            "attribute": "ordered" if ordered else "unordered",
            "content": children,
        }
        if ordered:
            block["start"] = start
        return block

    @staticmethod
    def _ordered_list_start(element: etree._Element) -> int:
        """读取有序列表唯一通用起始值，非法或负值统一回退为一。"""
        try:
            start = int(element.get("start") or 1)
        except ValueError:
            return 1
        return start if start >= 0 else 1

    @staticmethod
    def _formula_extraction(element: etree._Element) -> FormulaExtraction | None:
        """调用共享公式优先级，返回裸 LaTeX 及来源信息。"""
        return extract_formula(element)

    @staticmethod
    def _code_language_hint(element: etree._Element) -> str | None:
        """从 pre/code 的标准 class 或 data 属性提取安全语言提示。"""
        candidates = [element, *[child for child in element if isinstance(child.tag, str) and local_name(child) == "code"]]
        for candidate in candidates:
            for attribute in ("data-language", "data-lang"):
                value = (candidate.get(attribute) or "").strip()
                if re.fullmatch(r"[A-Za-z0-9_.+#-]+", value):
                    return value
            for token in (candidate.get("class") or "").split():
                normalized = token.casefold()
                for prefix in ("language-", "lang-"):
                    if normalized.startswith(prefix):
                        value = token[len(prefix) :]
                        if re.fullmatch(r"[A-Za-z0-9_.+#-]+", value):
                            return value
        return None


__all__ = [
    "BLOCK_TAGS",
    "MarkupContext",
    "MarkupProjector",
    "ResolvedMarkupImage",
    "SKIPPED_TAGS",
    "bounded_table_span",
    "clean_text_node",
    "entity_text",
    "local_name",
    "visible_raw_text_with_style",
    "visible_text",
]
