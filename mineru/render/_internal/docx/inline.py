# Copyright (c) Opendatalab. All rights reserved.
"""MiddleJson 行内节点到 Word run、超链接、书签和 OMML 的 DOCX 写入。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha1
import re

from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.text.run import Run
from docx.shared import Pt, RGBColor
from lxml import etree
from loguru import logger

from ....backend.postprocess.inline import (
    InlineCode,
    InlineEquation,
    InlineLink,
    InlineNode,
    InlineStyled,
    InlineText,
    inline_plain_text,
    join_inline_contents,
    parse_inline_content,
)
from .math import DocxFormulaError, latex_to_omml

_BOOKMARK_SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")
_BOOKMARK_MAX_LENGTH = 40
_INVALID_XML_TEXT_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
_BARE_SCRIPT_RE = re.compile(r"^\s*(?P<marker>[\^_])\s*\{(?P<content>[^{}]+)\}\s*$")
_VISIBLE_SPACE_STYLES = frozenset({"underline", "strikethrough", "emphasis"})
_NONBREAKING_SPACE = "\u00a0"


@dataclass(frozen=True, slots=True)
class InlineRenderContext:
    """保存行内渲染告警所需的 block 定位与书签表。"""

    bookmarks: BookmarkRegistry
    page_idx: int
    block_index: int | None
    block_type: str

    def location(self) -> str:
        """返回稳定、可读的 page/block 定位文本。"""
        return f"page_idx={self.page_idx}, block_index={self.block_index}, block_type={self.block_type}"


class BookmarkRegistry:
    """把 MiddleJson anchor 映射为合法且唯一的 Word bookmark 名称。"""

    def __init__(self, anchors: Iterable[str]) -> None:
        """预注册全部 anchor，保证标题、目录和脚注前向引用可提前解析。"""
        self._names: dict[str, str] = {}
        self._attached: set[str] = set()
        self._used_names: set[str] = set()
        self._next_id = 0
        for anchor in anchors:
            normalized = anchor.strip()
            if normalized and normalized not in self._names:
                self._names[normalized] = self._allocate_name(normalized)

    def _allocate_name(self, anchor: str) -> str:
        """为一个原始 anchor 分配满足 Word 限制的确定性名称。"""
        base = _BOOKMARK_SAFE_RE.sub("_", anchor).strip("_")
        if not base or not base[0].isalpha():
            base = f"b_{base}"
        candidate = base[:_BOOKMARK_MAX_LENGTH]
        if candidate not in self._used_names:
            self._used_names.add(candidate)
            return candidate

        digest = sha1(anchor.encode("utf-8")).hexdigest()[:8]
        prefix_length = _BOOKMARK_MAX_LENGTH - len(digest) - 1
        candidate = f"{base[:prefix_length]}_{digest}"
        ordinal = 1
        while candidate in self._used_names:
            suffix = f"_{ordinal}"
            candidate = f"{base[: _BOOKMARK_MAX_LENGTH - len(suffix)]}{suffix}"
            ordinal += 1
        self._used_names.add(candidate)
        return candidate

    def resolve(self, anchor: str | None) -> str | None:
        """解析已注册 anchor；空值或未知 anchor 返回 None。"""
        if not anchor:
            return None
        return self._names.get(anchor.strip())

    def attach(self, paragraph: Paragraph, anchor: str | None) -> bool:
        """把 anchor bookmark 包围当前段落内容；重复正文 anchor 只保留首个。"""
        normalized = (anchor or "").strip()
        name = self.resolve(normalized)
        if name is None:
            return False
        if normalized in self._attached:
            logger.warning("Duplicate DOCX bookmark anchor ignored: {}", normalized)
            return False

        bookmark_id = str(self._next_id)
        self._next_id += 1
        start = OxmlElement("w:bookmarkStart")
        start.set(qn("w:id"), bookmark_id)
        start.set(qn("w:name"), name)
        end = OxmlElement("w:bookmarkEnd")
        end.set(qn("w:id"), bookmark_id)

        paragraph_xml = paragraph._p
        insert_at = 1 if paragraph_xml.pPr is not None else 0
        paragraph_xml.insert(insert_at, start)
        paragraph_xml.append(end)
        self._attached.add(normalized)
        return True


def append_inline_content(
    paragraph: Paragraph,
    content: str,
    *,
    context: InlineRenderContext,
) -> None:
    """解析并把一段 MiddleJson 行内内容追加到 Word 段落。"""
    append_inline_nodes(paragraph, parse_inline_content(content), context=context)


def append_joined_inline_contents(
    paragraph: Paragraph,
    contents: list[str],
    *,
    context: InlineRenderContext,
) -> None:
    """按共享边界规则合并续段，并把中性行内节点写入 Word。"""
    append_inline_nodes(paragraph, join_inline_contents(contents), context=context)


def append_inline_nodes(
    paragraph: Paragraph,
    nodes: list[InlineNode],
    *,
    context: InlineRenderContext,
    inherited_styles: tuple[str, ...] = (),
) -> None:
    """把行内节点写入段落，同时保留嵌套样式、链接与公式语义。"""
    _append_nodes_to_container(
        paragraph._p,
        paragraph,
        nodes,
        context=context,
        inherited_styles=inherited_styles,
        hyperlink=False,
    )


def append_internal_link(
    paragraph: Paragraph,
    label_nodes: list[InlineNode],
    *,
    anchor: str | None,
    context: InlineRenderContext,
) -> bool:
    """把目录标签写为内部链接；目标缺失时退回普通行内内容。"""
    bookmark_name = context.bookmarks.resolve(anchor)
    if bookmark_name is None:
        append_inline_nodes(paragraph, label_nodes, context=context)
        if anchor:
            logger.warning("Unmatched DOCX index anchor: {} ({})", anchor, context.location())
        return False

    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("w:anchor"), bookmark_name)
    hyperlink.set(qn("w:history"), "1")
    paragraph._p.append(hyperlink)
    _append_nodes_to_container(
        hyperlink,
        paragraph,
        label_nodes,
        context=context,
        inherited_styles=(),
        hyperlink=True,
    )
    return True


def _append_nodes_to_container(
    container: etree._Element,
    paragraph: Paragraph,
    nodes: list[InlineNode],
    *,
    context: InlineRenderContext,
    inherited_styles: tuple[str, ...],
    hyperlink: bool,
) -> None:
    """递归写入一个段落或 hyperlink XML 容器。"""
    for node in nodes:
        if isinstance(node, InlineText):
            _append_text_run(
                container,
                paragraph,
                node.content,
                styles=inherited_styles,
                hyperlink=hyperlink,
                context=context,
            )
            continue
        if isinstance(node, InlineCode):
            run = _append_text_run(
                container,
                paragraph,
                node.content,
                styles=inherited_styles,
                hyperlink=hyperlink,
                context=context,
            )
            run.font.name = "Courier New"
            run.font.size = Pt(9)
            continue
        if isinstance(node, InlineStyled):
            styles = tuple(dict.fromkeys((*inherited_styles, *node.styles)))
            _append_nodes_to_container(
                container,
                paragraph,
                node.children,
                context=context,
                inherited_styles=styles,
                hyperlink=hyperlink,
            )
            continue
        if isinstance(node, InlineEquation):
            bare_script = _BARE_SCRIPT_RE.fullmatch(node.latex)
            if bare_script is not None:
                script_style = "superscript" if bare_script.group("marker") == "^" else "subscript"
                styles = tuple(dict.fromkeys((*inherited_styles, script_style)))
                _append_text_run(
                    container,
                    paragraph,
                    bare_script.group("content"),
                    styles=styles,
                    hyperlink=hyperlink,
                    context=context,
                )
                continue
            if hyperlink:
                _append_text_run(
                    container,
                    paragraph,
                    node.latex,
                    styles=inherited_styles,
                    hyperlink=True,
                    context=context,
                )
                continue
            try:
                container.append(latex_to_omml(node.latex, display=False))
            except DocxFormulaError as exc:
                logger.warning("DOCX inline formula fallback: {} ({})", exc, context.location())
                _append_text_run(
                    container,
                    paragraph,
                    node.latex,
                    styles=inherited_styles,
                    hyperlink=False,
                    formula_fallback=True,
                    context=context,
                )
            continue
        if isinstance(node, InlineLink):
            if hyperlink or not node.url or node.url == ".":
                _append_nodes_to_container(
                    container,
                    paragraph,
                    node.children,
                    context=context,
                    inherited_styles=inherited_styles,
                    hyperlink=hyperlink,
                )
                continue
            if node.url.startswith("#"):
                _append_inline_internal_link(
                    container,
                    paragraph,
                    node,
                    context=context,
                    inherited_styles=inherited_styles,
                )
                continue
            _append_external_link(
                paragraph,
                node,
                context=context,
                inherited_styles=inherited_styles,
            )
            continue
        raise TypeError(f"Unsupported inline node: {type(node).__name__}")


def _append_inline_internal_link(
    container: etree._Element,
    paragraph: Paragraph,
    node: InlineLink,
    *,
    context: InlineRenderContext,
    inherited_styles: tuple[str, ...],
) -> None:
    """把 #anchor 行内链接写为 Word bookmark 跳转，未知目标退化为普通文本。"""
    anchor = node.url[1:].strip()
    bookmark_name = context.bookmarks.resolve(anchor)
    if bookmark_name is None:
        _append_nodes_to_container(
            container,
            paragraph,
            node.children,
            context=context,
            inherited_styles=inherited_styles,
            hyperlink=False,
        )
        return

    internal_link = OxmlElement("w:hyperlink")
    internal_link.set(qn("w:anchor"), bookmark_name)
    internal_link.set(qn("w:history"), "1")
    container.append(internal_link)
    _append_nodes_to_container(
        internal_link,
        paragraph,
        node.children,
        context=context,
        inherited_styles=inherited_styles,
        hyperlink=True,
    )


def _append_external_link(
    paragraph: Paragraph,
    node: InlineLink,
    *,
    context: InlineRenderContext,
    inherited_styles: tuple[str, ...],
) -> None:
    """创建外部 hyperlink relationship，并写入完整标签内容。"""
    relation_id = paragraph.part.relate_to(
        sanitize_xml_text(node.url, context=context),
        RELATIONSHIP_TYPE.HYPERLINK,
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relation_id)
    paragraph._p.append(hyperlink)
    _append_nodes_to_container(
        hyperlink,
        paragraph,
        node.children,
        context=context,
        inherited_styles=inherited_styles,
        hyperlink=True,
    )


def _append_text_run(
    container: etree._Element,
    paragraph: Paragraph,
    content: str,
    *,
    styles: tuple[str, ...],
    hyperlink: bool,
    context: InlineRenderContext,
    formula_fallback: bool = False,
) -> Run:
    """向 XML 容器追加一个 run，并应用 MiddleJson 行内样式。"""
    run_xml = OxmlElement("w:r")
    container.append(run_xml)
    run = Run(run_xml, paragraph)
    sanitized = sanitize_xml_text(content, context=context)
    run.text = _make_visible_style_spaces(sanitized, styles)
    _apply_run_styles(run, styles)
    if hyperlink:
        run.font.color.rgb = _rgb_color("0563C1")
        run.underline = True
    if formula_fallback:
        run.font.name = "Courier New"
        run.font.size = Pt(9)
        run.font.color.rgb = _rgb_color("555555")
    return run


def _make_visible_style_spaces(content: str, styles: tuple[str, ...]) -> str:
    """把可见样式 run 的边界 ASCII 空格等量转为 NBSP，避免 Word 隐藏装饰线。"""
    if not content or not _VISIBLE_SPACE_STYLES.intersection(styles):
        return content

    leading_count = len(content) - len(content.lstrip(" "))
    trailing_count = len(content) - len(content.rstrip(" "))
    if leading_count == 0 and trailing_count == 0:
        return content
    if leading_count + trailing_count >= len(content):
        return _NONBREAKING_SPACE * len(content)

    content_end = len(content) - trailing_count if trailing_count else len(content)
    return _NONBREAKING_SPACE * leading_count + content[leading_count:content_end] + _NONBREAKING_SPACE * trailing_count


def sanitize_xml_text(content: str, *, context: InlineRenderContext) -> str:
    """把 XML 1.0 禁止字符替换为 U+FFFD，并记录可定位的 renderer 告警。"""
    sanitized, replacement_count = _INVALID_XML_TEXT_RE.subn("\ufffd", content)
    if replacement_count:
        logger.warning(
            "DOCX replaced {} XML-incompatible character(s) with U+FFFD ({})",
            replacement_count,
            context.location(),
        )
    return sanitized


def _apply_run_styles(run: Run, styles: tuple[str, ...]) -> None:
    """把已解析的行内样式应用到 Word run。"""
    style_set = set(styles)
    run.bold = "bold" in style_set
    run.italic = "italic" in style_set
    run.underline = "underline" in style_set
    run.font.strike = "strikethrough" in style_set
    run.font.superscript = "superscript" in style_set
    run.font.subscript = "subscript" in style_set and "superscript" not in style_set
    if "emphasis" in style_set:
        run_properties = run._element.get_or_add_rPr()
        emphasis = OxmlElement("w:em")
        emphasis.set(qn("w:val"), "underDot")
        run_properties.append(emphasis)


def _rgb_color(value: str) -> RGBColor:
    """延迟构造 RGBColor，避免在模块常量阶段加载额外对象。"""
    return RGBColor.from_string(value)


def plain_inline_text(content: str) -> str:
    """提取一段 MiddleJson 行内内容的可见文本。"""
    return inline_plain_text(parse_inline_content(content))


__all__ = [
    "BookmarkRegistry",
    "InlineRenderContext",
    "append_inline_content",
    "append_inline_nodes",
    "append_internal_link",
    "append_joined_inline_contents",
    "plain_inline_text",
    "sanitize_xml_text",
]
