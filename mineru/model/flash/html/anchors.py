# Copyright (c) Opendatalab. All rights reserved.
"""Standalone HTML 标题、脚注与 fragment anchor 规范化。"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

from lxml import etree  # type: ignore[reportMissingImports]

from .._shared.markup import MarkupAnchorDocument, MarkupAnchorRegistry, MarkupStylesheet, TextStyle, element_id
from .._shared.markup.projector import BLOCK_TAGS, SKIPPED_TAGS, local_name, visible_raw_text_with_style


_NOTE_TYPES = frozenset({"footnote", "endnote", "rearnote"})
_NOTE_ROLES = frozenset({"doc-footnote", "doc-endnote"})
_NON_TEXT_BLOCK_TAGS = frozenset(
    {
        "figure",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "math",
        "ol",
        "pre",
        "svg",
        "table",
        "ul",
    }
)


def is_note_element(element: etree._Element) -> bool:
    """判断元素是否表示一条可独立投影的 Footnote/Endnote。"""
    roles = frozenset((element.get("role") or "").casefold().split())
    classes = frozenset((element.get("class") or "").casefold().split())
    if (element.get("data-block-type") or "").casefold() == "page_footnote" or "mineru-page-footnote" in classes:
        return True
    types: set[str] = set()
    for name, value in element.attrib.items():
        attribute = etree.QName(name).localname if name.startswith("{") else name.split(":", 1)[-1]
        if attribute == "type":
            types.update(value.casefold().split())
    return bool(roles & _NOTE_ROLES or types & _NOTE_TYPES)


def append_referenced_notes(
    selected_root: etree._Element,
    original_body: etree._Element,
    *,
    stylesheet: MarkupStylesheet,
    resolve_same_document_fragment: Callable[[str], str | None],
) -> etree._Element:
    """把正文候选引用但位于候选外的脚注副本追加到内容根末尾。"""
    selected_ids = {
        identity
        for element in selected_root.iter()
        if isinstance(element.tag, str) and (identity := element_id(element)) is not None
    }
    targets = {
        identity: element
        for element in original_body.iter()
        if isinstance(element.tag, str) and (identity := element_id(element)) is not None and is_note_element(element)
    }
    referenced_ids = dict.fromkeys(
        fragment
        for element in selected_root.iter()
        if isinstance(element.tag, str)
        and local_name(element) == "a"
        and (fragment := resolve_same_document_fragment(element.get("href") or "")) is not None
    )
    companions = [
        element for identity, element in targets.items() if identity in referenced_ids and identity not in selected_ids
    ]
    companion_copies = [
        copy for companion in companions if (copy := _copy_note_with_source_visibility(companion, stylesheet)) is not None
    ]
    if not companion_copies:
        return selected_root
    wrapper = etree.Element("div")
    wrapper.append(selected_root)
    for companion in companion_copies:
        wrapper.append(companion)
    return wrapper


def _copy_note_with_source_visibility(
    note: etree._Element,
    stylesheet: MarkupStylesheet,
) -> etree._Element | None:
    """按原始祖先链复制 note；整树隐藏时丢弃，并保留继承文字样式与 visibility。"""
    inherited = TextStyle()
    visibility_hidden = False
    chain = [ancestor for ancestor in reversed(list(note.iterancestors())) if isinstance(ancestor.tag, str)]
    for current in chain:
        resolved = stylesheet.resolve(current, inherited, visibility_hidden)
        if resolved.subtree_hidden:
            return None
        inherited = resolved.text
        visibility_hidden = resolved.visibility_hidden
    if stylesheet.resolve(note, inherited, visibility_hidden).subtree_hidden:
        return None
    copied = deepcopy(note)
    declarations: list[str] = []
    if inherited.bold:
        declarations.append("font-weight:bold")
    if inherited.italic:
        declarations.append("font-style:italic")
    decorations = [
        decoration
        for enabled, decoration in (
            (inherited.underline, "underline"),
            (inherited.strikethrough, "line-through"),
        )
        if enabled
    ]
    if decorations:
        declarations.append(f"text-decoration:{' '.join(decorations)}")
    if inherited.superscript:
        declarations.append("vertical-align:super")
    elif inherited.subscript:
        declarations.append("vertical-align:sub")
    if visibility_hidden:
        declarations.append("visibility:hidden")
    if not declarations:
        return copied
    wrapper = etree.Element("div")
    wrapper.set("style", ";".join(declarations))
    if inherited.superscript and inherited.subscript:
        subscript_wrapper = etree.SubElement(wrapper, "sub")
        subscript_wrapper.append(copied)
    else:
        wrapper.append(copied)
    return wrapper


class _HtmlAnchorPolicy:
    """保持 standalone HTML 标题与脚注 identity 的既有生成规则。"""

    anchor_prefix = "html"
    register_document_start = False

    @staticmethod
    def heading_identity(element: etree._Element, ordinal: int) -> str:
        """按源 ID 或匿名标题序号生成 HTML 标题 identity。"""
        identity = element_id(element) or f"heading-{ordinal}"
        return f"heading-{identity}-{ordinal}"

    @staticmethod
    def is_materializable_note(element: etree._Element, document: MarkupAnchorDocument) -> bool:
        """沿用 HTML note marker 与顶层文本可落地性判断。"""
        return is_note_element(element) and _note_has_materializable_text_target(element, document.stylesheet)

    @staticmethod
    def note_identity(element: etree._Element, ordinal: int) -> str:
        """按源 ID 或匿名脚注序号生成 HTML 脚注 identity。"""
        identity = element_id(element) or f"note-{ordinal}"
        return f"note-{identity}-{ordinal}"


class HtmlAnchorRegistry:
    """把选中 DOM 的标题、note 和源 fragment 映射到稳定 anchor。"""

    def __init__(
        self,
        root: etree._Element,
        stylesheet: MarkupStylesheet,
        *,
        source_key: str = "html",
    ) -> None:
        """预扫描选中内容，建立 document-wide 唯一 anchor 映射。"""
        self._source_key = source_key
        document = MarkupAnchorDocument(
            key=source_key,
            root=root,
            stylesheet=stylesheet,
            visibility_scope="all_ancestors",
            text_normalization="unicode_whitespace",
        )
        self._registry = MarkupAnchorRegistry([document], _HtmlAnchorPolicy())

    def heading_anchor(self, heading: etree._Element) -> str | None:
        """返回标题的规范 anchor。"""
        return self._registry.heading_anchor(heading)

    def heading_label(self, anchor: str) -> str | None:
        """返回规范标题 anchor 对应的可见标签。"""
        return self._registry.heading_label(anchor)

    def note_anchor(self, note: etree._Element) -> str | None:
        """返回单条 Footnote/Endnote 的规范 anchor。"""
        return self._registry.note_anchor(note)

    def resolve_fragment(self, fragment: str) -> str | None:
        """把源文档 fragment 转换为实际可输出的内部链接。"""
        normalized = fragment.removeprefix("#").strip()
        anchor = self._registry.resolve_target(self._source_key, normalized)
        return f"#{anchor}" if anchor else None


def _note_has_materializable_text_target(element: etree._Element, stylesheet: MarkupStylesheet) -> bool:
    """判断 note 是否会投影出可挂载 anchor 的顶层文本 block。"""
    inherited = TextStyle()
    visibility_hidden = False
    chain = [ancestor for ancestor in reversed(list(element.iterancestors())) if isinstance(ancestor.tag, str)]
    for ancestor in chain:
        resolved = stylesheet.resolve(ancestor, inherited, visibility_hidden)
        if resolved.subtree_hidden:
            return False
        inherited = resolved.text
        visibility_hidden = resolved.visibility_hidden
    resolved = stylesheet.resolve(element, inherited, visibility_hidden)
    if resolved.subtree_hidden:
        return False
    return _container_materializes_text_block(
        element,
        stylesheet,
        resolved.text,
        resolved.visibility_hidden,
    )


def _container_materializes_text_block(
    element: etree._Element,
    stylesheet: MarkupStylesheet,
    style: TextStyle,
    visibility_hidden: bool,
) -> bool:
    """按共享 projector 的容器分块规则判断是否会产生顶层文本。"""
    if not visibility_hidden and (element.text or "").strip():
        return True
    for child in element:
        if isinstance(child.tag, str):
            resolved = stylesheet.resolve(child, style, visibility_hidden)
            if not resolved.subtree_hidden:
                name = local_name(child)
                if name == "p":
                    value = visible_raw_text_with_style(
                        child,
                        stylesheet,
                        resolved.text,
                        resolved.visibility_hidden,
                    )
                    if value.strip():
                        return True
                elif name in BLOCK_TAGS:
                    if name not in _NON_TEXT_BLOCK_TAGS and _container_materializes_text_block(
                        child,
                        stylesheet,
                        resolved.text,
                        resolved.visibility_hidden,
                    ):
                        return True
                elif name not in SKIPPED_TAGS:
                    value = visible_raw_text_with_style(
                        child,
                        stylesheet,
                        resolved.text,
                        resolved.visibility_hidden,
                    )
                    if value.strip():
                        return True
        if not visibility_hidden and (child.tail or "").strip():
            return True
    return False


__all__ = ["HtmlAnchorRegistry", "append_referenced_notes", "is_note_element"]
