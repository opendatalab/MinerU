# Copyright (c) Opendatalab. All rights reserved.
"""Standalone HTML 标题、脚注与 fragment anchor 规范化。"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import hashlib

from lxml import etree  # type: ignore[reportMissingImports]

from .._shared.markup import MarkupStylesheet, TextStyle
from .._shared.markup.projector import local_name, visible_raw_text_with_style


_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_NOTE_TYPES = frozenset({"footnote", "endnote", "rearnote"})
_NOTE_ROLES = frozenset({"doc-footnote", "doc-endnote"})
_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


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


def element_id(element: etree._Element) -> str | None:
    """返回元素的 HTML id 或 xml:id。"""
    value = (element.get("id") or element.get(_XML_ID) or "").strip()
    return value or None


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
        self._heading_anchors: dict[etree._Element, str] = {}
        self._note_anchors: dict[etree._Element, str] = {}
        self._heading_labels: dict[str, str] = {}
        self._targets: dict[str, str] = {}
        headings = [
            element
            for element in root.iter()
            if isinstance(element.tag, str)
            and local_name(element) in _HEADING_TAGS
            and _visible_element_text(element, stylesheet)
        ]
        for ordinal, heading in enumerate(headings):
            identity = element_id(heading) or f"heading-{ordinal}"
            anchor = _canonical_anchor(source_key, f"heading-{identity}-{ordinal}")
            self._heading_anchors[heading] = anchor
            self._heading_labels[anchor] = _visible_element_text(heading, stylesheet)
        notes = [
            element
            for element in root.iter()
            if isinstance(element.tag, str) and is_note_element(element) and _visible_element_text(element, stylesheet)
        ]
        for ordinal, note in enumerate(notes):
            identity = element_id(note) or f"note-{ordinal}"
            self._note_anchors[note] = _canonical_anchor(source_key, f"note-{identity}-{ordinal}")
        for element in root.iter():
            if not isinstance(element.tag, str) or not (identity := element_id(element)) or identity in self._targets:
                continue
            if anchor := self._target_anchor(element):
                self._targets[identity] = anchor

    def _target_anchor(self, element: etree._Element) -> str | None:
        """把任意 fragment 元素映射到自身、祖先或后代的可输出目标。"""
        direct = self._heading_anchors.get(element) or self._note_anchors.get(element)
        if direct:
            return direct
        for parent in element.iterancestors():
            anchor = self._heading_anchors.get(parent) or self._note_anchors.get(parent)
            if anchor:
                return anchor
        for child in element.iterdescendants():
            anchor = self._heading_anchors.get(child) or self._note_anchors.get(child)
            if anchor:
                return anchor
        return None

    def heading_anchor(self, heading: etree._Element) -> str | None:
        """返回标题的规范 anchor。"""
        return self._heading_anchors.get(heading)

    def heading_label(self, anchor: str) -> str | None:
        """返回规范标题 anchor 对应的可见标签。"""
        return self._heading_labels.get(anchor)

    def note_anchor(self, note: etree._Element) -> str | None:
        """返回单条 Footnote/Endnote 的规范 anchor。"""
        return self._note_anchors.get(note)

    def resolve_fragment(self, fragment: str) -> str | None:
        """把源文档 fragment 转换为实际可输出的内部链接。"""
        normalized = fragment.removeprefix("#").strip()
        anchor = self._targets.get(normalized)
        return f"#{anchor}" if anchor else None


def _canonical_anchor(source_key: str, identity: str) -> str:
    """为 HTML 标题或脚注生成短而稳定的规范 anchor。"""
    digest = hashlib.sha256(f"{source_key}#{identity}".encode()).hexdigest()[:20]
    return f"html-{digest}"


def _visible_element_text(element: etree._Element, stylesheet: MarkupStylesheet) -> str:
    """按祖先样式链提取元素最终会由 projector 输出的可见文本。"""
    inherited = TextStyle()
    visibility_hidden = False
    chain = [ancestor for ancestor in reversed(list(element.iterancestors())) if isinstance(ancestor.tag, str)]
    chain.append(element)
    for current in chain:
        resolved = stylesheet.resolve(current, inherited, visibility_hidden)
        if resolved.subtree_hidden:
            return ""
        inherited = resolved.text
        visibility_hidden = resolved.visibility_hidden
    value = visible_raw_text_with_style(element, stylesheet, inherited, visibility_hidden)
    return " ".join(value.split())


__all__ = ["HtmlAnchorRegistry", "append_referenced_notes", "element_id", "is_note_element"]
