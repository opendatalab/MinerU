# Copyright (c) Opendatalab. All rights reserved.
"""集中建立静态 HTML/XHTML 标题、脚注与 fragment anchor 索引。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import re
from typing import Literal, Protocol, TypeAlias

from lxml import etree  # type: ignore[reportMissingImports]

from .projector import local_name, visible_raw_text_with_style
from .styles import MarkupStylesheet, TextStyle


AnchorVisibilityScope: TypeAlias = Literal["all_ancestors", "nearest_body"]
AnchorTextNormalization: TypeAlias = Literal["unicode_whitespace", "xhtml_whitespace"]

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_XHTML_WHITESPACE_RE = re.compile(r"[\t\r\n\f ]+")
_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


@dataclass(frozen=True, slots=True)
class MarkupAnchorDocument:
    """描述一个待建立 anchor 索引的 DOM、样式表与兼容可见性规则。"""

    key: str
    root: etree._Element
    stylesheet: MarkupStylesheet
    visibility_scope: AnchorVisibilityScope = "all_ancestors"
    text_normalization: AnchorTextNormalization = "unicode_whitespace"


class MarkupAnchorPolicy(Protocol):
    """定义格式适配器生成标题、脚注 anchor 所需的稳定策略。"""

    anchor_prefix: str
    register_document_start: bool

    def heading_identity(self, element: etree._Element, ordinal: int) -> str:
        """返回当前标题参与稳定摘要的格式专属 identity。"""

    def is_materializable_note(self, element: etree._Element, document: MarkupAnchorDocument) -> bool:
        """判断当前元素是否是能够兑现文本 anchor 的格式专属脚注。"""

    def note_identity(self, element: etree._Element, ordinal: int) -> str:
        """返回当前脚注参与稳定摘要的格式专属 identity。"""


def element_id(element: etree._Element) -> str | None:
    """返回元素去除首尾空白后的 HTML id 或 xml:id。"""
    value = (element.get("id") or element.get(_XML_ID) or "").strip()
    return value or None


def visible_element_text(element: etree._Element, document: MarkupAnchorDocument) -> str:
    """按文档兼容配置解析祖先样式链，并返回最终可输出的纯文本。"""
    inherited = TextStyle()
    visibility_hidden = False
    chain = [ancestor for ancestor in reversed(list(element.iterancestors())) if isinstance(ancestor.tag, str)]
    if document.visibility_scope == "nearest_body":
        body_index = next((index for index, ancestor in enumerate(chain) if local_name(ancestor) == "body"), None)
        if body_index is not None:
            chain = chain[body_index:]
    chain.append(element)
    for current in chain:
        resolved = document.stylesheet.resolve(current, inherited, visibility_hidden)
        if resolved.subtree_hidden:
            return ""
        inherited = resolved.text
        visibility_hidden = resolved.visibility_hidden
    value = visible_raw_text_with_style(element, document.stylesheet, inherited, visibility_hidden)
    if document.text_normalization == "xhtml_whitespace":
        return _XHTML_WHITESPACE_RE.sub(" ", html.unescape(value)).strip()
    return " ".join(value.split())


def canonical_anchor(prefix: str, document_key: str, identity: str) -> str:
    """按格式前缀、文档 key 与 identity 生成稳定的二十位摘要 anchor。"""
    digest = hashlib.sha256(f"{document_key}#{identity}".encode()).hexdigest()[:20]
    return f"{prefix}-{digest}"


class MarkupAnchorRegistry:
    """统一登记多文档标题、脚注及源 fragment 到实际输出 anchor 的映射。"""

    def __init__(self, documents: list[MarkupAnchorDocument], policy: MarkupAnchorPolicy) -> None:
        """按调用方文档顺序建立稳定索引，并保留格式专属 identity 规则。"""
        self._policy = policy
        self._heading_anchors: dict[etree._Element, str] = {}
        self._note_anchors: dict[etree._Element, str] = {}
        self._targets: dict[tuple[str, str | None], str] = {}
        self._heading_labels: dict[str, str] = {}
        for document in documents:
            self._register_document(document)

    def _register_document(self, document: MarkupAnchorDocument) -> None:
        """登记单个 DOM 的标题、脚注及全部可解析 fragment 别名。"""
        headings: list[tuple[etree._Element, str]] = []
        for element in document.root.iter():
            if not isinstance(element.tag, str) or local_name(element) not in _HEADING_TAGS:
                continue
            if label := visible_element_text(element, document):
                headings.append((element, label))
        for ordinal, (heading, label) in enumerate(headings):
            identity = self._policy.heading_identity(heading, ordinal)
            anchor = canonical_anchor(self._policy.anchor_prefix, document.key, identity)
            self._heading_anchors[heading] = anchor
            self._heading_labels[anchor] = label

        notes = [
            element
            for element in document.root.iter()
            if isinstance(element.tag, str) and self._policy.is_materializable_note(element, document)
        ]
        for ordinal, note in enumerate(notes):
            identity = self._policy.note_identity(note, ordinal)
            self._note_anchors[note] = canonical_anchor(self._policy.anchor_prefix, document.key, identity)

        if self._policy.register_document_start and headings:
            self._targets[(document.key, None)] = self._heading_anchors[headings[0][0]]
        for element in document.root.iter():
            if not isinstance(element.tag, str) or not (fragment := element_id(element)):
                continue
            target_key = (document.key, fragment)
            if target_key in self._targets:
                continue
            if anchor := self._target_anchor(element):
                self._targets[target_key] = anchor

    def _target_anchor(self, element: etree._Element) -> str | None:
        """把任意 fragment 元素映射到自身、最近祖先或首个后代输出目标。"""
        direct = self._heading_anchors.get(element) or self._note_anchors.get(element)
        if direct is not None:
            return direct
        ancestor = next(
            (parent for parent in element.iterancestors() if parent in self._heading_anchors or parent in self._note_anchors),
            None,
        )
        if ancestor is not None:
            return self._heading_anchors.get(ancestor) or self._note_anchors.get(ancestor)
        descendant = next(
            (
                child
                for child in element.iterdescendants()
                if isinstance(child.tag, str) and (child in self._heading_anchors or child in self._note_anchors)
            ),
            None,
        )
        if descendant is None:
            return None
        return self._heading_anchors.get(descendant) or self._note_anchors.get(descendant)

    def heading_anchor(self, heading: etree._Element) -> str | None:
        """返回一个已登记标题的规范 anchor。"""
        return self._heading_anchors.get(heading)

    def heading_label(self, anchor: str) -> str | None:
        """返回规范标题 anchor 对应的可见标题文本。"""
        return self._heading_labels.get(anchor)

    def note_anchor(self, note: etree._Element) -> str | None:
        """返回一个已登记脚注的规范 anchor。"""
        return self._note_anchors.get(note)

    def resolve_target(self, document_key: str, fragment: str | None) -> str | None:
        """按文档 key 与可选源 fragment 返回不带井号的规范 anchor。"""
        return self._targets.get((document_key, fragment))


__all__ = [
    "AnchorTextNormalization",
    "AnchorVisibilityScope",
    "MarkupAnchorDocument",
    "MarkupAnchorPolicy",
    "MarkupAnchorRegistry",
    "canonical_anchor",
    "element_id",
    "visible_element_text",
]
