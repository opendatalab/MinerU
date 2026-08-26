# Copyright (c) Opendatalab. All rights reserved.
"""把 EPUB3 navigation、EPUB2 NCX 或正文标题转换为 raw IndexBlock。"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from loguru import logger
from lxml import etree  # type: ignore[reportMissingImports]

from ....types import BlockType
from .errors import EpubEncryptedError, EpubParseError, EpubResourceLimitError
from .package import EpubPackage
from .xhtml import EpubAnchorRegistry, EpubHeadingEntry


_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class TocEntry:
    """保存一个目录标签、可选正文 anchor 和嵌套子项。"""

    label: str
    anchor: str | None
    children: tuple[TocEntry, ...] = ()


@dataclass(slots=True)
class _MutableTocEntry:
    """供 heading level stack 构造目录树使用的可变节点。"""

    label: str
    anchor: str | None
    children: list[_MutableTocEntry]


def _local_name(element: etree._Element) -> str:
    """返回 navigation/NCX 元素不含命名空间的小写本地名。"""
    return etree.QName(element).localname.casefold()


def _epub_types(element: etree._Element) -> frozenset[str]:
    """读取 EPUB 命名空间或未命名 type 属性中的结构语义 token。"""
    values: list[str] = []
    for name, value in element.attrib.items():
        local_name = etree.QName(name).localname if name.startswith("{") else name.split(":", 1)[-1]
        if local_name == "type":
            values.extend(value.casefold().split())
    return frozenset(values)


def _normalized_text(value: str) -> str:
    """折叠目录标签空白并返回可见文本。"""
    return _WHITESPACE_RE.sub(" ", html.unescape(value)).strip()


def _label_text(element: etree._Element) -> str:
    """提取目录标签文本，并用图片 alt/title 代替非文本内容。"""
    parts: list[str] = []

    def walk(node: etree._Element) -> None:
        """按 DOM 顺序收集文本和图片替代文本。"""
        if node.text:
            parts.append(node.text)
        for child in node:
            if not isinstance(child.tag, str):
                entity_name = getattr(child, "name", "")
                if entity_name:
                    parts.append(html.unescape(f"&{entity_name};"))
                if child.tail:
                    parts.append(child.tail)
                continue
            name = _local_name(child)
            if name in {"img", "image"}:
                alternative = child.get("alt") or child.get("title") or ""
                if alternative:
                    parts.append(alternative)
            elif name not in {"ol", "ul"}:
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(element)
    return _normalized_text("".join(parts))


def _direct_children(element: etree._Element, name: str) -> list[etree._Element]:
    """返回指定本地名的直接普通子元素。"""
    return [
        child
        for child in element
        if isinstance(child.tag, str) and _local_name(child) == name
    ]


def _parse_nav_list(
    list_element: etree._Element,
    *,
    base_part: str,
    anchors: EpubAnchorRegistry,
) -> tuple[TocEntry, ...]:
    """递归解析 EPUB3 toc nav 的 ol/li 层级。"""
    entries: list[TocEntry] = []
    for item in _direct_children(list_element, "li"):
        label_element = next(
            (
                child
                for child in item
                if isinstance(child.tag, str) and _local_name(child) in {"a", "span"}
            ),
            None,
        )
        nested_lists = [
            child
            for child in item
            if isinstance(child.tag, str) and _local_name(child) in {"ol", "ul"}
        ]
        children = tuple(
            child_entry
            for nested in nested_lists
            for child_entry in _parse_nav_list(nested, base_part=base_part, anchors=anchors)
        )
        if label_element is None:
            entries.extend(children)
            continue
        label = _label_text(label_element)
        if not label:
            entries.extend(children)
            continue
        href = label_element.get("href") or ""
        anchor = anchors.resolve_anchor(href, base_part=base_part) if href else None
        entries.append(TocEntry(label, anchor, children))
    return tuple(entries)


def _parse_epub3_navigation(
    package: EpubPackage,
    anchors: EpubAnchorRegistry,
) -> tuple[TocEntry, ...]:
    """读取 EPUB3 navigation document 中唯一需要的主 toc nav。"""
    if package.navigation_path is None:
        return ()
    try:
        root = package.xml_part(package.navigation_path, allow_external_doctype=True)
    except EpubResourceLimitError:
        raise
    except (EpubEncryptedError, EpubParseError) as exc:
        logger.warning("Skipping unusable EPUB navigation document {!r}: {}", package.navigation_path, exc)
        return ()
    if root is None:
        return ()
    toc_nav = next(
        (
            element
            for element in root.iter()
            if isinstance(element.tag, str)
            and _local_name(element) == "nav"
            and (
                "toc" in _epub_types(element)
                or "doc-toc" in (element.get("role") or "").casefold().split()
            )
        ),
        None,
    )
    if toc_nav is None:
        return ()
    root_list = next(
        (
            element
            for element in toc_nav
            if isinstance(element.tag, str) and _local_name(element) in {"ol", "ul"}
        ),
        None,
    )
    return _parse_nav_list(root_list, base_part=package.navigation_path, anchors=anchors) if root_list is not None else ()


def _parse_ncx_point(
    point: etree._Element,
    *,
    base_part: str,
    anchors: EpubAnchorRegistry,
) -> TocEntry | None:
    """递归解析一个 NCX navPoint。"""
    label_container = next(iter(_direct_children(point, "navlabel")), None)
    label_element = (
        next(
            (
                element
                for element in label_container.iter()
                if isinstance(element.tag, str) and _local_name(element) == "text"
            ),
            None,
        )
        if label_container is not None
        else None
    )
    label = _label_text(label_element) if label_element is not None else ""
    children = tuple(
        child_entry
        for child in _direct_children(point, "navpoint")
        if (child_entry := _parse_ncx_point(child, base_part=base_part, anchors=anchors)) is not None
    )
    if not label:
        return TocEntry("", None, children) if children else None
    content = next(iter(_direct_children(point, "content")), None)
    src = content.get("src") if content is not None else ""
    anchor = anchors.resolve_anchor(src or "", base_part=base_part) if src else None
    return TocEntry(label, anchor, children)


def _flatten_empty_ncx_entries(entries: tuple[TocEntry, ...]) -> tuple[TocEntry, ...]:
    """移除 NCX 无标签容器，同时保留其有效子项。"""
    result: list[TocEntry] = []
    for entry in entries:
        children = _flatten_empty_ncx_entries(entry.children)
        if entry.label:
            result.append(TocEntry(entry.label, entry.anchor, children))
        else:
            result.extend(children)
    return tuple(result)


def _parse_epub2_ncx(
    package: EpubPackage,
    anchors: EpubAnchorRegistry,
) -> tuple[TocEntry, ...]:
    """读取 EPUB2 NCX navMap，并保留嵌套 navPoint。"""
    if package.ncx_path is None:
        return ()
    try:
        root = package.xml_part(package.ncx_path, allow_external_doctype=True)
    except EpubResourceLimitError:
        raise
    except (EpubEncryptedError, EpubParseError) as exc:
        logger.warning("Skipping unusable EPUB NCX {!r}: {}", package.ncx_path, exc)
        return ()
    if root is None:
        return ()
    nav_map = next(
        (
            element
            for element in root.iter()
            if isinstance(element.tag, str) and _local_name(element) == "navmap"
        ),
        None,
    )
    if nav_map is None:
        return ()
    parsed = tuple(
        entry
        for point in _direct_children(nav_map, "navpoint")
        if (entry := _parse_ncx_point(point, base_part=package.ncx_path, anchors=anchors)) is not None
    )
    return _flatten_empty_ncx_entries(parsed)


def _freeze_heading_entry(entry: _MutableTocEntry) -> TocEntry:
    """把可变 heading 节点递归冻结为 TocEntry。"""
    return TocEntry(entry.label, entry.anchor, tuple(_freeze_heading_entry(child) for child in entry.children))


def _toc_from_headings(headings: tuple[EpubHeadingEntry, ...]) -> tuple[TocEntry, ...]:
    """按 heading level stack 构造目录树，跨级时不生成空节点。"""
    roots: list[_MutableTocEntry] = []
    stack: list[tuple[int, _MutableTocEntry]] = []
    for heading in headings:
        entry = _MutableTocEntry(heading.label, heading.anchor, [])
        while stack and heading.level <= stack[-1][0]:
            stack.pop()
        if stack:
            stack[-1][1].children.append(entry)
        else:
            roots.append(entry)
        stack.append((heading.level, entry))
    return tuple(_freeze_heading_entry(entry) for entry in roots)


def _raw_index_content(entries: tuple[TocEntry, ...], *, depth: int) -> list[dict[str, object]]:
    """把 TocEntry 树转换为现有 raw IndexBlock 叶子与嵌套结构。"""
    content: list[dict[str, object]] = []
    for entry in entries:
        leaf: dict[str, object] = {
            "type": BlockType.TEXT,
            "content": html.escape(entry.label, quote=False),
        }
        if entry.anchor:
            leaf["anchor"] = entry.anchor
        content.append(leaf)
        if entry.children:
            content.append(
                {
                    "type": BlockType.INDEX,
                    "ilevel": depth + 1,
                    "content": _raw_index_content(entry.children, depth=depth + 1),
                }
            )
    return content


def build_epub_toc_index(
    package: EpubPackage,
    anchors: EpubAnchorRegistry,
) -> dict[str, object] | None:
    """按 EPUB3 nav、EPUB2 NCX、正文标题优先级构造顶层 raw IndexBlock。"""
    entries = _parse_epub3_navigation(package, anchors)
    if not entries:
        entries = _parse_epub2_ncx(package, anchors)
    if not entries:
        entries = _toc_from_headings(anchors.heading_entries())
    if not entries:
        return None
    return {
        "type": BlockType.INDEX,
        "ilevel": 0,
        "content": _raw_index_content(entries, depth=0),
    }


__all__ = ["TocEntry", "build_epub_toc_index"]
