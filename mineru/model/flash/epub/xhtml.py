# Copyright (c) Opendatalab. All rights reserved.
"""把 EPUB XHTML/SVG 内容文档转换为 MinerU raw blocks。"""

from __future__ import annotations

import base64
import hashlib
import html
import re
from dataclasses import dataclass

from lxml import etree  # type: ignore[reportMissingImports]

from ....utils.image_payload import parse_image_data_uri_strict
from .._shared.hyperlink import sanitize_hyperlink_target
from .._shared.markup import MarkupProjector, MarkupStylesheet, ResolvedMarkupImage, TextStyle
from .._shared.markup.projector import (
    BLOCK_TAGS as _BLOCK_TAGS,
    SKIPPED_TAGS as _SKIPPED_TAGS,
    clean_text_node as _clean_text_node,
    entity_text as _entity_text,
    local_name as _local_name,
    visible_raw_text_with_style as _visible_raw_text_with_style,
)
from .constants import IMAGE_MEDIA_BY_EXTENSION, SVG_MEDIA_TYPE
from .package import EpubPackage

_INDIVIDUAL_NOTE_TYPE_ORDER = ("footnote", "endnote", "rearnote")
_INDIVIDUAL_NOTE_ROLE_ORDER = ("doc-footnote", "doc-endnote")
_INDIVIDUAL_NOTE_TYPES = frozenset(_INDIVIDUAL_NOTE_TYPE_ORDER)
_INDIVIDUAL_NOTE_ROLES = frozenset(_INDIVIDUAL_NOTE_ROLE_ORDER)
_NOTE_BLOCK_TAGS = _BLOCK_TAGS | {"li"}
_NOTE_NON_TEXT_SUBTREES = frozenset(
    {
        "figure",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "math",
        "ol",
        "pre",
        "svg",
        "table",
        "ul",
    }
)
_WHITESPACE_RE = re.compile(r"[\t\r\n\f ]+")
_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def _element_id(element: etree._Element) -> str | None:
    """返回元素的 HTML id 或 xml:id。"""
    value = (element.get("id") or element.get(_XML_ID) or "").strip()
    return value or None


def _epub_types(element: etree._Element) -> frozenset[str]:
    """读取 EPUB 命名空间或未命名 type 属性中的结构语义 token。"""
    values: list[str] = []
    for name, value in element.attrib.items():
        local_name = etree.QName(name).localname if name.startswith("{") else name.split(":", 1)[-1]
        if local_name == "type":
            values.extend(value.casefold().split())
    return frozenset(values)


def _roles(element: etree._Element) -> frozenset[str]:
    """读取 ARIA role 属性中的小写语义 token。"""
    return frozenset((element.get("role") or "").casefold().split())


def _is_individual_note(element: etree._Element) -> bool:
    """判断块级元素是否表示单条 EPUB Footnote/Endnote。"""
    if _local_name(element) not in _NOTE_BLOCK_TAGS:
        return False
    return bool(_epub_types(element) & _INDIVIDUAL_NOTE_TYPES or _roles(element) & _INDIVIDUAL_NOTE_ROLES)


def _note_semantic(element: etree._Element) -> str:
    """按固定优先级返回 note 的 EPUB type 或 ARIA role。"""
    epub_types = _epub_types(element)
    for note_type in _INDIVIDUAL_NOTE_TYPE_ORDER:
        if note_type in epub_types:
            return note_type
    roles = _roles(element)
    for role in _INDIVIDUAL_NOTE_ROLE_ORDER:
        if role in roles:
            return role
    return "note"


def _note_has_text_block(element: etree._Element) -> bool:
    """判断 note 是否能产生非空 text，从而避免注册没有正文目标的 anchor。"""
    if _clean_text_node(element.text).strip():
        return True
    for child in element:
        if child.tail and _clean_text_node(child.tail).strip():
            return True
        if not isinstance(child.tag, str):
            if _entity_text(child):
                return True
            continue
        name = _local_name(child)
        if name in _SKIPPED_TAGS or name in _NOTE_NON_TEXT_SUBTREES or name in {"img", "image"}:
            continue
        if _note_has_text_block(child):
            return True
    return False


def _canonical_anchor(chapter_path: str, identity: str) -> str:
    """为章节内标题或 note 生成稳定且适合各 renderer 的短锚点。"""
    digest = hashlib.sha256(f"{chapter_path}#{identity}".encode("utf-8")).hexdigest()[:20]
    return f"epub-{digest}"


@dataclass(frozen=True, slots=True)
class _ChapterTree:
    """保存一个已解析 spine XHTML 内容树。"""

    path: str
    root: etree._Element


def _load_chapter_stylesheet(package: EpubPackage, chapter_path: str, root: etree._Element) -> MarkupStylesheet:
    """按章节 head 顺序加载包内 CSS 与内联 style。"""
    stylesheet = MarkupStylesheet()
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        name = _local_name(element)
        if name == "link" and "stylesheet" in (element.get("rel") or "").casefold().split():
            target = package.resolve_reference(element.get("href") or "", base_part=chapter_path)
            if target is None:
                continue
            data = package.read_part(target.path)
            if data is not None:
                stylesheet.add(data.decode("utf-8-sig", errors="replace"))
        elif name == "style":
            stylesheet.add("".join(element.itertext()))
    return stylesheet


def _element_visible_text(element: etree._Element, stylesheet: MarkupStylesheet) -> str:
    """按祖先到自身的样式链返回元素最终可输出的纯文本。"""
    inherited = TextStyle()
    inherited_visibility_hidden = False
    chain = [ancestor for ancestor in reversed(list(element.iterancestors())) if isinstance(ancestor.tag, str)]
    body_index = next((index for index, ancestor in enumerate(chain) if _local_name(ancestor) == "body"), None)
    if body_index is not None:
        chain = chain[body_index:]
    chain.append(element)
    for current in chain:
        resolved = stylesheet.resolve(current, inherited, inherited_visibility_hidden)
        if resolved.subtree_hidden:
            return ""
        inherited = resolved.text
        inherited_visibility_hidden = resolved.visibility_hidden
    value = _visible_raw_text_with_style(element, stylesheet, inherited, inherited_visibility_hidden)
    return _WHITESPACE_RE.sub(" ", html.unescape(value)).strip()


def _element_is_hidden(element: etree._Element, stylesheet: MarkupStylesheet) -> bool:
    """判断元素是否不会产生任何最终可输出的文本。"""
    return not _element_visible_text(element, stylesheet)


class EpubAnchorRegistry:
    """建立章节路径、标题与 note fragment 到实际 canonical anchor 的别名表。"""

    def __init__(self, chapters: list[_ChapterTree], package: EpubPackage) -> None:
        """预扫描全部选中 XHTML 章节，建立标题、note 与章节起点映射。"""
        self._package = package
        self._heading_anchors: dict[etree._Element, str] = {}
        self._note_anchors: dict[etree._Element, str] = {}
        self._targets: dict[tuple[str, str | None], str] = {}
        self._heading_labels: dict[str, str] = {}
        for chapter in chapters:
            stylesheet = _load_chapter_stylesheet(package, chapter.path, chapter.root)
            headings: list[tuple[etree._Element, str]] = []
            for element in chapter.root.iter():
                if not isinstance(element.tag, str) or _local_name(element) not in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                    continue
                if label := _element_visible_text(element, stylesheet):
                    headings.append((element, label))
            for ordinal, (heading, label) in enumerate(headings):
                source_id = _element_id(heading)
                identity = f"{source_id or 'heading'}-{ordinal}"
                anchor = _canonical_anchor(chapter.path, identity)
                self._heading_anchors[heading] = anchor
                self._heading_labels[anchor] = label
            notes = [
                element
                for element in chapter.root.iter()
                if isinstance(element.tag, str)
                and _is_individual_note(element)
                and _note_has_text_block(element)
                and not _element_is_hidden(element, stylesheet)
            ]
            for ordinal, note in enumerate(notes):
                source_id = _element_id(note)
                note_type = _note_semantic(note)
                identity = f"note-{note_type}-{source_id or 'anonymous'}-{ordinal}"
                self._note_anchors[note] = _canonical_anchor(chapter.path, identity)

            if headings:
                first_anchor = self._heading_anchors[headings[0][0]]
                self._targets[(chapter.path, None)] = first_anchor
            for element in chapter.root.iter():
                if not isinstance(element.tag, str) or not (fragment := _element_id(element)):
                    continue
                if (chapter.path, fragment) in self._targets:
                    continue
                target_anchor = self._target_anchor_for_element(element)
                if target_anchor is not None:
                    self._targets[(chapter.path, fragment)] = target_anchor

    def _target_anchor_for_element(self, element: etree._Element) -> str | None:
        """把任意 fragment 元素映射到自身、祖先或后代的可输出标题与脚注。"""
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
        """返回一个已预扫描标题的规范 anchor。"""
        return self._heading_anchors.get(heading)

    def heading_label(self, anchor: str) -> str | None:
        """返回 canonical 标题 anchor 对应的可见标题文本。"""
        return self._heading_labels.get(anchor)

    def note_anchor(self, note: etree._Element) -> str | None:
        """返回一个已预扫描 Footnote/Endnote 的 canonical anchor。"""
        return self._note_anchors.get(note)

    def resolve_anchor(self, href: str, *, base_part: str) -> str | None:
        """解析指向正文标题或 note 的 EPUB 包内链接，并返回不带井号的 anchor。"""
        normalized = sanitize_hyperlink_target(
            href,
            allowed_schemes=(),
            allow_relative=True,
            allow_fragment=True,
        )
        if normalized is None:
            return None
        target = self._package.resolve_reference(normalized, base_part=base_part)
        if target is None:
            return None
        return self._targets.get((target.path, target.fragment))

    def resolve_link(self, href: str, *, base_part: str) -> str | None:
        """解析安全外部链接或指向已输出标题/note 的 EPUB 内部链接。"""
        external = sanitize_hyperlink_target(href)
        if external is not None:
            return external
        anchor = self.resolve_anchor(href, base_part=base_part)
        return f"#{anchor}" if anchor else None


def build_anchor_registry(chapters: list[tuple[str, etree._Element]], package: EpubPackage) -> EpubAnchorRegistry:
    """从已解析章节元组建立跨章节锚点注册表。"""
    return EpubAnchorRegistry([_ChapterTree(path, root) for path, root in chapters], package)


@dataclass(frozen=True, slots=True)
class _EpubMarkupContext:
    """把 EPUB 包资源与文档级 anchor 适配到共享 markup projector。"""

    package: EpubPackage
    chapter_path: str
    anchors: EpubAnchorRegistry

    def resolve_link(self, href: str) -> str | None:
        """解析安全外部链接或实际存在的 EPUB 包内 anchor。"""
        return self.anchors.resolve_link(href, base_part=self.chapter_path)

    def resolve_image(self, source: str, *, alt: str = "") -> ResolvedMarkupImage | None:
        """读取并严格校验一个 EPUB 包内栅格图片引用。"""
        target = self.package.resolve_reference(source, base_part=self.chapter_path)
        if target is None:
            return ResolvedMarkupImage(alt=alt) if alt else None
        media_type = (self.package.content_type_for(target.path) or "").casefold()
        extension = target.path.rsplit(".", 1)[-1].casefold() if "." in target.path else ""
        media_type = media_type or IMAGE_MEDIA_BY_EXTENSION.get(extension, "")
        if not media_type.startswith("image/") or media_type == SVG_MEDIA_TYPE:
            return ResolvedMarkupImage(alt=alt) if alt else None
        payload = self.package.read_part(target.path, asset=True)
        if payload is None:
            return ResolvedMarkupImage(alt=alt) if alt else None
        data_uri = f"data:{media_type};base64,{base64.b64encode(payload).decode('ascii')}"
        try:
            parse_image_data_uri_strict(data_uri)
        except ValueError:
            return ResolvedMarkupImage(alt=alt) if alt else None
        return ResolvedMarkupImage(image_base64=data_uri, alt=alt)

    def heading_anchor(self, heading: etree._Element) -> str | None:
        """返回一个已预扫描 EPUB 标题的规范 anchor。"""
        return self.anchors.heading_anchor(heading)

    def heading_label(self, anchor: str) -> str | None:
        """返回规范 EPUB 标题 anchor 对应的可见标签。"""
        return self.anchors.heading_label(anchor)

    def note_anchor(self, note: etree._Element) -> str | None:
        """返回一个已预扫描 EPUB Footnote/Endnote anchor。"""
        return self.anchors.note_anchor(note)


class EpubChapterConverter:
    """把一个 XHTML spine item 通过共享 projector 投影为 raw blocks。"""

    def __init__(
        self,
        package: EpubPackage,
        chapter_path: str,
        root: etree._Element,
        anchors: EpubAnchorRegistry,
    ) -> None:
        """绑定单个章节的包、路径、DOM 与文档级 anchor 注册表。"""
        self.package = package
        self.chapter_path = chapter_path
        self.root = root
        self.anchors = anchors
        self.stylesheet = _load_chapter_stylesheet(package, chapter_path, root)

    def convert(self) -> list[dict[str, object]]:
        """解析 XHTML body，并保持既有 EPUB 标题与脚注语义。"""
        body = next(
            (element for element in self.root.iter() if isinstance(element.tag, str) and _local_name(element) == "body"),
            None,
        )
        if body is None:
            return []
        context = _EpubMarkupContext(self.package, self.chapter_path, self.anchors)
        return MarkupProjector(
            body,
            context,
            self.stylesheet,
            single_document_title=False,
        ).convert()


def convert_svg_spine(
    package: EpubPackage,
    chapter_path: str,
    root: etree._Element,
) -> list[dict[str, object]]:
    """把 standalone SVG spine item 尽力转换为文本和包内栅格图片。"""
    empty_registry = EpubAnchorRegistry([], package)
    context = _EpubMarkupContext(package, chapter_path, empty_registry)
    stylesheet = _load_chapter_stylesheet(package, chapter_path, root)
    return MarkupProjector(root, context, stylesheet).convert_svg()


__all__ = [
    "EpubAnchorRegistry",
    "EpubChapterConverter",
    "build_anchor_registry",
    "convert_svg_spine",
]
