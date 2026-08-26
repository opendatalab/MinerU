# Copyright (c) Opendatalab. All rights reserved.
"""把 EPUB XHTML/SVG 内容文档转换为 MinerU raw blocks。"""

from __future__ import annotations

import base64
import hashlib
import html
import re
from dataclasses import dataclass

from lxml import etree  # type: ignore[reportMissingImports]

from ....types import BlockType
from ....utils.image_payload import parse_image_data_uri_strict
from .._shared.hyperlink import render_inline_hyperlink, sanitize_hyperlink_target
from .._shared.mathml import mathml_to_latex
from .constants import IMAGE_MEDIA_BY_EXTENSION, SVG_MEDIA_TYPE
from .package import EpubPackage
from .styles import EpubStylesheet, TextStyle


_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "dd",
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
        "svg",
        "table",
        "ul",
    }
)
_SKIPPED_TAGS = frozenset(
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
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
MAX_EPUB_TABLE_SPAN = 1_000


def _local_name(element: etree._Element) -> str:
    """返回 XHTML/SVG 元素不含命名空间的小写本地名。"""
    return etree.QName(element).localname.casefold()


def _clean_text_node(value: str | None) -> str:
    """把 XML 排版空白折叠为单个普通空格，同时保留边界空格。"""
    if not value:
        return ""
    return _WHITESPACE_RE.sub(" ", value)


def _bounded_table_span(value: str) -> str | None:
    """规范化 EPUB 表格跨度，并在整数转换前拒绝超长或越界值。"""
    if not value.isdigit():
        return None
    normalized = value.lstrip("0")
    if not normalized or len(normalized) > len(str(MAX_EPUB_TABLE_SPAN)):
        return None
    span = int(normalized)
    return str(span) if span <= MAX_EPUB_TABLE_SPAN else None


def _visible_text(element: etree._Element) -> str:
    """提取元素折叠空白后的可见纯文本。"""
    return _WHITESPACE_RE.sub(" ", html.unescape("".join(element.itertext()))).strip()


def _normalized_toc_label(value: str) -> str:
    """规范目录标签的实体、空白和大小写，供严格标题匹配使用。"""
    return _WHITESPACE_RE.sub(" ", html.unescape(value)).strip().casefold()


def _entity_text(element: etree._Element) -> str:
    """把未解析的安全 HTML 命名实体转换为可见文本。"""
    name = getattr(element, "name", "")
    return html.unescape(f"&{name};") if name else ""


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


def _load_chapter_stylesheet(package: EpubPackage, chapter_path: str, root: etree._Element) -> EpubStylesheet:
    """按章节 head 顺序加载包内 CSS 与内联 style。"""
    stylesheet = EpubStylesheet()
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


def _element_is_hidden(element: etree._Element, stylesheet: EpubStylesheet) -> bool:
    """按 converter 的祖先到自身样式解析顺序判断元素是否不会输出。"""
    inherited = TextStyle()
    chain = [ancestor for ancestor in reversed(list(element.iterancestors())) if isinstance(ancestor.tag, str)]
    body_index = next((index for index, ancestor in enumerate(chain) if _local_name(ancestor) == "body"), None)
    if body_index is not None:
        chain = chain[body_index:]
    chain.append(element)
    for current in chain:
        resolved = stylesheet.resolve(current, inherited)
        if resolved.hidden:
            return True
        inherited = resolved.text
    return False


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
            headings = [
                element
                for element in chapter.root.iter()
                if isinstance(element.tag, str)
                and _local_name(element) in {"h1", "h2", "h3", "h4", "h5", "h6"}
                and _visible_text(element)
                and not _element_is_hidden(element, stylesheet)
            ]
            for ordinal, heading in enumerate(headings):
                source_id = _element_id(heading)
                identity = f"{source_id or 'heading'}-{ordinal}"
                anchor = _canonical_anchor(chapter.path, identity)
                self._heading_anchors[heading] = anchor
                self._heading_labels[anchor] = _visible_text(heading)
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
                first_anchor = self._heading_anchors[headings[0]]
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


class EpubChapterConverter:
    """把一个 XHTML spine item 投影为现有 raw model-list。"""

    def __init__(
        self,
        package: EpubPackage,
        chapter_path: str,
        root: etree._Element,
        anchors: EpubAnchorRegistry,
    ) -> None:
        """绑定单个章节的包、路径、DOM 和文档级锚点注册表。"""
        self.package = package
        self.chapter_path = chapter_path
        self.root = root
        self.anchors = anchors
        self.stylesheet = self._load_stylesheet()

    def _load_stylesheet(self) -> EpubStylesheet:
        """按 head 文档顺序加载包内 CSS 和内联 style 元素。"""
        return _load_chapter_stylesheet(self.package, self.chapter_path, self.root)

    def convert(self) -> list[dict[str, object]]:
        """解析 XHTML body 并返回按 DOM 顺序排列的 raw blocks。"""
        body = next(
            (element for element in self.root.iter() if isinstance(element.tag, str) and _local_name(element) == "body"),
            None,
        )
        if body is None:
            return []
        body_style = self.stylesheet.resolve(body, TextStyle())
        if body_style.hidden:
            return []
        return self._parse_container_contents(body, body_style.text)

    def _parse_container_contents(self, element: etree._Element, style: TextStyle) -> list[dict[str, object]]:
        """把容器的行内碎片和块级子元素按源顺序拆成 raw blocks。"""
        blocks: list[dict[str, object]] = []
        inline_parts: list[str] = [self._render_text(element.text, style)]

        def flush_inline() -> None:
            """把当前连续行内片段写为一个普通正文 block。"""
            content = "".join(inline_parts).strip()
            inline_parts.clear()
            if content:
                blocks.append({"type": BlockType.TEXT, "content": content})

        for child in element:
            if not isinstance(child.tag, str):
                inline_parts.append(self._render_text(_entity_text(child), style))
                inline_parts.append(self._render_text(child.tail, style))
                continue
            name = _local_name(child)
            if name in _BLOCK_TAGS:
                flush_inline()
                blocks.extend(self._parse_block(child, style))
            else:
                rendered, extras = self._render_inline_element(child, style)
                inline_parts.append(rendered)
                if extras:
                    flush_inline()
                    blocks.extend(extras)
            inline_parts.append(self._render_text(child.tail, style))
        flush_inline()
        return blocks

    def _parse_block(self, element: etree._Element, inherited: TextStyle) -> list[dict[str, object]]:
        """把一个块级 XHTML 元素分派到对应 raw block 转换逻辑。"""
        resolved = self.stylesheet.resolve(element, inherited)
        if resolved.hidden:
            return []
        name = _local_name(element)
        if name in _SKIPPED_TAGS or name == "hr":
            return []
        if self.anchors.note_anchor(element) is not None:
            return self._parse_note_element(element, resolved.text)
        if name in {"h1", "h2", "h3", "h4", "h5", "h6", "p"}:
            content, extras = self._render_inline_children(element, resolved.text)
            blocks: list[dict[str, object]] = []
            if content.strip():
                if name == "h1":
                    block: dict[str, object] = {"type": BlockType.DOC_TITLE, "level": 1, "content": content.strip()}
                elif name.startswith("h"):
                    block = {
                        "type": BlockType.PARAGRAPH_TITLE,
                        "level": min(max(int(name[1:]), 2), 6),
                        "is_numbered_style": False,
                        "content": content.strip(),
                    }
                else:
                    block = {"type": BlockType.TEXT, "content": content.strip()}
                if name.startswith("h") and (anchor := self.anchors.heading_anchor(element)):
                    block["anchor"] = anchor
                blocks.append(block)
            return [*blocks, *extras]
        if name in {"ul", "ol"}:
            list_block, extras = self._parse_list(element, resolved.text)
            return ([list_block] if list_block is not None else []) + extras
        if name == "table":
            return self._parse_table(element, resolved.text)
        if name == "pre":
            content = "".join(element.itertext())
            return [{"type": BlockType.CODE, "content": content}] if content.strip() else []
        if name == "math":
            latex = mathml_to_latex(element)
            return [{"type": BlockType.EQUATION, "content": latex}] if latex else []
        if name == "figure":
            return self._parse_figure(element, resolved.text)
        if name == "svg":
            return self._parse_svg(element, resolved.text)
        return self._parse_container_contents(element, resolved.text)

    def _parse_note_element(self, element: etree._Element, style: TextStyle) -> list[dict[str, object]]:
        """逐块转换单条 Footnote/Endnote，并只给首个文本脚注挂载 anchor。"""
        blocks = self._parse_container_contents(element, style)
        anchor = self.anchors.note_anchor(element)
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
    ) -> tuple[str, list[dict[str, object]]]:
        """渲染元素的连续行内内容，并旁路其中的视觉 blocks。"""
        parts = [self._render_text(element.text, style)]
        extras: list[dict[str, object]] = []
        for child in element:
            if not isinstance(child.tag, str):
                parts.append(self._render_text(_entity_text(child), style))
                parts.append(self._render_text(child.tail, style))
                continue
            rendered, child_extras = self._render_inline_element(child, style)
            parts.append(rendered)
            extras.extend(child_extras)
            parts.append(self._render_text(child.tail, style))
        return "".join(parts), extras

    def _render_inline_element(
        self,
        element: etree._Element,
        inherited: TextStyle,
    ) -> tuple[str, list[dict[str, object]]]:
        """把一个 XHTML 行内元素转换为内部富文本协议和可选视觉块。"""
        resolved = self.stylesheet.resolve(element, inherited)
        if resolved.hidden:
            return "", []
        name = _local_name(element)
        if name in _SKIPPED_TAGS:
            return "", []
        if name == "br":
            return "\n", []
        if name in {"img", "image"}:
            return "", self._image_blocks(element)
        if name == "math":
            latex = mathml_to_latex(element)
            return (f"<eq>{html.escape(latex, quote=False)}</eq>" if latex else ""), []
        content, extras = self._render_inline_children(element, resolved.text)
        if name == "a":
            href = element.get("href") or element.get(_XLINK_HREF) or ""
            target = self.anchors.resolve_link(href, base_part=self.chapter_path)
            if target and content.strip():
                return render_inline_hyperlink(content, target), extras
        return content, extras

    @staticmethod
    def _render_text(value: str | None, style: TextStyle) -> str:
        """折叠并转义文本节点，再按现有行内协议应用文字样式。"""
        text = _clean_text_node(value)
        if not text:
            return ""
        escaped = html.escape(text, quote=False)
        names = style.names()
        return f'<text style="{",".join(names)}">{escaped}</text>' if names else escaped

    def _image_blocks(self, element: etree._Element, *, caption: str | None = None) -> list[dict[str, object]]:
        """把包内栅格图片转换为 image block，并用 caption/alt 补说明。"""
        src = element.get("src") or element.get("href") or element.get(_XLINK_HREF) or ""
        data_uri = self._image_data_uri(src)
        alt = (caption or element.get("alt") or element.get("title") or "").strip()
        if data_uri is None:
            return [{"type": BlockType.TEXT, "content": html.escape(alt, quote=False)}] if alt else []
        blocks: list[dict[str, object]] = [{"type": BlockType.IMAGE, "content": "", "image_base64": data_uri}]
        if alt:
            blocks.append({"type": BlockType.IMAGE_CAPTION, "content": html.escape(alt, quote=False)})
        return blocks

    def _image_data_uri(self, src: str) -> str | None:
        """读取、编码并严格校验一个包内栅格图片引用。"""
        target = self.package.resolve_reference(src, base_part=self.chapter_path)
        if target is None:
            return None
        media_type = (self.package.content_type_for(target.path) or "").casefold()
        extension = target.path.rsplit(".", 1)[-1].casefold() if "." in target.path else ""
        media_type = media_type or IMAGE_MEDIA_BY_EXTENSION.get(extension, "")
        if not media_type.startswith("image/") or media_type == SVG_MEDIA_TYPE:
            return None
        payload = self.package.read_part(target.path, asset=True)
        if payload is None:
            return None
        data_uri = f"data:{media_type};base64,{base64.b64encode(payload).decode('ascii')}"
        try:
            parse_image_data_uri_strict(data_uri)
        except ValueError:
            return None
        return data_uri

    def _parse_figure(self, element: etree._Element, style: TextStyle) -> list[dict[str, object]]:
        """解析 figure 中的图片和 figcaption，其余内容按普通容器处理。"""
        caption_element = next(
            (child for child in element if isinstance(child.tag, str) and _local_name(child) == "figcaption"),
            None,
        )
        caption = _visible_text(caption_element) if caption_element is not None else ""
        blocks: list[dict[str, object]] = []
        for child in element:
            if not isinstance(child.tag, str):
                continue
            name = _local_name(child)
            if name in {"img", "image"}:
                child_style = self.stylesheet.resolve(child, style)
                if not child_style.hidden:
                    blocks.extend(self._image_blocks(child, caption=caption))
            elif name != "figcaption":
                child_blocks = (
                    self._parse_block(child, style) if name in _BLOCK_TAGS else self._render_inline_element(child, style)[1]
                )
                blocks.extend(child_blocks)
        if not blocks and caption:
            blocks.append({"type": BlockType.TEXT, "content": html.escape(caption, quote=False)})
        return blocks

    def _visible_svg_text(self, element: etree._Element, style: TextStyle) -> str:
        """递归提取 SVG 可见文本，并排除隐藏后代的内容。"""
        parts = [_clean_text_node(element.text)]
        for child in element:
            if not isinstance(child.tag, str):
                parts.append(_clean_text_node(_entity_text(child)))
                parts.append(_clean_text_node(child.tail))
                continue
            resolved = self.stylesheet.resolve(child, style)
            if not resolved.hidden:
                parts.append(self._visible_svg_text(child, resolved.text))
            parts.append(_clean_text_node(child.tail))
        return _WHITESPACE_RE.sub(" ", html.unescape("".join(parts))).strip()

    def _parse_svg(self, element: etree._Element, style: TextStyle) -> list[dict[str, object]]:
        """从 SVG 尽力提取 title/desc/text 和包内栅格 image。"""
        blocks: list[dict[str, object]] = []
        texts: list[str] = []

        def visit(parent: etree._Element, inherited: TextStyle) -> None:
            """按 SVG 树顺序访问可见候选节点，并让祖先隐藏状态截断子树。"""
            for child in parent:
                if not isinstance(child.tag, str):
                    continue
                resolved = self.stylesheet.resolve(child, inherited)
                if resolved.hidden:
                    continue
                name = _local_name(child)
                if name in {"title", "desc", "text"}:
                    value = self._visible_svg_text(child, resolved.text)
                    if value and value not in texts:
                        texts.append(value)
                elif name == "image":
                    blocks.extend(self._image_blocks(child))
                else:
                    visit(child, resolved.text)

        visit(element, style)
        if texts:
            blocks.insert(0, {"type": BlockType.TEXT, "content": html.escape("\n".join(texts), quote=False)})
        return blocks

    def _parse_table(self, table: etree._Element, style: TextStyle) -> list[dict[str, object]]:
        """重建白名单化 HTML 表格，并把 caption 投影为表格说明。"""
        markup = self._serialize_table_node(table, style)
        if not markup:
            return []
        blocks: list[dict[str, object]] = [{"type": BlockType.TABLE, "content": markup}]
        caption_element = next(
            (child for child in table if isinstance(child.tag, str) and _local_name(child) == "caption"),
            None,
        )
        if caption_element is not None and (caption := _visible_text(caption_element)):
            blocks.append({"type": BlockType.TABLE_CAPTION, "content": html.escape(caption, quote=False)})
        return blocks

    def _serialize_table_node(
        self,
        element: etree._Element,
        inherited: TextStyle,
        *,
        row_link_target: str | None = None,
    ) -> str:
        """递归序列化安全表格结构、行内样式、链接、公式和包内图片。"""
        resolved = self.stylesheet.resolve(element, inherited)
        if resolved.hidden:
            return ""
        name = _local_name(element)
        if name == "caption" or name in _SKIPPED_TAGS:
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
            return self._serialize_table_children(element, resolved.text, row_link_target=row_link_target)
        if name == "br":
            return "<br>"
        if name == "math":
            latex = mathml_to_latex(element)
            return f"<eq>{html.escape(latex, quote=False)}</eq>" if latex else ""
        if name == "img":
            src = element.get("src") or ""
            data_uri = self._image_data_uri(src)
            alt = html.escape(element.get("alt") or "", quote=True)
            return f'<img src="{html.escape(data_uri, quote=True)}" alt="{alt}">' if data_uri else alt
        if name == "tr":
            row_link_target = self._toc_table_row_target(element)
        attributes: list[str] = []
        if name in {"td", "th"}:
            for attribute in ("colspan", "rowspan", "scope"):
                value = (element.get(attribute) or "").strip()
                if attribute == "scope" and value:
                    attributes.append(f'{attribute}="{html.escape(value, quote=True)}"')
                elif span := _bounded_table_span(value):
                    attributes.append(f'{attribute}="{span}"')
        elif name in {"col", "colgroup"}:
            value = (element.get("span") or "").strip()
            if span := _bounded_table_span(value):
                attributes.append(f'span="{span}"')
        if name == "a":
            target = self.anchors.resolve_link(element.get("href") or "", base_part=self.chapter_path)
            if target:
                attributes.append(f'href="{html.escape(target, quote=True)}"')
        inner = self._serialize_table_children(element, resolved.text, row_link_target=row_link_target)
        if name in {"td", "th"} and row_link_target and self._table_cell_can_inherit_toc_link(element):
            inner = f'<a href="{html.escape(row_link_target, quote=True)}">{inner}</a>'
        attrs = f" {' '.join(attributes)}" if attributes else ""
        return f"<{name}{attrs}>{inner}</{name}>"

    def _serialize_table_children(
        self,
        element: etree._Element,
        style: TextStyle,
        *,
        row_link_target: str | None = None,
    ) -> str:
        """序列化表格节点的文本、子元素和 tail。"""
        parts = [self._render_table_text(element.text, style)]
        for child in element:
            if isinstance(child.tag, str):
                parts.append(self._serialize_table_node(child, style, row_link_target=row_link_target))
            else:
                parts.append(self._render_table_text(_entity_text(child), style))
            parts.append(self._render_table_text(child.tail, style))
        return "".join(parts)

    def _toc_table_row_target(self, row: etree._Element) -> str | None:
        """为严格匹配单一目标标题的目录表格行返回内部链接。"""
        links = [
            element
            for element in row.iter()
            if isinstance(element.tag, str) and _local_name(element) == "a" and (element.get("href") or "").strip()
        ]
        if not links:
            return None
        resolved_targets: list[str] = []
        for link in links:
            target = self.anchors.resolve_link(link.get("href") or "", base_part=self.chapter_path)
            if target is None or not target.startswith("#"):
                return None
            resolved_targets.append(target)
        if len(set(resolved_targets)) != 1:
            return None
        target = resolved_targets[0]
        title = self.anchors.heading_label(target[1:])
        if title is None:
            return None
        cells = [child for child in row if isinstance(child.tag, str) and _local_name(child) in {"td", "th"}]
        row_label = " ".join(value for cell in cells if (value := _visible_text(cell)))
        if not row_label or _normalized_toc_label(row_label) != _normalized_toc_label(title):
            return None
        return target

    @staticmethod
    def _table_cell_can_inherit_toc_link(cell: etree._Element) -> bool:
        """只允许纯文本与行内样式单元格继承目录行的唯一内部链接。"""
        if not _visible_text(cell):
            return False
        allowed_inline = {"b", "br", "code", "em", "i", "s", "span", "strong", "sub", "sup", "u"}
        return all(isinstance(child.tag, str) and _local_name(child) in allowed_inline for child in cell.iterdescendants())

    @staticmethod
    def _render_table_text(value: str | None, style: TextStyle) -> str:
        """把表格文字转义后包装为 renderer 支持的安全 HTML 样式标签。"""
        rendered = html.escape(_clean_text_node(value), quote=False)
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
    ) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
        """解析有序/无序列表，并投影为跨 renderer 的连续编号结构。"""
        ordered = _local_name(element) == "ol"
        items = [child for child in element if isinstance(child.tag, str) and _local_name(child) == "li"]
        if not items:
            return None, []
        children: list[dict[str, object]] = []
        extras: list[dict[str, object]] = []
        for item in items:
            item_style = self.stylesheet.resolve(item, style)
            if item_style.hidden:
                continue
            if self.anchors.note_anchor(item) is not None:
                extras.extend(self._parse_note_element(item, item_style.text))
                continue
            content_parts: list[str] = [self._render_text(item.text, item_style.text)]
            nested_lists: list[dict[str, object]] = []
            for child in item:
                if not isinstance(child.tag, str):
                    content_parts.append(self._render_text(_entity_text(child), item_style.text))
                    content_parts.append(self._render_text(child.tail, item_style.text))
                    continue
                name = _local_name(child)
                if name in {"ul", "ol"}:
                    nested, nested_extras = self._parse_list(child, item_style.text)
                    if nested is not None:
                        nested_lists.append(nested)
                    extras.extend(nested_extras)
                elif name in {"table", "figure", "svg"}:
                    extras.extend(self._parse_block(child, item_style.text))
                elif name in _BLOCK_TAGS:
                    if self.anchors.note_anchor(child) is not None:
                        extras.extend(self._parse_note_element(child, item_style.text))
                    else:
                        rendered, child_extras = self._render_inline_children(child, item_style.text)
                        if rendered.strip():
                            content_parts.append(rendered)
                        extras.extend(child_extras)
                else:
                    rendered, child_extras = self._render_inline_element(child, item_style.text)
                    content_parts.append(rendered)
                    extras.extend(child_extras)
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
    def _ordered_list_start(element: etree._Element) -> int:
        """读取有序列表唯一通用起始值，非法或负值统一回退为一。"""
        try:
            start = int(element.get("start") or 1)
        except ValueError:
            return 1
        return start if start >= 0 else 1


def convert_svg_spine(
    package: EpubPackage,
    chapter_path: str,
    root: etree._Element,
) -> list[dict[str, object]]:
    """把 standalone SVG spine item 尽力转换为文本和包内栅格图片。"""
    empty_registry = EpubAnchorRegistry([], package)
    converter = EpubChapterConverter(package, chapter_path, root, empty_registry)
    resolved = converter.stylesheet.resolve(root, TextStyle())
    return [] if resolved.hidden else converter._parse_svg(root, resolved.text)


__all__ = [
    "EpubAnchorRegistry",
    "EpubChapterConverter",
    "build_anchor_registry",
    "convert_svg_spine",
]
