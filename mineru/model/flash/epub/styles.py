# Copyright (c) Opendatalab. All rights reserved.
"""解析 EPUB XHTML 使用的有限语义 CSS 子集。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lxml import etree  # type: ignore[reportMissingImports]


_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


@dataclass(frozen=True, slots=True)
class TextStyle:
    """保存可投影到 Middle JSON 行内协议的文字样式。"""

    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    superscript: bool = False
    subscript: bool = False

    def merge(self, other: TextStyle) -> TextStyle:
        """合并继承样式和当前元素显式开启的样式。"""
        return TextStyle(
            bold=self.bold or other.bold,
            italic=self.italic or other.italic,
            underline=self.underline or other.underline,
            strikethrough=self.strikethrough or other.strikethrough,
            superscript=self.superscript or other.superscript,
            subscript=self.subscript or other.subscript,
        )

    def names(self) -> tuple[str, ...]:
        """按稳定顺序返回现有行内协议识别的样式名称。"""
        return tuple(
            name
            for enabled, name in (
                (self.bold, "bold"),
                (self.italic, "italic"),
                (self.underline, "underline"),
                (self.strikethrough, "strikethrough"),
                (self.superscript, "superscript"),
                (self.subscript, "subscript"),
            )
            if enabled
        )


@dataclass(frozen=True, slots=True)
class TextStyleDelta:
    """保存 CSS 对各文字样式的显式开启、关闭或未声明状态。"""

    bold: bool | None = None
    italic: bool | None = None
    underline: bool | None = None
    strikethrough: bool | None = None
    superscript: bool | None = None
    subscript: bool | None = None

    def apply(self, base: TextStyle) -> TextStyle:
        """把当前声明覆盖到已解析的继承/标签样式。"""
        return TextStyle(
            bold=base.bold if self.bold is None else self.bold,
            italic=base.italic if self.italic is None else self.italic,
            underline=base.underline if self.underline is None else self.underline,
            strikethrough=base.strikethrough if self.strikethrough is None else self.strikethrough,
            superscript=base.superscript if self.superscript is None else self.superscript,
            subscript=base.subscript if self.subscript is None else self.subscript,
        )

    def is_empty(self) -> bool:
        """返回当前声明是否没有触及任何受支持样式。"""
        return all(
            value is None
            for value in (
                self.bold,
                self.italic,
                self.underline,
                self.strikethrough,
                self.superscript,
                self.subscript,
            )
        )


@dataclass(frozen=True, slots=True)
class ElementStyle:
    """保存元素最终文字样式和可见性。"""

    text: TextStyle
    hidden: bool = False


@dataclass(frozen=True, slots=True)
class _Rule:
    """保存一个受支持的简单 CSS selector 及其声明。"""

    tag: str | None
    class_name: str | None
    priority: int
    order: int
    style: TextStyleDelta
    hidden: bool | None


def _local_name(element: etree._Element) -> str:
    """返回 XHTML 元素不含命名空间的小写本地名。"""
    return etree.QName(element).localname.casefold()


def _parse_declarations(value: str) -> tuple[TextStyleDelta, bool | None]:
    """从声明串提取字体语义和 display/visibility 隐藏状态。"""
    bold = italic = underline = strikethrough = superscript = subscript = None
    hidden: bool | None = None
    for raw_declaration in value.split(";"):
        if ":" not in raw_declaration:
            continue
        name, raw_value = raw_declaration.split(":", 1)
        name = name.strip().casefold()
        normalized = raw_value.split("!important", 1)[0].strip().casefold()
        if name == "font-weight":
            bold = normalized in {"bold", "bolder"} or normalized.isdigit() and int(normalized) >= 600
        elif name == "font-style":
            italic = normalized in {"italic", "oblique"}
        elif name in {"text-decoration", "text-decoration-line"}:
            if normalized == "none":
                underline = False
                strikethrough = False
            else:
                underline = "underline" in normalized
                strikethrough = "line-through" in normalized
        elif name == "vertical-align":
            superscript = normalized in {"super", "text-top"}
            subscript = normalized in {"sub", "text-bottom"}
        elif name == "display":
            hidden = normalized == "none"
        elif name == "visibility":
            hidden = normalized in {"hidden", "collapse"}
    return TextStyleDelta(bold, italic, underline, strikethrough, superscript, subscript), hidden


class EpubStylesheet:
    """保存按文档顺序解析的简单 tag/class CSS 规则。"""

    def __init__(self) -> None:
        """初始化空规则列表。"""
        self._rules: list[_Rule] = []

    def add(self, css: str) -> None:
        """追加一个 stylesheet 中受支持的简单 selector 规则。"""
        normalized_css = _CSS_COMMENT_RE.sub("", css)
        for chunk in normalized_css.split("}"):
            if "{" not in chunk:
                continue
            selectors, declarations = chunk.split("{", 1)
            style, hidden = _parse_declarations(declarations)
            if style.is_empty() and hidden is None:
                continue
            for selector in selectors.split(","):
                parsed = self._parse_selector(selector)
                if parsed is None:
                    continue
                tag, class_name, priority = parsed
                self._rules.append(_Rule(tag, class_name, priority, len(self._rules), style, hidden))

    @staticmethod
    def _parse_selector(selector: str) -> tuple[str | None, str | None, int] | None:
        """只接受 tag、.class 和 tag.class，拒绝组合器及伪类。"""
        normalized = selector.strip()
        if not normalized or any(token in normalized for token in (" ", ">", "+", "~", ":", "[", "#")):
            return None
        if "." in normalized:
            tag_text, class_name = normalized.split(".", 1)
            if not class_name or "." in class_name:
                return None
            tag = tag_text.casefold() or None
            return tag, class_name, 10 + (1 if tag else 0)
        return normalized.casefold(), None, 1

    def resolve(self, element: etree._Element, inherited: TextStyle) -> ElementStyle:
        """计算元素的继承样式、标签默认样式、CSS 规则和 inline style。"""
        tag = _local_name(element)
        classes = frozenset((element.get("class") or "").split())
        tag_style = TextStyle(
            bold=tag in {"b", "strong"},
            italic=tag in {"cite", "dfn", "em", "i", "var"},
            strikethrough=tag in {"del", "s", "strike"},
            superscript=tag == "sup",
            subscript=tag == "sub",
        )
        style = inherited.merge(tag_style)
        hidden = element.get("hidden") is not None or (element.get("aria-hidden") or "").casefold() == "true"
        matching = sorted(
            (
                rule
                for rule in self._rules
                if (rule.tag is None or rule.tag == tag)
                and (rule.class_name is None or rule.class_name in classes)
            ),
            key=lambda rule: (rule.priority, rule.order),
        )
        for rule in matching:
            style = rule.style.apply(style)
            if rule.hidden is not None:
                hidden = rule.hidden
        inline_style, inline_hidden = _parse_declarations(element.get("style") or "")
        style = inline_style.apply(style)
        if inline_hidden is not None:
            hidden = inline_hidden
        return ElementStyle(style, hidden)


__all__ = ["ElementStyle", "EpubStylesheet", "TextStyle", "TextStyleDelta"]
