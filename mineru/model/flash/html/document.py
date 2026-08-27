# Copyright (c) Opendatalab. All rights reserved.
"""安全加载、规范化并描述一个 standalone HTML 文档。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Literal

from lxml import etree, html as lxml_html  # type: ignore[reportMissingImports]

from .._shared.markup.formula import FormulaExtraction, extract_formula, is_tex_script
from .._shared.markup.projector import local_name
from .constants import MAX_HTML_BYTES, MAX_HTML_DEPTH, MAX_HTML_NODES
from .contracts import HtmlSourceContext
from .errors import HtmlParseError, HtmlResourceLimitError


_ACTIVE_TAGS = frozenset(
    {
        "applet",
        "audio",
        "button",
        "canvas",
        "embed",
        "form",
        "iframe",
        "input",
        "object",
        "script",
        "select",
        "style",
        "template",
        "textarea",
        "video",
    }
)
_FORMULA_GENERATOR_CLASS_TOKENS = frozenset({"katex", "mathjax", "mineru-math"})
_GENERIC_FORMULA_CLASS_TOKENS = frozenset({"formula", "math", "tex"})
_MEANINGFUL_FORMULA_SIBLING_TAGS = frozenset(
    {"audio", "br", "canvas", "figure", "hr", "iframe", "image", "img", "object", "svg", "table", "video"}
)
_ASCIIMATH_SCRIPT_TYPE_RE = re.compile(r"^math/asciimath(?:\s*;.*)?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class HtmlStylesheetSource:
    """保存一个按 head 源顺序出现的内联或外链 stylesheet。"""

    kind: Literal["inline", "link"]
    value: str


@dataclass(frozen=True, slots=True)
class HtmlDocument:
    """保存已规范化 DOM、标题、样式引用与来源上下文。"""

    root: etree._Element
    body: etree._Element
    stylesheets: tuple[HtmlStylesheetSource, ...]
    base_href: str | None
    title: str | None
    open_graph_title: str | None
    site_name: str | None
    source_context: HtmlSourceContext


def parse_html_document(file_bytes: bytes, source_context: HtmlSourceContext | None = None) -> HtmlDocument:
    """从受限字节输入构造不执行脚本且资源引用尚未加载的 HTML DOM。"""
    if len(file_bytes) > MAX_HTML_BYTES:
        raise HtmlResourceLimitError(f"HTML resource limit exceeded: max_html_bytes={MAX_HTML_BYTES}")
    context = source_context or HtmlSourceContext()
    if not file_bytes.strip():
        root = etree.Element("html")
        body = etree.SubElement(root, "body")
        return HtmlDocument(root, body, (), None, None, None, None, context)

    parser = lxml_html.HTMLParser(
        recover=True,
        no_network=True,
        remove_comments=False,
        huge_tree=False,
    )
    try:
        root = lxml_html.document_fromstring(file_bytes, parser=parser)
    except (etree.ParserError, etree.XMLSyntaxError, UnicodeError, ValueError) as exc:
        raise HtmlParseError(f"Malformed HTML document: {exc}") from exc
    _validate_dom_shape(root)

    stylesheets: list[HtmlStylesheetSource] = []
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        name = local_name(element)
        if name == "style":
            stylesheets.append(HtmlStylesheetSource("inline", "".join(element.itertext())))
        elif name == "link" and "stylesheet" in (element.get("rel") or "").casefold().split():
            if href := (element.get("href") or "").strip():
                stylesheets.append(HtmlStylesheetSource("link", href))
    base_href = next(
        (
            value
            for element in root.iter()
            if isinstance(element.tag, str) and local_name(element) == "base" and (value := (element.get("href") or "").strip())
        ),
        None,
    )
    title = next(
        (
            _collapsed_text(element)
            for element in root.iter()
            if isinstance(element.tag, str) and local_name(element) == "title" and _collapsed_text(element)
        ),
        None,
    )
    open_graph_title = _meta_content(root, property_name="og:title")
    site_name = _meta_content(root, property_name="og:site_name")

    _normalize_formula_sources(root)
    _remove_active_content(root)
    body = next(
        (element for element in root.iter() if isinstance(element.tag, str) and local_name(element) == "body"),
        None,
    )
    if body is None:
        body = etree.Element("body")
        body.append(deepcopy(root))
        root = etree.Element("html")
        root.append(body)
    return HtmlDocument(
        root=root,
        body=body,
        stylesheets=tuple(stylesheets),
        base_href=base_href,
        title=title,
        open_graph_title=open_graph_title,
        site_name=site_name,
        source_context=context,
    )


def _validate_dom_shape(root: etree._Element) -> None:
    """迭代校验 DOM 节点数和最大深度，避免深层递归继续传播。"""
    node_count = 0
    stack: list[tuple[etree._Element, int]] = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        node_count += 1
        if node_count > MAX_HTML_NODES:
            raise HtmlResourceLimitError(f"HTML resource limit exceeded: max_html_nodes={MAX_HTML_NODES}")
        if not isinstance(node.tag, str):
            continue
        if depth > MAX_HTML_DEPTH:
            raise HtmlResourceLimitError(f"HTML resource limit exceeded: max_html_depth={MAX_HTML_DEPTH}")
        stack.extend((child, depth + 1) for child in node)


def _meta_content(root: etree._Element, *, property_name: str) -> str | None:
    """返回首个匹配 property/name 的非空 meta content。"""
    target = property_name.casefold()
    for element in root.iter():
        if not isinstance(element.tag, str) or local_name(element) != "meta":
            continue
        name = (element.get("property") or element.get("name") or "").strip().casefold()
        content = (element.get("content") or "").strip()
        if name == target and content:
            return content
    return None


def _collapsed_text(element: etree._Element) -> str:
    """折叠元素纯文本中的 HTML 排版空白。"""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def _normalize_formula_sources(root: etree._Element) -> None:
    """按共享优先级把成功来源收敛为携带裸 LaTeX 的静态 math 元素。"""
    _preserve_asciimath_text(root)
    for element in list(root.iter()):
        if not isinstance(element.tag, str) or not _is_attached(root, element):
            continue
        classes = frozenset((element.get("class") or "").casefold().split())
        is_candidate = _is_formula_carrier(element) or (
            bool(classes & _GENERIC_FORMULA_CLASS_TOKENS) and _formula_wrapper_contains_only_carrier(element)
        )
        if not is_candidate:
            continue
        if formula := extract_formula(element):
            _replace_with_formula(element, formula)


def _is_formula_carrier(element: etree._Element) -> bool:
    """判断元素自身是否携带公式来源，而不是仅从任意后代继承。"""
    if local_name(element) == "math" or is_tex_script(element):
        return True
    if any((element.get(attribute) or "").strip() for attribute in ("data-mineru-latex", "data-tex", "data-expr")):
        return True
    classes = frozenset((element.get("class") or "").casefold().split())
    return bool(classes & _FORMULA_GENERATOR_CLASS_TOKENS)


def _formula_wrapper_contains_only_carrier(element: etree._Element) -> bool:
    """仅允许恰好一个 carrier 且其外没有可见文本或媒体的通用 wrapper 整体折叠。"""
    carriers: list[etree._Element] = []
    for candidate in element.iterdescendants():
        if not isinstance(candidate.tag, str) or not _is_formula_carrier(candidate):
            continue
        if any(ancestor in carriers for ancestor in candidate.iterancestors()):
            continue
        carriers.append(candidate)
    if len(carriers) != 1:
        return False
    carrier = carriers[0]

    def inside_carrier(candidate: etree._Element | None) -> bool:
        """判断节点正文是否位于唯一 carrier 子树内。"""
        return candidate is not None and (
            candidate is carrier or any(ancestor is carrier for ancestor in candidate.iterancestors())
        )

    outside_text = [element.text or ""]
    for candidate in element.iterdescendants():
        if not isinstance(candidate.tag, str):
            if not inside_carrier(candidate.getparent()):
                outside_text.append(candidate.tail or "")
            continue
        if not inside_carrier(candidate):
            outside_text.append(candidate.text or "")
            if local_name(candidate) in _MEANINGFUL_FORMULA_SIBLING_TAGS:
                return False
        if not inside_carrier(candidate.getparent()):
            outside_text.append(candidate.tail or "")
    return not any(value.strip() for value in outside_text)


def _preserve_asciimath_text(root: etree._Element) -> None:
    """把暂不支持的 AsciiMath script 转为可见静态文本，避免活动内容清理时丢失。"""
    for element in list(root.iter()):
        if not isinstance(element.tag, str) or local_name(element) != "script":
            continue
        script_type = (element.get("type") or "").strip()
        if _ASCIIMATH_SCRIPT_TYPE_RE.fullmatch(script_type) is None:
            continue
        value = "".join(element.itertext()).strip()
        if not value:
            continue
        parent = element.getparent()
        if parent is None:
            continue
        replacement = etree.Element("span")
        replacement.set("class", "mineru-formula-fallback")
        replacement.text = value
        replacement.tail = element.tail
        parent.replace(element, replacement)


def _replace_with_formula(element: etree._Element, formula: FormulaExtraction) -> None:
    """用携带规范 LaTeX 的安全 math 占位替换一个网页公式节点。"""
    parent = element.getparent()
    if parent is None:
        return
    replacement = etree.Element("math")
    replacement.set("data-mineru-latex", formula.latex)
    replacement.set("data-formula-display", formula.display)
    if formula.display == "block":
        replacement.set("display", "block")
    if (element.get("data-block-type") or "").strip() == "equation":
        replacement.set("data-block-type", "equation")
    replacement.tail = element.tail
    parent.replace(element, replacement)


def _is_attached(root: etree._Element, element: etree._Element) -> bool:
    """判断预扫描元素是否仍属于当前 DOM，跳过已被外层公式替换的旧后代。"""
    return element is root or any(ancestor is root for ancestor in element.iterancestors())


def _remove_active_content(root: etree._Element) -> None:
    """删除活动内容并把 noscript 静态回退转换为普通容器。"""
    for element in list(root.iter()):
        if isinstance(element, etree._Comment):
            _drop_tree_preserve_tail(element)
            continue
        if not isinstance(element.tag, str):
            continue
        name = local_name(element)
        if name == "noscript":
            element.tag = "div"
            continue
        if name in _ACTIVE_TAGS:
            _drop_tree_preserve_tail(element)


def _drop_tree_preserve_tail(element: etree._Element) -> None:
    """删除节点整棵子树，同时把 tail 归还给相邻文本位置。"""
    parent = element.getparent()
    if parent is None:
        return
    tail = element.tail or ""
    previous = element.getprevious()
    if tail:
        if previous is not None:
            previous.tail = (previous.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
    parent.remove(element)


__all__ = ["HtmlDocument", "HtmlStylesheetSource", "parse_html_document"]
