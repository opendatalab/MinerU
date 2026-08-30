# Copyright (c) Opendatalab. All rights reserved.
"""Markdown renderer 的 HTML 表格判型与无损 GFM 转换。"""

from __future__ import annotations

import html
import re

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from ....config import LatexDelimitersConfig
from .assets import prefix_html_image_sources
from .inline import markdown_styles_require_html, render_styled_markdown_text

_INLINE_EQ_RE = re.compile(r"<eq>(?P<latex>.*?)</eq>", re.IGNORECASE | re.DOTALL)
_GFM_FORMULA_PIPE_RE = re.compile(r"(?P<slashes>\\*)\|")
_COMPLEX_CELL_TAGS = {
    "blockquote",
    "div",
    "dl",
    "figure",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "img",
    "li",
    "math",
    "object",
    "ol",
    "pre",
    "svg",
    "table",
    "ul",
}
_ALLOWED_INLINE_TAGS = {
    "a",
    "b",
    "br",
    "code",
    "em",
    "eq",
    "i",
    "p",
    "s",
    "span",
    "strong",
    "sub",
    "sup",
    "u",
}
_STYLE_TAGS = {
    "b": "bold",
    "strong": "bold",
    "em": "italic",
    "i": "italic",
    "u": "underline",
    "s": "strikethrough",
    "sup": "superscript",
    "sub": "subscript",
}


def _strip_embedded_images(markup: str) -> str:
    """移除自定义图片 renderer 接管的 HTML 图片，并识别清理后的空内容。"""
    soup = BeautifulSoup(markup, "html.parser")
    images = soup.find_all("img")
    if not images:
        return markup
    for image in images:
        image.decompose()
    if not soup.get_text(strip=True):
        return ""
    return str(soup)


def render_html_table(
    content: str,
    *,
    asset_base_url: str,
    delimiters: LatexDelimitersConfig,
) -> str | None:
    """将 HTML table 按复杂度输出原 HTML 或转换为 GFM 表格。"""
    prefixed = prefix_html_image_sources(content, asset_base_url)
    soup = BeautifulSoup(prefixed, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return None
    table = tables[0]
    if len(tables) != 1 or _is_complex_table(table):
        return format_embedded_html(prefixed, asset_base_url="", delimiters=delimiters).strip()
    markdown = _convert_simple_table(table, delimiters)
    if markdown is not None:
        return markdown
    return format_embedded_html(prefixed, asset_base_url="", delimiters=delimiters).strip()


def format_embedded_html(
    markup: str,
    *,
    asset_base_url: str,
    delimiters: LatexDelimitersConfig,
) -> str:
    """统一处理嵌入 HTML 的图片地址与行内公式标签。"""
    prefixed = prefix_html_image_sources(markup, asset_base_url)

    def _replace_inline_equation(match: re.Match[str]) -> str:
        """把单个 HTML eq 标签替换为配置的行内公式定界符。"""
        return f" {delimiters.inline.left}{html.unescape(match.group('latex')).strip()}{delimiters.inline.right} "

    return _INLINE_EQ_RE.sub(_replace_inline_equation, prefixed)


def _is_complex_table(table: Tag) -> bool:
    """判断表格是否包含 GFM 无法无损表达的结构。"""
    if table.find("table") is not None:
        return True
    thead = table.find("thead")
    if thead is not None and len(thead.find_all("tr", recursive=False)) > 1:
        return True
    for cell in table.find_all(("th", "td")):
        if cell.has_attr("rowspan") or cell.has_attr("colspan"):
            return True
        if len(cell.find_all("p")) > 1:
            return True
        for descendant in cell.descendants:
            if not isinstance(descendant, Tag):
                continue
            if descendant.name in _COMPLEX_CELL_TAGS:
                return True
            if descendant.name not in _ALLOWED_INLINE_TAGS:
                return True
            if descendant.name == "span" and descendant.attrs:
                return True
    return False


def _convert_simple_table(table: Tag, delimiters: LatexDelimitersConfig) -> str | None:
    """把已确认简单的单层 HTML table 转换为 GFM。"""
    rows = table.find_all("tr")
    if not rows:
        return None
    rendered_rows: list[list[str]] = []
    header_flags: list[list[bool]] = []
    for row in rows:
        cells = row.find_all(("th", "td"), recursive=False)
        if not cells:
            return None
        rendered_rows.append([_normalize_cell_text(_render_inline_children(cell, delimiters)) for cell in cells])
        header_flags.append([cell.name == "th" for cell in cells])

    width = max(len(row) for row in rendered_rows)
    for row in rendered_rows:
        row.extend([""] * (width - len(row)))
    header_index = _detect_header_index(table, header_flags)
    header = rendered_rows[header_index]
    body = rendered_rows[:header_index] + rendered_rows[header_index + 1 :]
    return "\n".join(
        [
            _format_markdown_row(header),
            _format_markdown_row(["---"] * width),
            *[_format_markdown_row(row) for row in body],
        ]
    )


def _detect_header_index(table: Tag, header_flags: list[list[bool]]) -> int:
    """优先选择 thead 或全 th 行，否则使用首行作为 GFM 表头。"""
    thead = table.find("thead")
    if thead is not None and thead.find("tr", recursive=False) is not None:
        return 0
    for index, flags in enumerate(header_flags):
        if flags and all(flags):
            return index
    return 0


def _render_inline_children(
    node: Tag,
    delimiters: LatexDelimitersConfig,
    inherited_styles: tuple[str, ...] = (),
) -> str:
    """递归渲染简单单元格中的安全行内 HTML。"""
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            escaped = _escape_cell_text(str(child))
            parts.append(render_styled_markdown_text(escaped, inherited_styles))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        style = _STYLE_TAGS.get(name)
        styles = tuple(dict.fromkeys((*inherited_styles, style))) if style is not None else inherited_styles
        rendered = _render_inline_children(child, delimiters, styles)
        if name in {"p", "span"}:
            parts.append(rendered)
        elif name == "br":
            parts.append("<br>")
        elif name == "code":
            parts.append(_render_inline_code(child.get_text()))
        elif name == "eq":
            latex = html.unescape(child.get_text()).strip()
            escaped_latex = _escape_gfm_formula_pipes(latex)
            parts.append(f"{delimiters.inline.left}{escaped_latex}{delimiters.inline.right}" if escaped_latex else "")
        elif name == "a":
            href = str(child.get("href", "")).strip()
            if not href:
                parts.append(rendered)
            elif _node_has_complex_text_style(child, inherited_styles):
                parts.append(f'<a href="{html.escape(href, quote=True)}">{rendered}</a>')
            else:
                parts.append(f"[{rendered}]({_escape_link_url(href)})")
        elif name in _STYLE_TAGS:
            parts.append(rendered)
        else:
            parts.append(rendered)
    return "".join(parts)


def _node_has_complex_text_style(node: Tag, inherited_styles: tuple[str, ...]) -> bool:
    """判断节点后代是否含必须用 HTML 表达的有效文字样式组合。"""
    for child in node.children:
        if isinstance(child, NavigableString):
            if str(child) and markdown_styles_require_html(inherited_styles):
                return True
            continue
        if not isinstance(child, Tag):
            continue
        style = _STYLE_TAGS.get(child.name)
        styles = tuple(dict.fromkeys((*inherited_styles, style))) if style is not None else inherited_styles
        if _node_has_complex_text_style(child, styles):
            return True
    return False


def _render_inline_code(content: str) -> str:
    """使用足够长的反引号包装表格单元格内代码。"""
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", content)), default=0)
    fence = "`" * max(1, longest + 1)
    return f"{fence}{_escape_cell_text(content)}{fence}"


def _escape_gfm_formula_pipes(latex: str) -> str:
    """转义公式竖线，并保证 GFM 解析后恢复原始反斜杠数量。"""

    def _replace(match: re.Match[str]) -> str:
        """把竖线前 n 个反斜杠扩展为 2n+1 个 Markdown 反斜杠。"""
        slash_count = len(match.group("slashes"))
        return "\\" * (2 * slash_count + 1) + "|"

    return _GFM_FORMULA_PIPE_RE.sub(_replace, latex)


def _escape_link_url(url: str) -> str:
    """转义 GFM 表格链接目标中的空格、反斜杠和括号。"""
    return url.replace("\\", "%5C").replace(" ", "%20").replace("(", "%28").replace(")", "%29").replace("|", "%7C")


def _escape_cell_text(content: str) -> str:
    """转义 GFM 单元格中的 HTML、反斜杠与管道符，避免源文本注入活动标签。"""
    escaped_html = content.replace("&", "&amp;").replace("<", "&lt;")
    return escaped_html.replace("\\", "\\\\").replace("|", r"\|")


def _normalize_cell_text(content: str) -> str:
    """压缩普通空白，同时保留显式 br 换行。"""
    content = re.sub(r"[ \t\r\f\v]+", " ", content)
    content = re.sub(r" *\n+ *", " ", content)
    content = re.sub(r" *<br> *", "<br>", content)
    return content.strip()


def _format_markdown_row(row: list[str]) -> str:
    """把一行单元格格式化为 GFM 行。"""
    return f"| {' | '.join(row)} |"


__all__ = ["format_embedded_html", "render_html_table"]
