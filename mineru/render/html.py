# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到 MinerU 风格 HTML 的公共渲染实现。"""

from __future__ import annotations

import html
import re
from functools import lru_cache
from importlib import resources
from urllib.parse import quote

from bs4 import BeautifulSoup
from loguru import logger

from mineru.render._internal.common.index import strip_index_page_tail
from mineru.render._internal.common.inline import inline_plain_text, parse_inline_content
from mineru.render._internal.common.list_items import ListItem, parse_list_item_marker, reference_list_needs_bullets
from mineru.render._internal.common.planner import PlannedBlock, build_render_plan
from mineru.render._internal.html.inline import (
    HtmlInlineResult,
    render_inline_content_html,
    render_inline_nodes_html,
    render_joined_inline_contents_html,
    render_math_html,
)
from mineru.render._internal.html.sanitizer import (
    is_supported_html_markup,
    sanitize_html_fragment,
    sanitize_image_source,
)
from mineru.render._internal.html.table import looks_like_gfm_table, render_gfm_table_html
from mineru.render.contracts import RenderMode
from mineru.types import (
    PAGE_AUXILIARY_BLOCK_TYPES,
    RAW_ALGORITHM,
    BlockType,
    ChartAnnotationBlock,
    ChartBlock,
    ChartBodyBlock,
    CodeAnnotationBlock,
    CodeBlock,
    CodeBodyBlock,
    DocTitleBlock,
    EquationBlock,
    ImageAnnotationBlock,
    ImageBlock,
    ImageBodyBlock,
    IndexBlock,
    ListBlock,
    MiddleJson,
    PageAuxTextBlock,
    ParagraphTitleBlock,
    RefTextBlock,
    TableAnnotationBlock,
    TableBlock,
    TableBodyBlock,
    TextBlock,
    TitleBlockBase,
)

_STYLE_RESOURCE_NAME = "mineru.min.css"
_MATHJAX_URL = "https://cdn.jsdelivr.net/npm/mathjax@4.1.2/tex-chtml.js"
_MATHJAX_INTEGRITY = "sha384-zAhQQhdaMeHsMProNntGGg6nOUVcfuF9F22C3d1qJ9NZAVzCplXk1X85D2O5iufn"
_PRISM_CORE_URL = "https://cdn.jsdelivr.net/npm/prismjs@1.30.0/components/prism-core.min.js"
_PRISM_CORE_INTEGRITY = "sha384-zLRFO4dwowZvh8kzutOb5AWhH7f39HeJp+N7PtHF1SQtTBnifRx0AtmvTYs3F4YV"
_PRISM_AUTOLOADER_URL = "https://cdn.jsdelivr.net/npm/prismjs@1.30.0/plugins/autoloader/prism-autoloader.min.js"
_PRISM_AUTOLOADER_INTEGRITY = "sha384-Uq05+JLko69eOiPr39ta9bh7kld5PKZoU+fF7g0EXTAriEollhZ+DrN8Q/Oi8J2Q"
_PRISM_LANGUAGES_PATH = "https://cdn.jsdelivr.net/npm/prismjs@1.30.0/components/"
_MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@11.16.1/dist/mermaid.min.js"
_MERMAID_INTEGRITY = "sha384-aBQXj4hK6Jm05i7aQAsUV3bLdSUrHX1BGYfMB0166TtWt/RRaw+h0Eelme9OCOvy"
_MERMAID_MAX_TEXT_SIZE = 50_000
_MERMAID_FLOWCHART_HEADER_RE = re.compile(
    r"(?:graph|flowchart)\s+(?:TB|TD|BT|RL|LR)\b",
    re.IGNORECASE,
)
_SAFE_PRISM_LANGUAGE_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
_BLOCKED_PRISM_LANGUAGE_NAMES = {"constructor", "prototype"}
_PRISM_LANGUAGE_ALIASES = {
    "c#": "csharp",
    "c++": "cpp",
    "html": "markup",
    "js": "javascript",
    "py": "python",
    "sh": "bash",
    "shell": "bash",
    "none": "",
    "text": "",
    "txt": "",
    "ts": "typescript",
    "xml": "markup",
    "yml": "yaml",
}


class _HtmlRenderer:
    """维护一次 HTML 渲染所需的 anchor、公式与代码资源状态。"""

    def __init__(
        self,
        middle_json: MiddleJson,
        *,
        mode: RenderMode,
        asset_base_url: str,
        standalone: bool,
        document_title: str | None,
    ) -> None:
        """保存严格输入和调用选项，并预收集真实正文标题 anchor。"""
        self.middle_json = middle_json
        self.mode = mode
        self.asset_base_url = asset_base_url
        self.standalone = standalone
        self.document_title = document_title
        self.anchor_targets = _collect_title_anchor_ids(middle_json)
        self.emitted_anchors: set[str] = set()
        self.has_math = False
        self.has_prism = False
        self.has_mermaid = False

    def render(self) -> str:
        """生成 fragment article，并按需包装完整 HTML 文档。"""
        planned_pages = build_render_plan(self.middle_json, self.mode)
        if self.mode is RenderMode.FULL:
            body = self._render_full_pages(planned_pages)
        else:
            body = self._render_default_pages(planned_pages)
        article = f'<article class="mineru-document mineru-document--{self.mode.value}">\n{body}\n</article>'
        if not self.standalone:
            return article
        # 最终依赖以真实 DOM 为准，避免回退分支或普通文本中的 class 字样误触发。
        article_soup = BeautifulSoup(article, "html.parser")
        self.has_math = article_soup.select_one(".mineru-math") is not None
        self.has_prism = any(
            any(str(class_name).startswith("language-") for class_name in code.get("class", []))
            for code in article_soup.find_all("code")
            if code.get_text()
        )
        self.has_mermaid = article_soup.select_one(".mineru-flowchart") is not None
        return self._render_standalone(article)

    def _render_default_pages(self, planned_pages: list[list[PlannedBlock]]) -> str:
        """把 DEFAULT 计划展平为无页面边界的连续阅读内容。"""
        rendered: list[str] = []
        for page in planned_pages:
            for planned in page:
                block = self._render_planned_block(planned)
                if block:
                    rendered.append(block)
        return "\n".join(rendered)

    def _render_full_pages(self, planned_pages: list[list[PlannedBlock]]) -> str:
        """为 FULL 计划保留所有页面 section 和相邻页面分隔。"""
        sections: list[str] = []
        for page in planned_pages:
            page_idx = page[0].page_idx if page else None
            if page_idx is None:
                page_position = len(sections)
                page_idx = self.middle_json.pages[page_position].page_idx
            blocks = [rendered for planned in page if (rendered := self._render_planned_block(planned))]
            rendered_blocks = "\n".join(blocks)
            sections.append(f'<section class="mineru-page" data-page-idx="{page_idx}">\n{rendered_blocks}\n</section>')
        return '\n<hr class="mineru-page-break" aria-hidden="true">\n'.join(sections)

    def _render_planned_block(self, planned: PlannedBlock) -> str:
        """过滤计划块、分派类型 visitor，并添加稳定来源元数据。"""
        if planned.removed:
            return ""
        block = planned.block
        if self.mode is RenderMode.DEFAULT and block.type in PAGE_AUXILIARY_BLOCK_TYPES:
            return ""
        content = self._render_block_content(planned)
        if not content or not content.strip():
            return ""
        attrs = [
            'class="mineru-block"',
            f'data-page-idx="{planned.page_idx}"',
            f'data-block-type="{html.escape(str(block.type), quote=True)}"',
        ]
        if block.index is not None:
            attrs.append(f'data-block-index="{block.index}"')
        return f"<div {' '.join(attrs)}>\n{content}\n</div>"

    def _render_block_content(self, planned: PlannedBlock) -> str:
        """按严格 PageBlock 具体类型返回对应语义 HTML。"""
        block = planned.block
        if isinstance(block, TextBlock):
            rendered = render_joined_inline_contents_html(planned.text_contents or [block.content])
            self._observe_inline(rendered)
            return f'<p class="mineru-text">{rendered.html}</p>' if rendered.html else ""
        if isinstance(block, RefTextBlock):
            rendered = render_joined_inline_contents_html(planned.text_contents or [block.content])
            self._observe_inline(rendered)
            return f'<p class="mineru-ref-text">{rendered.html}</p>' if rendered.html else ""
        if isinstance(block, (DocTitleBlock, ParagraphTitleBlock)):
            return self._render_title(block)
        if isinstance(block, PageAuxTextBlock):
            rendered = render_inline_content_html(block.content)
            self._observe_inline(rendered)
            class_name = html.escape(str(block.type).replace("_", "-"), quote=True)
            return f'<div class="mineru-page-aux mineru-page-aux--{class_name}">{rendered.html}</div>' if rendered.html else ""
        if isinstance(block, EquationBlock):
            return self._render_equation(block)
        if isinstance(block, ListBlock):
            return self._render_list(block)
        if isinstance(block, IndexBlock):
            return self._render_index(block)
        if isinstance(block, ImageBlock):
            return self._render_image_block(block)
        if isinstance(block, TableBlock):
            return self._render_table_block(block)
        if isinstance(block, ChartBlock):
            return self._render_chart_block(block)
        if isinstance(block, CodeBlock):
            return self._render_code_block(block)
        raise TypeError(f"Unsupported PageBlock type: {type(block).__name__}")

    def _render_title(self, block: DocTitleBlock | ParagraphTitleBlock) -> str:
        """渲染标题，并保证正文 anchor 只在首次出现时生成 id。"""
        rendered = render_inline_content_html(block.content)
        self._observe_inline(rendered)
        if not rendered.html:
            return ""
        level = min(max(block.level, 1), 6)
        attrs = [f'class="mineru-heading mineru-heading--{level}"', f'data-heading-level="{block.level}"']
        anchor = _anchor_key(block.anchor)
        if anchor:
            if anchor not in self.emitted_anchors:
                anchor_id = self.anchor_targets.get(anchor)
                if anchor_id:
                    attrs.append(f'id="{html.escape(anchor_id, quote=True)}"')
                self.emitted_anchors.add(anchor)
            else:
                logger.warning("Duplicate HTML title anchor omitted: {!r}", anchor)
        return f"<h{level} {' '.join(attrs)}>{rendered.html}</h{level}>"

    def _render_equation(self, block: EquationBlock) -> str:
        """优先输出行间公式，空 LaTeX 时回退到安全图片来源。"""
        if block.content.strip():
            rendered = render_math_html(block.content, display=True)
            self._observe_inline(rendered)
            return rendered.html
        source = self._safe_block_image_source(block)
        return self._render_image(source, alt="formula", class_name="mineru-equation-image") if source else ""

    def _render_list(self, block: ListBlock) -> str:
        """按直属 marker 类型渲染一层列表，并递归挂接嵌套列表。"""
        parsed_leaves = [
            parse_list_item_marker(child.content)
            for child in block.content
            if not isinstance(child, ListBlock) and child.content.strip()
        ]
        add_reference_bullets = reference_list_needs_bullets(block)
        container_tag, list_type, class_name = _classify_list(parsed_leaves, add_reference_bullets)
        items: list[dict[str, object]] = []
        expected_value: int | None = None

        for child in block.content:
            if isinstance(child, ListBlock):
                nested = self._render_list(child)
                if not nested:
                    continue
                if not items:
                    items.append({"content": "", "attrs": ' class="mineru-list-item--orphan"', "nested": []})
                nested_items = items[-1]["nested"]
                assert isinstance(nested_items, list)
                nested_items.append(nested)
                continue

            parsed = parse_list_item_marker(child.content)
            item_content, marker = _list_item_content(
                parsed,
                add_reference_bullets,
                explicit_markers=class_name == "mineru-list--explicit",
            )
            rendered = render_inline_content_html(item_content)
            self._observe_inline(rendered)
            if not rendered.html and not marker:
                items.append({"content": "", "attrs": ' class="mineru-list-item--markerless"', "nested": []})
                continue

            attrs: list[str] = []
            if class_name == "mineru-list--explicit":
                attrs.append('class="mineru-list-item--explicit"')
            if container_tag == "ol" and parsed.kind == "ordered" and parsed.value is not None:
                if expected_value is None:
                    expected_value = parsed.value
                if parsed.value != expected_value:
                    attrs.append(f'value="{parsed.value}"')
                expected_value = parsed.value + 1
            marker_html = (
                f'<span class="mineru-list-marker">{html.escape(_replace_html_controls(marker), quote=False)}</span>'
                if marker
                else ""
            )
            if class_name == "mineru-list--explicit" and not marker_html:
                marker_html = '<span class="mineru-list-marker"></span>'
            items.append(
                {
                    "content": f'{marker_html}<span class="mineru-list-content">{rendered.html}</span>',
                    "attrs": f" {' '.join(attrs)}" if attrs else "",
                    "nested": [],
                }
            )

        rendered_items: list[str] = []
        for item in items:
            nested = item["nested"]
            assert isinstance(nested, list)
            item_content = str(item["content"])
            if not item_content and not nested:
                continue
            rendered_items.append(f"<li{item['attrs']}>{item_content}{''.join(str(value) for value in nested)}</li>")
        if not rendered_items:
            return ""

        container_attrs = [f'class="mineru-list {class_name}"']
        if list_type:
            container_attrs.append(f'type="{list_type}"')
        if container_tag == "ol" and parsed_leaves and parsed_leaves[0].value not in (None, 1):
            container_attrs.append(f'start="{parsed_leaves[0].value}"')
        return f"<{container_tag} {' '.join(container_attrs)}>{''.join(rendered_items)}</{container_tag}>"

    def _render_index(self, block: IndexBlock) -> str:
        """递归渲染目录，并仅链接到真实正文标题 anchor。"""
        items: list[dict[str, object]] = []
        for child in block.content:
            if isinstance(child, IndexBlock):
                nested = self._render_index_list(child)
                if not nested:
                    continue
                if not items:
                    items.append({"content": "", "orphan": True, "nested": []})
                nested_items = items[-1]["nested"]
                assert isinstance(nested_items, list)
                nested_items.append(nested)
                continue
            rendered = self._render_index_leaf(child)
            items.append({"content": rendered, "orphan": False, "nested": []})
        inner = _serialize_index_items(items)
        return f'<nav class="mineru-index" aria-label="Table of contents"><ul>{inner}</ul></nav>' if inner else ""

    def _render_index_list(self, block: IndexBlock) -> str:
        """渲染一个嵌套 IndexBlock 为可挂接到父项的 ul。"""
        items: list[dict[str, object]] = []
        for child in block.content:
            if isinstance(child, IndexBlock):
                nested = self._render_index_list(child)
                if nested:
                    if not items:
                        items.append({"content": "", "orphan": True, "nested": []})
                    nested_items = items[-1]["nested"]
                    assert isinstance(nested_items, list)
                    nested_items.append(nested)
                continue
            rendered = self._render_index_leaf(child)
            items.append({"content": rendered, "orphan": False, "nested": []})
        inner = _serialize_index_items(items)
        return f"<ul>{inner}</ul>" if inner else ""

    def _render_index_leaf(self, block: TextBlock | TitleBlockBase) -> str:
        """渲染目录叶子，并在 anchor 命中正文标题时生成内部链接。"""
        content = strip_index_page_tail(block.content)
        rendered = render_inline_content_html(content)
        self._observe_inline(rendered)
        if not rendered.html:
            return ""
        anchor = _anchor_key(block.anchor) if isinstance(block, TitleBlockBase) else ""
        if anchor and (anchor_id := self.anchor_targets.get(anchor)):
            href = quote(anchor_id, safe="-._~")
            return f'<a href="#{href}">{rendered.html}</a>'
        return rendered.html

    def _render_image_block(self, block: ImageBlock) -> str:
        """按原始子块顺序渲染图片主体及重复说明。"""
        parts: list[str] = []
        for child in block.content:
            if isinstance(child, ImageBodyBlock):
                body = self._render_image_body(block, child)
            elif isinstance(child, ImageAnnotationBlock):
                body = self._render_annotation(child)
            else:
                raise TypeError(f"Unsupported image child: {type(child).__name__}")
            if body:
                parts.append(body)
        return f'<figure class="mineru-figure mineru-figure--image">{"".join(parts)}</figure>' if parts else ""

    def _render_image_body(self, parent: ImageBlock, block: ImageBodyBlock) -> str:
        """渲染图片，并把并存的识别内容放入原生 details。"""
        if parent.sub_type == "flowchart":
            mermaid_source = _extract_mermaid_flowchart_source(block.content)
            if mermaid_source is not None:
                return self._render_flowchart_body(block, mermaid_source)
        source = self._safe_block_image_source(block)
        content = self._render_embedded_content(block.content, linkify_text=parent.sub_type != "flowchart")
        alt = _plain_content_text(block.content) or parent.sub_type or "image"
        parts = [self._render_image(source, alt=alt, class_name="mineru-image")] if source else []
        if content.html:
            if source:
                parts.append(self._render_details(content.html, parent.sub_type or "image content"))
            else:
                parts.append(content.html)
        return "".join(parts)

    def _render_flowchart_body(self, block: ImageBodyBlock, mermaid_source: str) -> str:
        """输出 Mermaid canvas，并保留 raster 与源码两级失败回退。"""
        source = self._safe_block_image_source(block)
        escaped_source = html.escape(_replace_html_controls(mermaid_source), quote=False)
        fallback = self._render_image(source, alt="flowchart", class_name="mineru-flowchart-fallback") if source else ""
        fallback_class = " mineru-flowchart--has-raster" if fallback else ""
        details_open = "" if fallback else " open"
        return (
            f'<div class="mineru-flowchart{fallback_class}" data-mermaid-state="pending">'
            '<div class="mineru-flowchart-canvas" role="img" aria-label="flowchart"></div>'
            f"{fallback}</div>"
            f'<details class="mineru-details mineru-flowchart-details"{details_open}>'
            '<summary>flowchart source</summary>'
            f'<pre class="mineru-flowchart-source"><code>{escaped_source}</code></pre>'
            "</details>"
        )

    def _render_table_block(self, block: TableBlock) -> str:
        """按原始子块顺序渲染表格主体、标题与脚注。"""
        parts: list[str] = []
        for child in block.content:
            if isinstance(child, TableBodyBlock):
                body = self._render_table_body(child)
            elif isinstance(child, TableAnnotationBlock):
                body = self._render_annotation(child)
            else:
                raise TypeError(f"Unsupported table child: {type(child).__name__}")
            if body:
                parts.append(body)
        return f'<figure class="mineru-figure mineru-figure--table">{"".join(parts)}</figure>' if parts else ""

    def _render_table_body(self, block: TableBodyBlock) -> str:
        """优先保留安全 HTML table，空间文本和整体图片依次回退。"""
        normalized_content = _replace_html_controls(block.content)
        content = normalized_content.strip()
        if content:
            source_soup = BeautifulSoup(content, "html.parser")
            if source_soup.find("table") is not None:
                rendered = self._render_embedded_content(content)
                if rendered.html and _contains_usable_table(rendered.html):
                    return rendered.html
                source = self._safe_block_image_source(block)
                if source:
                    return self._render_image(source, alt="table", class_name="mineru-table-image")
                return _render_raw_fallback(content)
            return f'<pre class="mineru-table-text">{html.escape(normalized_content, quote=False)}</pre>'
        source = self._safe_block_image_source(block)
        return self._render_image(source, alt="table", class_name="mineru-table-image") if source else ""

    def _render_chart_block(self, block: ChartBlock) -> str:
        """按原始子块顺序渲染 chart 图片、结构内容及说明。"""
        parts: list[str] = []
        for child in block.content:
            if isinstance(child, ChartBodyBlock):
                body = self._render_chart_body(block, child)
            elif isinstance(child, ChartAnnotationBlock):
                body = self._render_annotation(child)
            else:
                raise TypeError(f"Unsupported chart child: {type(child).__name__}")
            if body:
                parts.append(body)
        return f'<figure class="mineru-figure mineru-figure--chart">{"".join(parts)}</figure>' if parts else ""

    def _render_chart_body(self, parent: ChartBlock, block: ChartBodyBlock) -> str:
        """渲染 chart 图片，并将并存结构内容放入 details。"""
        source = self._safe_block_image_source(block)
        content = self._render_chart_content(block.content)
        parts = [self._render_image(source, alt=parent.sub_type or "chart", class_name="mineru-chart-image")] if source else []
        if content.html:
            if source:
                parts.append(self._render_details(content.html, parent.sub_type or "chart content"))
            else:
                parts.append(content.html)
        return "".join(parts)

    def _render_chart_content(self, content: str) -> HtmlInlineResult:
        """按 HTML、严格 GFM 表格、普通行内内容的顺序渲染 chart content。"""
        normalized = content.strip()
        if not normalized:
            return HtmlInlineResult("")
        if is_supported_html_markup(normalized):
            return self._render_embedded_content(normalized)
        if looks_like_gfm_table(normalized):
            rendered = render_gfm_table_html(normalized)
            if rendered is None:
                return HtmlInlineResult(_render_raw_fallback(content))
            self._observe_inline(rendered)
            return rendered
        rendered = render_inline_content_html(content)
        self._observe_inline(rendered)
        return rendered

    def _render_code_block(self, block: CodeBlock) -> str:
        """按原始子块顺序渲染代码或算法及其说明。"""
        parts: list[str] = []
        for child in block.content:
            if isinstance(child, CodeBodyBlock):
                body = self._render_code_body(block, child)
            elif isinstance(child, CodeAnnotationBlock):
                body = self._render_annotation(child)
            else:
                raise TypeError(f"Unsupported code child: {type(child).__name__}")
            if body:
                parts.append(body)
        return f'<figure class="mineru-figure mineru-figure--code">{"".join(parts)}</figure>' if parts else ""

    def _render_code_body(self, parent: CodeBlock, block: CodeBodyBlock) -> str:
        """代码使用 Prism class，算法使用共享 inline AST 保留空白。"""
        if parent.sub_type == BlockType.CODE:
            escaped = html.escape(_replace_html_controls(block.content), quote=False)
            if not block.content:
                return '<pre class="mineru-code"><code></code></pre>'
            language = _normalize_prism_language(parent.guess_lang)
            class_attr = f' class="language-{language}"' if language else ""
            pre_class = f"mineru-code language-{language}" if language else "mineru-code"
            self.has_prism = bool(language) or self.has_prism
            return f'<pre class="{pre_class}"><code{class_attr}>{escaped}</code></pre>'
        if parent.sub_type == RAW_ALGORITHM:
            rendered = render_inline_nodes_html(
                parse_inline_content(block.content),
                linkify_text=False,
                separate_adjacent_math=True,
                preserve_newlines=True,
            )
            self._observe_inline(rendered)
            return f'<div class="mineru-algorithm">{rendered.html}</div>' if rendered.html else ""
        raise ValueError(f"Unsupported code subtype: {parent.sub_type}")

    def _render_annotation(
        self,
        block: ImageAnnotationBlock | TableAnnotationBlock | ChartAnnotationBlock | CodeAnnotationBlock,
    ) -> str:
        """把视觉说明渲染为保持原类型的独立段落。"""
        rendered = render_inline_content_html(block.content)
        self._observe_inline(rendered)
        if not rendered.html:
            return ""
        is_caption = str(block.type).endswith("caption")
        role_class = "mineru-caption" if is_caption else "mineru-footnote"
        type_class = html.escape(str(block.type).replace("_", "-"), quote=True)
        return f'<p class="{role_class} {role_class}--{type_class}">{rendered.html}</p>'

    def _render_embedded_content(self, content: str, *, linkify_text: bool = True) -> HtmlInlineResult:
        """安全处理 body 富 HTML；普通内容走共享 inline AST。"""
        normalized = content.strip()
        if not normalized:
            return HtmlInlineResult("")
        if not is_supported_html_markup(normalized):
            rendered = render_inline_nodes_html(parse_inline_content(content), linkify_text=linkify_text)
            self._observe_inline(rendered)
            return rendered
        sanitized = sanitize_html_fragment(normalized, asset_base_url=self.asset_base_url)
        rendered = self._replace_safe_equations(sanitized)
        self._observe_inline(rendered)
        return rendered

    def _replace_safe_equations(self, markup: str) -> HtmlInlineResult:
        """把 sanitizer 保留的 eq 标签替换为可信 MathJax carrier。"""
        if not markup:
            return HtmlInlineResult("")
        soup = BeautifulSoup(markup, "html.parser")
        has_math = False
        for equation in list(soup.find_all("eq")):
            rendered = render_math_html(equation.get_text(), display=False)
            if not rendered.html:
                equation.decompose()
                continue
            replacement = BeautifulSoup(rendered.html, "html.parser")
            equation.replace_with(*list(replacement.contents))
            has_math = True
        return HtmlInlineResult(str(soup), has_math)

    def _safe_block_image_source(self, block: EquationBlock | ImageBodyBlock | TableBodyBlock | ChartBodyBlock) -> str | None:
        """解析 block 图片来源，并应用 HTML renderer 的 URL 安全策略。"""
        if block.image_path:
            return sanitize_image_source(block.image_path, asset_base_url=self.asset_base_url)
        if block.image_base64:
            return sanitize_image_source(block.image_base64)
        return None

    def _render_image(self, source: str, *, alt: str, class_name: str) -> str:
        """构造属性已转义的安全 img 元素。"""
        return (
            f'<img class="{class_name}" src="{html.escape(source, quote=True)}" '
            f'alt="{html.escape(_replace_html_controls(alt), quote=True)}">'
        )

    def _render_details(self, content: str, summary: str) -> str:
        """构造浏览器原生折叠详情，不引入额外脚本。"""
        return (
            '<details class="mineru-details">'
            f"<summary>{html.escape(_replace_html_controls(summary), quote=False)}</summary>{content}</details>"
        )

    def _observe_inline(self, rendered: HtmlInlineResult) -> None:
        """累计当前文档是否需要加载 MathJax。"""
        self.has_math = rendered.has_math or self.has_math

    def _render_standalone(self, article: str) -> str:
        """用固定 CSS 和按需 CDN 资源包装完整 HTML 文档。"""
        title = html.escape(
            _replace_html_controls(_resolve_document_title(self.middle_json, self.document_title)),
            quote=False,
        )
        styles = _load_html_styles()
        dependencies = []
        if self.has_math:
            dependencies.append(_mathjax_head())
        if self.has_prism:
            dependencies.append(_prism_head())
        if self.has_mermaid:
            dependencies.append(_mermaid_head())
        dependency_html = "\n".join(dependencies)
        return (
            "<!doctype html>\n"
            '<html lang="und">\n'
            "<head>\n"
            '<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>{title}</title>\n"
            f"<style>{styles}</style>\n"
            f"{dependency_html}\n"
            "</head>\n"
            '<body class="mineru-html-body">\n'
            f"{article}\n"
            "</body>\n"
            "</html>"
        )


def render_html(
    middle_json: MiddleJson,
    *,
    mode: RenderMode = RenderMode.DEFAULT,
    asset_base_url: str = "",
    standalone: bool = True,
    document_title: str | None = None,
) -> str:
    """把严格 MiddleJson 无副作用地渲染为 HTML 文档或单根 fragment。"""
    if not isinstance(middle_json, MiddleJson):
        raise TypeError("render_html expects a MiddleJson instance")
    if not isinstance(mode, RenderMode):
        raise TypeError("mode must be a RenderMode value")
    if not isinstance(asset_base_url, str):
        raise TypeError("asset_base_url must be a string")
    if not isinstance(standalone, bool):
        raise TypeError("standalone must be a bool")
    if document_title is not None and not isinstance(document_title, str):
        raise TypeError("document_title must be a string or None")
    return _HtmlRenderer(
        middle_json,
        mode=mode,
        asset_base_url=asset_base_url,
        standalone=standalone,
        document_title=document_title,
    ).render()


def _classify_list(items: list[ListItem], add_reference_bullets: bool) -> tuple[str, str | None, str]:
    """根据直属 marker 选择原生列表类型或显式 marker 模式。"""
    if add_reference_bullets:
        return "ul", None, "mineru-list--reference"
    if items and all(item.kind == "unordered" for item in items):
        return "ul", None, "mineru-list--unordered"
    if items and all(item.kind == "ordered" for item in items):
        styles = {item.ordered_style for item in items}
        if len(styles) == 1:
            style = next(iter(styles))
            list_type = {
                "lower-alpha": "a",
                "upper-alpha": "A",
                "lower-roman": "i",
                "upper-roman": "I",
            }.get(style or "")
            return "ol", list_type, "mineru-list--ordered"
    if items and all(item.kind == "none" for item in items):
        return "ul", None, "mineru-list--unmarked"
    return "ul", None, "mineru-list--explicit"


def _list_item_content(
    item: ListItem,
    add_reference_bullets: bool,
    *,
    explicit_markers: bool,
) -> tuple[str, str | None]:
    """决定一个列表项应剥离、保留还是显式显示源 marker。"""
    if add_reference_bullets:
        if item.kind == "unordered":
            return item.body, None
        original = f"{item.leading}{item.marker or ''}{item.separator}{item.body}"
        return original, None
    if explicit_markers:
        return item.body, item.marker
    if item.kind in {"unordered", "ordered"}:
        return item.body, None
    return item.body, item.marker


def _serialize_index_items(items: list[dict[str, object]]) -> str:
    """把目录项及其已挂接的嵌套 ul 序列化为 li。"""
    rendered: list[str] = []
    for item in items:
        nested = item["nested"]
        assert isinstance(nested, list)
        content = str(item["content"])
        if not content and not nested:
            continue
        class_attr = ' class="mineru-list-item--orphan"' if item["orphan"] else ""
        rendered.append(f"<li{class_attr}>{content}{''.join(str(value) for value in nested)}</li>")
    return "".join(rendered)


def _collect_title_anchor_ids(middle_json: MiddleJson) -> dict[str, str]:
    """为正文首次出现的非空 anchor 分配无空白且 document-wide 唯一的 HTML id。"""
    anchor_ids: dict[str, str] = {}
    used_ids: set[str] = set()
    for page in middle_json.pages:
        for block in page.blocks:
            if not isinstance(block, TitleBlockBase):
                continue
            anchor = _anchor_key(block.anchor)
            visible_title = inline_plain_text(parse_inline_content(block.content)).strip()
            if not anchor or not visible_title or anchor in anchor_ids:
                continue
            base_id = _html_anchor_base(anchor)
            candidate = base_id
            suffix = 2
            while candidate in used_ids:
                candidate = f"{base_id}-{suffix}"
                suffix += 1
            anchor_ids[anchor] = candidate
            used_ids.add(candidate)
    return anchor_ids


def _plain_content_text(content: str) -> str:
    """从 body 内容提取安全 img alt 所需的可见纯文本。"""
    if not content:
        return ""
    if is_supported_html_markup(content):
        return BeautifulSoup(content, "html.parser").get_text(" ", strip=True)
    return inline_plain_text(parse_inline_content(content)).strip()


def _extract_mermaid_flowchart_source(content: str) -> str | None:
    """提取受限 Mermaid flowchart fence，拒绝其他图类型和可改写配置的语法。"""
    normalized = _replace_html_controls(content).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return None
    lines = normalized.split("\n")
    if len(lines) < 3:
        return None
    opening = re.fullmatch(r"(?P<fence>\x60{3,})[ \t]*mermaid[ \t]*", lines[0], re.IGNORECASE)
    if opening is None:
        return None
    fence_length = len(opening.group("fence"))
    if re.fullmatch(rf"\x60{{{fence_length},}}[ \t]*", lines[-1]) is None:
        return None
    source = "\n".join(lines[1:-1]).strip("\n")
    if not source or len(source) > _MERMAID_MAX_TEXT_SIZE:
        return None
    if source.lstrip().startswith("---") or "%%{" in source:
        return None
    first_statement = next(
        (line.strip() for line in source.splitlines() if line.strip() and not line.lstrip().startswith("%%")),
        "",
    )
    if _MERMAID_FLOWCHART_HEADER_RE.match(first_statement) is None:
        return None
    return source


def _contains_usable_table(markup: str) -> bool:
    """确认清洗后的 markup 至少保留一个含单元格的 table。"""
    soup = BeautifulSoup(markup, "html.parser")
    return any(table.find(("td", "th")) is not None for table in soup.find_all("table"))


def _render_raw_fallback(content: str) -> str:
    """把无法安全结构化的原内容转义为可见 pre，避免静默丢失。"""
    return f'<pre class="mineru-raw-fallback">{html.escape(_replace_html_controls(content), quote=False)}</pre>'


def _normalize_prism_language(language: str | None) -> str | None:
    """把模型语言名归一化为不可构造路径穿越的 Prism component 名。"""
    normalized = (language or "").strip().lower()
    normalized = _PRISM_LANGUAGE_ALIASES.get(normalized, normalized.replace("_", "-"))
    if not normalized or normalized in _BLOCKED_PRISM_LANGUAGE_NAMES:
        return None
    return normalized if _SAFE_PRISM_LANGUAGE_RE.fullmatch(normalized) else None


def _resolve_document_title(middle_json: MiddleJson, explicit_title: str | None) -> str:
    """按显式参数、首个正文文档标题、固定回退值解析 HTML title。"""
    if explicit_title and explicit_title.strip():
        return explicit_title.strip()
    for page in middle_json.pages:
        for block in page.blocks:
            if isinstance(block, DocTitleBlock):
                title = inline_plain_text(parse_inline_content(block.content)).strip()
                if title:
                    return title
    return "MinerU Document"


def _anchor_key(anchor: str | None) -> str:
    """保留 producer anchor 身份，仅去除首尾空白供标题与目录精确匹配。"""
    return (anchor or "").strip()


def _html_anchor_base(anchor: str) -> str:
    """把原始 anchor 转成无空白的 HTML id 基值，碰撞由 document registry 处理。"""
    return re.sub(r"\s+", "-", _replace_html_controls(anchor))


def _replace_html_controls(content: str) -> str:
    """替换 HTML 不允许的 C0 控制字符与不可编码的孤立 surrogate。"""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]", "\ufffd", content)


@lru_cache(maxsize=1)
def _load_html_styles() -> str:
    """从包资源读取并缓存压缩后的 MinerU 独立样式。"""
    root = resources.files("mineru").joinpath("resources", "html")
    return root.joinpath(_STYLE_RESOURCE_NAME).read_text(encoding="utf-8")


def _mathjax_head() -> str:
    """返回固定、收紧公式输入能力的 MathJax 4.1.2 head 片段。"""
    return f"""<script>
window.MathJax = {{
  loader: {{load: ['ui/safe']}},
  tex: {{
    inlineMath: [['\\\\(', '\\\\)']],
    displayMath: [['\\\\[', '\\\\]']],
    processEscapes: true,
    processEnvironments: false,
    processRefs: false,
    packages: {{'[-]': ['require']}}
  }},
  options: {{
    ignoreHtmlClass: 'mineru-document',
    processHtmlClass: 'mineru-math',
    enableMenu: false,
    enableEnrichment: false,
    safeOptions: {{allow: {{URLs: 'none', classes: 'none', cssIDs: 'none', styles: 'none'}}}}
  }},
  startup: {{
    ready: function () {{
      window.MathJax.startup.defaultReady();
      var options = window.MathJax.startup.document.options;
      options.enableMenu = false;
      options.enableEnrichment = false;
    }}
  }}
}};
</script>
<script id="MathJax-script" defer src="{_MATHJAX_URL}"
  integrity="{_MATHJAX_INTEGRITY}" crossorigin="anonymous" referrerpolicy="no-referrer"></script>"""


def _prism_head() -> str:
    """返回固定 Prism core、Autoloader 和当前文档局部高亮脚本。"""
    return f"""<script defer data-manual src="{_PRISM_CORE_URL}"
  integrity="{_PRISM_CORE_INTEGRITY}" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<script defer src="{_PRISM_AUTOLOADER_URL}"
  integrity="{_PRISM_AUTOLOADER_INTEGRITY}" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<script>
document.addEventListener('DOMContentLoaded', function () {{
  if (!window.Prism || !Prism.plugins || !Prism.plugins.autoloader) return;
  Prism.plugins.autoloader.languages_path = '{_PRISM_LANGUAGES_PATH}';
  var root = document.querySelector('.mineru-document');
  if (root) Prism.highlightAllUnder(root);
}});
</script>"""


def _mermaid_head() -> str:
    """返回固定 Mermaid 入口和逐图安全渲染、失败回退脚本。"""
    return f"""<script defer src="{_MERMAID_URL}"
  integrity="{_MERMAID_INTEGRITY}" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<script>
function markMineruMermaidError(host) {{
  host.dataset.mermaidState = 'error';
  if (!host.classList.contains('mineru-flowchart--has-raster')) {{
    var details = host.nextElementSibling;
    if (details && details.classList.contains('mineru-flowchart-details')) details.open = true;
  }}
}}
document.addEventListener('DOMContentLoaded', function () {{
  var root = document.querySelector('.mineru-document');
  var hosts = root ? Array.from(root.querySelectorAll('.mineru-flowchart')) : [];
  if (!hosts.length) return;
  if (!window.mermaid) {{
    hosts.forEach(markMineruMermaidError);
    return;
  }}
  try {{
    window.mermaid.initialize({{
      startOnLoad: false,
      securityLevel: 'strict',
      suppressErrorRendering: true,
      theme: 'neutral',
      maxTextSize: 50000,
      maxEdges: 500,
      flowchart: {{htmlLabels: false, useMaxWidth: true}}
    }});
  }} catch (_error) {{
    hosts.forEach(markMineruMermaidError);
    return;
  }}
  (async function () {{
    for (var index = 0; index < hosts.length; index += 1) {{
      var host = hosts[index];
      var details = host.nextElementSibling;
      var source = details && details.querySelector('.mineru-flowchart-source code');
      var canvas = host.querySelector('.mineru-flowchart-canvas');
      if (!source || !canvas) {{
        markMineruMermaidError(host);
        continue;
      }}
      var renderId = 'mineru-mermaid-' + index;
      while (document.getElementById(renderId)) renderId += '-x';
      host.dataset.mermaidState = 'rendering';
      try {{
        var result = await window.mermaid.render(renderId, source.textContent);
        canvas.innerHTML = result.svg;
        host.dataset.mermaidState = 'rendered';
      }} catch (_error) {{
        markMineruMermaidError(host);
      }}
    }}
  }})();
}});
</script>"""


__all__ = ["render_html"]
