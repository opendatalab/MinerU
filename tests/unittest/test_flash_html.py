from __future__ import annotations

import asyncio
import base64
from copy import copy
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from bs4 import BeautifulSoup
from PIL import Image

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.errors import InvalidRequestError
from mineru.doclib.services.parse_svc import ParseService
from mineru.model.flash import HtmlModel
from mineru.model.flash.html import HtmlResourceLimitError, HtmlSourceContext
from mineru.model.flash.html import converter as html_converter_module
from mineru.model.flash.html import document as html_document_module
from mineru.model.flash.html import resources as html_resources_module
from mineru.model.flash.html.resources import HtmlResourceContext
from mineru.parser import ParseResult, parse, parse_async
from mineru.parser import api_server
from mineru.parser.api_server import CreateJobRequest, FileStore
from mineru.render import RenderMode
from mineru.render.docx import render_docx
from mineru.render.html import render_html
from mineru.render.markdown import render_markdown
from mineru.render.structured_content import render_structured_content
from mineru.types import BlockType, CodeBlock, CodeBodyBlock, ImageBlock, ImageBodyBlock, MiddleJson


_PNG_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2l9sAAAAASUVORK5CYII="


def _all_raw_blocks(model_pages: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    """按页展开 raw model-list，方便断言 HTML 映射结果。"""
    return [block for page in model_pages for block in page]


def _image_body(middle: MiddleJson) -> ImageBodyBlock:
    """返回文档中首个严格图片 body。"""
    image = next(block for block in middle.pages[0].blocks if isinstance(block, ImageBlock))
    return next(child for child in image.content if isinstance(child, ImageBodyBlock))


def _wire_contract_middle() -> MiddleJson:
    """构造覆盖全部顶层类型、visual child、列表和目录叶子的严格文档。"""
    return MiddleJson.model_validate(
        {
            "pages": [
                {
                    "page_idx": 3,
                    "blocks": [
                        {"type": "doc_title", "index": 0, "level": 1, "anchor": "doc-anchor", "content": "Document"},
                        {
                            "type": "paragraph_title",
                            "index": 1,
                            "level": 2,
                            "anchor": "section-anchor",
                            "content": "Section",
                        },
                        {
                            "type": "text",
                            "index": 2,
                            "content": "Text & value < 3 <eq>x+1</eq> &lt;eq&gt;literal&lt;/eq&gt;",
                        },
                        {"type": "ref_text", "index": 3, "content": "Reference text"},
                        {
                            "type": "list",
                            "index": 4,
                            "sub_type": "ref_text",
                            "content": [
                                {"type": "ref_text", "content": "[1] Reference"},
                                {
                                    "type": "list",
                                    "content": [{"type": "text", "content": "- Nested item"}],
                                },
                            ],
                        },
                        {
                            "type": "index",
                            "index": 5,
                            "content": [
                                {
                                    "type": "paragraph_title",
                                    "level": 2,
                                    "anchor": "section-anchor",
                                    "content": "Section\t9",
                                },
                                {
                                    "type": "index",
                                    "content": [{"type": "text", "content": "Unlinked entry"}],
                                },
                            ],
                        },
                        {
                            "type": "image",
                            "index": 6,
                            "sub_type": "diagram",
                            "content": [
                                {
                                    "type": "image_body",
                                    "index": 6,
                                    "content": "Image body",
                                    "image_base64": _PNG_URI,
                                },
                                {"type": "image_caption", "index": 7, "content": "Image caption"},
                                {"type": "image_footnote", "index": 8, "content": "Image footnote"},
                            ],
                        },
                        {
                            "type": "table",
                            "index": 9,
                            "content": [
                                {
                                    "type": "table_body",
                                    "index": 9,
                                    "content": "<table><tr><td>A</td></tr></table>",
                                },
                                {"type": "table_caption", "index": 10, "content": "Table caption"},
                                {"type": "table_footnote", "index": 11, "content": "Table footnote"},
                            ],
                        },
                        {
                            "type": "chart",
                            "index": 12,
                            "sub_type": "bar",
                            "content": [
                                {
                                    "type": "chart_body",
                                    "index": 12,
                                    "content": "| A | B |\n| - | - |\n| 1 | 2 |",
                                    "image_base64": _PNG_URI,
                                },
                                {"type": "chart_caption", "index": 13, "content": "Chart caption"},
                                {"type": "chart_footnote", "index": 14, "content": "Chart footnote"},
                            ],
                        },
                        {
                            "type": "code",
                            "index": 15,
                            "sub_type": "code",
                            "guess_lang": "python",
                            "content": [
                                {"type": "code_body", "index": 15, "content": "print('ok')"},
                                {"type": "code_caption", "index": 16, "content": "Code caption"},
                                {"type": "code_footnote", "index": 17, "content": "Code footnote"},
                            ],
                        },
                        {
                            "type": "code",
                            "index": 18,
                            "sub_type": "algorithm",
                            "content": [
                                {"type": "code_body", "index": 18, "content": "if x < y:\n  z=<eq>a</eq>"},
                                {"type": "code_caption", "index": 19, "content": "Algorithm caption"},
                                {"type": "code_footnote", "index": 20, "content": "Algorithm footnote"},
                            ],
                        },
                        {"type": "equation", "index": 21, "content": "y^2\\tag{1}"},
                        {"type": "page_footnote", "index": 22, "anchor": "note-anchor", "content": "Page footnote"},
                        {"type": "header", "index": 23, "content": "Header"},
                        {"type": "footer", "index": 24, "content": "Footer"},
                        {"type": "page_number", "index": 25, "content": "3"},
                        {"type": "aside_text", "index": 26, "content": "Aside"},
                    ],
                }
            ],
            "is_full_document": True,
            "file_suffix": "html",
            "effort": "flash",
            "parse_mode": "txt",
            "mineru_version": "test",
        }
    )


def _semantic_block_signature(block: object) -> tuple[object, ...]:
    """提取往返测试关心的类型、元数据和递归 child 类型。"""
    content = getattr(block, "content", None)
    children = tuple(_semantic_block_signature(child) for child in content) if isinstance(content, list) else ()
    return (
        str(getattr(block, "type")),
        getattr(block, "sub_type", None),
        getattr(block, "guess_lang", None),
        getattr(block, "anchor", None),
        getattr(block, "level", None),
        children,
    )


def test_html_doc_analyze_projects_static_semantics_and_renderers() -> None:
    """验证 HTML 主链路保留正文结构、行内语义、公式、图片 URL 与固定元数据。"""
    payload = b"""<!doctype html>
    <html><head><title>Demo - Example</title><meta property="og:site_name" content="Example">
    <style>.hidden { display:none } .strong { font-weight:700 }</style></head>
    <body><nav>menu links</nav><main><article>
      <h1 id="top">Demo</h1>
      <p>Hello <span class="strong">world</span>, <code>a`b</code>,
         <a href="#top">back</a>.</p>
      <p class="hidden">secret</p><script>alert(1)</script>
      <ol start="3" reversed><li value="9">Three</li><li>Four</li></ol>
      <table><caption>Data</caption><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>
      <pre><code class="language-python">print(1)</code></pre>
      <script type="math/tex; mode=display">x^2</script>
      <img src="https://cdn.example.com/a.png" alt="Remote image">
    </article></main></body></html>"""

    middle, model = doc_analyze(payload, effort="xhigh", parse_mode="ocr", file_suffix="html")
    async_middle, async_model = asyncio.run(aio_doc_analyze(payload, effort="medium", parse_mode="auto", file_suffix="html"))

    assert middle.model_dump() == async_middle.model_dump()
    assert model.pages == async_model.pages
    assert middle.file_suffix == model.file_suffix == "html"
    assert middle.effort == model.effort == "flash"
    assert middle.parse_mode == model.parse_mode == "txt"
    assert middle.is_full_document is True
    assert [page.page_idx for page in middle.pages] == [0]
    assert all(block.bbox is None for block in middle.pages[0].blocks)

    raw_blocks = _all_raw_blocks(model.pages)
    raw_types = {block["type"] for block in raw_blocks}
    assert {
        BlockType.DOC_TITLE,
        BlockType.TEXT,
        BlockType.LIST,
        BlockType.TABLE,
        BlockType.CODE,
        BlockType.EQUATION,
        BlockType.IMAGE,
    } <= raw_types
    assert "secret" not in str(raw_blocks)
    assert "alert(1)" not in str(raw_blocks)
    assert any(block.get("guess_lang") == "python" for block in raw_blocks)
    assert _image_body(middle).image_url == "https://cdn.example.com/a.png"
    assert ParseResult.from_dict(ParseResult(middle_json=middle).to_dict()).middle_json == middle

    markdown = render_markdown(middle)
    assert "# Demo" in markdown
    assert "**world**" in markdown
    assert "``a`b``" in markdown
    assert "3. Three" in markdown and "4. Four" in markdown
    assert "```python" in markdown and "x^2" in markdown
    assert "https://cdn.example.com/a.png" in markdown
    assert "<table" in render_html(middle)
    assert render_structured_content(middle)["file_suffix"] == "html"
    docx = render_docx(middle)
    assert docx.startswith(b"PK")
    with ZipFile(BytesIO(docx)) as archive:
        relationships = archive.read("word/_rels/document.xml.rels").decode()
        assert "https://cdn.example.com/a.png" in relationships


def test_html_auto_selection_preserves_all_repeated_forum_posts() -> None:
    """验证重复 article 场景不会只保留论坛中的首个帖子。"""
    payload = b"""<html><body><header>Forum</header><main>
      <article class="post"><h2>First post</h2><p>First body with enough useful discussion text.</p></article>
      <article class="post"><h2>Second post</h2><p>Second body with another useful discussion answer.</p></article>
    </main><footer>footer</footer></body></html>"""

    markdown = render_markdown(doc_analyze(payload, file_suffix="html")[0])

    assert "First post" in markdown and "First body" in markdown
    assert "Second post" in markdown and "Second body" in markdown


def test_html_auto_selection_preserves_ancestor_text_styles() -> None:
    """验证正文候选复制后仍继承 body 与外层容器的受支持文字样式。"""
    detail = "Inherited text " * 20
    payload = f"""<html><body style="font-weight:bold"><aside style="font-style:italic;text-decoration:underline">
      <main><p>{detail}</p></main></aside></body></html>""".encode()

    _, model = doc_analyze(payload, file_suffix="html")

    assert model.pages[0] == [
        {
            "type": BlockType.TEXT,
            "content": f'<text style="bold,italic,underline">{detail}</text>',
        }
    ]


def test_html_auto_selection_handles_colonized_candidate_ancestor() -> None:
    """验证正文候选位于 Office 冒号标签下时可复制祖先链且不会抛出非法标签异常。"""
    detail = "Colonized ancestor content " * 20
    payload = f"""<html><body><o:smarttag style="font-style:italic">
      <main><p>{detail}</p></main></o:smarttag></body></html>""".encode()

    _, model = doc_analyze(payload, file_suffix="html")

    assert model.pages[0] == [{"type": BlockType.TEXT, "content": f'<text style="italic">{detail}</text>'}]


def test_html_auto_selection_rejects_single_section_from_document_index() -> None:
    """验证多个同级 section 构成的文档索引会保留全部章节而非选择最长一节。"""
    detail = b" useful explanatory content with enough words to qualify as an independent scored candidate" * 4
    payload = (
        b"<html><body><h1>Review index</h1>"
        + b"<section><h2>First</h2><p>First section"
        + detail
        + b"</p><pre>first code</pre></section>"
        + b"<section><h2>Second</h2><p>Second section"
        + detail
        + b"</p><pre>second code</pre></section>"
        + b"<section><h2>Third</h2><p>Third section"
        + detail
        + b"</p><pre>third code</pre></section></body></html>"
    )

    markdown = render_markdown(doc_analyze(payload, file_suffix="html")[0])

    assert "First section" in markdown and "first code" in markdown
    assert "Second section" in markdown and "second code" in markdown
    assert "Third section" in markdown and "third code" in markdown


def test_html_referenced_external_footnote_keeps_anchor_and_content() -> None:
    """验证正文候选外但被引用的 HTML footnote 会追加并生成可兑现 anchor。"""
    payload = b"""<html><body><article><h1>Notes</h1>
      <p>Claim <a href="#note-1">[1]</a>.</p></article>
      <footer><aside id="note-1" role="doc-footnote"><p>Footnote body.</p></aside></footer>
    </body></html>"""

    middle, _ = doc_analyze(payload, file_suffix="html")
    markdown = render_markdown(middle)
    footnote = next(block for block in middle.pages[0].blocks if block.type == BlockType.PAGE_FOOTNOTE)

    assert footnote.anchor is not None  # type: ignore[union-attr]
    assert "Footnote body." in footnote.content  # type: ignore[union-attr]
    assert f"](#{footnote.anchor})" in markdown  # type: ignore[union-attr]
    assert f'id="{footnote.anchor}" class="mineru-page-footnote"' in markdown  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "ancestor_attributes",
    ['style="display:none"', 'style="opacity:0"', "hidden", 'aria-hidden="true"'],
)
def test_html_referenced_external_footnote_respects_hidden_ancestor(ancestor_attributes: str) -> None:
    """验证正文外引用脚注不会脱离原始整树隐藏祖先后泄漏到输出。"""
    detail = "Useful main article text " * 20
    payload = f"""<html><body><main><h1>Title</h1><p>{detail}<a href="#fn1">[1]</a></p></main>
      <aside {ancestor_attributes}><div id="fn1" role="doc-footnote">HIDDEN NOTE</div></aside>
      </body></html>""".encode()

    middle = doc_analyze(payload, file_suffix="html")[0]
    markdown = render_markdown(middle)

    assert "HIDDEN NOTE" not in markdown
    assert not any(block.type == BlockType.PAGE_FOOTNOTE for block in middle.pages[0].blocks)


def test_html_referenced_external_footnote_preserves_inherited_visibility() -> None:
    """验证复制脚注保留祖先 visibility:hidden，同时允许后代显式恢复可见。"""
    detail = "Useful main article text " * 20
    payload = f"""<html><body><main><h1>Title</h1><p>{detail}<a href="#fn1">[1]</a></p></main>
      <aside style="visibility:hidden"><div id="fn1" role="doc-footnote">HIDDEN NOTE
      <span style="visibility:visible">VISIBLE NOTE</span></div></aside></body></html>""".encode()

    middle = doc_analyze(payload, file_suffix="html")[0]
    markdown = render_markdown(middle)
    footnote = next(block for block in middle.pages[0].blocks if block.type == BlockType.PAGE_FOOTNOTE)

    assert "HIDDEN NOTE" not in markdown
    assert footnote.content == "VISIBLE NOTE"  # type: ignore[union-attr]
    assert f"](#{footnote.anchor})" in markdown  # type: ignore[union-attr]


def test_html_referenced_external_footnote_preserves_inherited_text_styles() -> None:
    """验证正文外引用脚注复制后仍携带祖先提供的受支持文字样式。"""
    detail = "Useful main article text " * 20
    payload = f"""<html><body><main><p>{detail}<a href="#fn1">[1]</a></p></main>
      <aside style="font-weight:bold;font-style:italic;text-decoration:underline line-through">
      <div id="fn1" role="doc-footnote">Styled note</div></aside></body></html>""".encode()

    _, model = doc_analyze(payload, file_suffix="html")
    footnote = next(block for block in model.pages[0] if block["type"] == BlockType.PAGE_FOOTNOTE)

    assert footnote["content"] == '<text style="bold,italic,underline,strikethrough">Styled note</text>'


@pytest.mark.parametrize(
    "href",
    [
        "#fn1",
        "page.html#fn1",
        "https://example.com/page.html#fn1",
    ],
)
def test_html_auto_selection_appends_notes_for_all_same_document_url_forms(href: str) -> None:
    """验证 auto 正文外脚注可由纯 fragment、相对或绝对同文档 URL 引用。"""
    detail = "Useful main article text " * 20
    payload = f"""<html><body><main><article><h1>Title</h1><p>{detail}
      <a href="{href}">[1]</a></p></article></main>
      <footer><aside id="fn1" role="doc-footnote"><p>Outside footnote.</p></aside></footer>
      </body></html>""".encode()
    context = HtmlSourceContext(source_uri="https://example.com/page.html")

    middle = doc_analyze(payload, file_suffix="html", source_context=context)[0]
    markdown = render_markdown(middle)
    footnote = next(block for block in middle.pages[0].blocks if block.type == BlockType.PAGE_FOOTNOTE)

    assert footnote.content == "Outside footnote."  # type: ignore[union-attr]
    assert f"](#{footnote.anchor})" in markdown  # type: ignore[union-attr]
    assert "https://example.com/page.html#fn1" not in markdown


@pytest.mark.parametrize(
    ("href", "note_id"),
    [
        ("#fn%31", "fn1"),
        ("page.html#note%20one", "note one"),
        ("https://example.com/page.html#%E8%84%9A%E6%B3%A8", "脚注"),
    ],
)
def test_html_auto_selection_decodes_same_document_note_fragments(href: str, note_id: str) -> None:
    """验证数字、空格与非 ASCII fragment 解码后可关联正文外脚注。"""
    detail = "Useful main article text " * 20
    payload = f"""<html><head><meta charset="utf-8"></head><body><main><article><h1>Title</h1><p>{detail}
      <a href="{href}">[1]</a></p></article></main>
      <footer><aside id="{note_id}" role="doc-footnote"><p>Outside footnote.</p></aside></footer>
      </body></html>""".encode()
    context = HtmlSourceContext(source_uri="https://example.com/page.html")

    middle = doc_analyze(payload, file_suffix="html", source_context=context)[0]
    markdown = render_markdown(middle)
    footnote = next(block for block in middle.pages[0].blocks if block.type == BlockType.PAGE_FOOTNOTE)

    assert footnote.content == "Outside footnote."  # type: ignore[union-attr]
    assert f"](#{footnote.anchor})" in markdown  # type: ignore[union-attr]
    assert href not in markdown


@pytest.mark.parametrize(
    ("href", "expected_target"),
    [
        ("other.html#fn1", "https://example.com/other.html#fn1"),
        ("https://other.example/page.html#fn1", "https://other.example/page.html#fn1"),
        ("page.html?view=2#fn1", "https://example.com/page.html?view=2#fn1"),
    ],
)
def test_html_auto_selection_does_not_append_notes_for_other_documents(href: str, expected_target: str) -> None:
    """验证不同 path、origin 或 query 的 URL 不会借 fragment 追加当前文档脚注。"""
    detail = "Useful main article text " * 20
    payload = f"""<html><body><main><article><h1>Title</h1><p>{detail}
      <a href="{href}">[1]</a></p></article></main>
      <footer><aside id="fn1" role="doc-footnote"><p>Outside footnote.</p></aside></footer>
      </body></html>""".encode()
    context = HtmlSourceContext(source_uri="https://example.com/page.html")

    middle = doc_analyze(payload, file_suffix="html", source_context=context)[0]
    markdown = render_markdown(middle)

    assert "Outside footnote." not in markdown
    assert not any(block.type == BlockType.PAGE_FOOTNOTE for block in middle.pages[0].blocks)
    assert expected_target in markdown


@pytest.mark.parametrize(
    ("base_href", "expected_note"),
    [
        ("/docs/", True),
        ("/other/", False),
    ],
)
def test_html_same_document_note_resolution_honors_base_href(base_href: str, expected_note: bool) -> None:
    """验证 base href 参与相对脚注 URL 的文档身份判定。"""
    detail = "Useful main article text " * 20
    payload = f"""<html><head><base href="{base_href}"></head><body><main><article><h1>Title</h1>
      <p>{detail}<a href="page.html#fn1">[1]</a></p></article></main>
      <footer><aside id="fn1" role="doc-footnote"><p>Outside footnote.</p></aside></footer>
      </body></html>""".encode()
    context = HtmlSourceContext(source_uri="https://example.com/docs/page.html")

    markdown = render_markdown(doc_analyze(payload, file_suffix="html", source_context=context)[0])

    assert ("Outside footnote." in markdown) is expected_note


def test_html_local_self_url_appends_referenced_note(tmp_path: Path) -> None:
    """验证本地 HTML 使用自身文件名 fragment 时同样保留正文选择外脚注。"""
    detail = "Useful main article text " * 20
    source = tmp_path / "page.html"
    source.write_text(
        f"""<html><body><main><article><h1>Title</h1><p>{detail}
        <a href="page.html#fn1">[1]</a></p></article></main>
        <footer><aside id="fn1" role="doc-footnote"><p>Outside footnote.</p></aside></footer>
        </body></html>""",
        encoding="utf-8",
    )

    markdown = parse(source).markdown()

    assert "Outside footnote." in markdown
    assert "file:" not in markdown


@pytest.mark.parametrize("href", ["javascript:alert(1)#fn1", "http://["])
def test_html_unsafe_or_malformed_note_urls_do_not_append_notes(href: str) -> None:
    """验证危险协议与畸形 URL 不参与同文档脚注关联，且不会中断正文解析。"""
    detail = "Useful main article text " * 20
    payload = f"""<html><body><main><article><h1>Title</h1><p>{detail}
      <a href="{href}">[1]</a></p></article></main>
      <footer><aside id="fn1" role="doc-footnote"><p>Outside footnote.</p></aside></footer>
      </body></html>""".encode()
    context = HtmlSourceContext(source_uri="https://example.com/page.html")

    markdown = render_markdown(doc_analyze(payload, file_suffix="html", source_context=context)[0])

    assert "Title" in markdown and "[1]" in markdown
    assert "Outside footnote." not in markdown
    assert href not in markdown


def test_html_formula_sources_are_normalized_without_duplicate_katex_text() -> None:
    """验证 MathML、公式生成器 wrapper 与 data-expr 按统一公式协议输出且不重复。"""
    payload = rb"""<html><body><h1>Math</h1><p>Inline
      <math><semantics><mi>x</mi><annotation encoding="application/x-tex">x+1</annotation></semantics></math>
      <span class="katex"><span class="katex-mathml"><math><semantics><mi>y</mi>
      <annotation encoding="application/x-tex">y^2</annotation></semantics></math></span>
      <span class="katex-html">duplicate visible</span></span>
      <span class="mathjax"><math><semantics><mi>q</mi>
      <annotation encoding="application/x-tex">q_4</annotation></semantics></math>
      <span>mathjax duplicate</span></span>
      <span data-expr="z_3">formula fallback</span></p>
      <div class="mineru-math mineru-math--block">\[w^4\]</div></body></html>"""

    middle = doc_analyze(payload, file_suffix="html")[0]
    markdown = render_markdown(middle)

    assert "x+1" in markdown and "y^2" in markdown and "q_4" in markdown and "z_3" in markdown and "w^4" in markdown
    assert any(block.type == BlockType.EQUATION and block.content == "w^4" for block in middle.pages[0].blocks)  # type: ignore[union-attr]
    assert "duplicate visible" not in markdown
    assert "mathjax duplicate" not in markdown
    assert "formula fallback" not in markdown


def test_html_generic_formula_class_wrappers_preserve_mixed_content() -> None:
    """验证通用公式 class 只规范化真实 carrier，不吞掉外层说明内容。"""
    payload = b"""<html><body><main>
      <div class="math"><p>Before explanation.</p><math><mi>x</mi></math><p>After explanation.</p></div>
      <div class="formula">Prefix <span><math data-tex="y"></math></span> suffix.</div>
      <div class="tex"><p>Script before.</p><script type="math/tex; mode=display">z</script>
      <p>Script after.</p></div>
      <p>Exclusive before<span class="math math-display"><span><math data-tex="u"></math></span></span>Exclusive after</p>
      <div class="math"><math data-tex="a"></math><math data-tex="b"></math></div></main></body></html>"""

    middle, model = doc_analyze(payload, file_suffix="html")
    raw = [(block["type"], block.get("content")) for block in model.pages[0]]

    assert raw == [
        (BlockType.TEXT, "Before explanation."),
        (BlockType.EQUATION, "x"),
        (BlockType.TEXT, "After explanation."),
        (BlockType.TEXT, "Prefix <eq>y</eq> suffix."),
        (BlockType.TEXT, "Script before."),
        (BlockType.EQUATION, "z"),
        (BlockType.TEXT, "Script after."),
        (BlockType.TEXT, "Exclusive before"),
        (BlockType.EQUATION, "u"),
        (BlockType.TEXT, "Exclusive after"),
        (BlockType.EQUATION, "a"),
        (BlockType.EQUATION, "b"),
    ]
    markdown = render_markdown(middle)
    assert markdown.index("Before explanation.") < markdown.index("$$\nx\n$$") < markdown.index("After explanation.")
    assert markdown.index("Script before.") < markdown.index("$$\nz\n$$") < markdown.index("Script after.")


def test_html_mineru_page_footnote_marker_roundtrips_as_page_footnote() -> None:
    """验证 MinerU HTML renderer 的轻量脚注 marker 可恢复统一 page_footnote block。"""
    payload = b"""<html><body><h1>Footnote</h1>
      <div class="mineru-page-footnote" data-block-type="page_footnote">Rendered footnote.</div>
    </body></html>"""

    middle = doc_analyze(payload, file_suffix="html")[0]

    footnote = next(block for block in middle.pages[0].blocks if block.type == BlockType.PAGE_FOOTNOTE)
    assert footnote.content == "Rendered footnote."  # type: ignore[union-attr]


def test_html_malformed_urls_degrade_without_aborting_document() -> None:
    """验证 urlsplit 无法解析的链接与图片只降级标签文本，不中断整份 HTML。"""
    payload = (
        b'<html><body><h1>URLs</h1><p><a href="http://[">Broken link</a></p>'
        b'<img src="http://[" alt="Broken image"></body></html>'
    )

    markdown = render_markdown(doc_analyze(payload, file_suffix="html")[0])

    assert "Broken link" in markdown and "Broken image" in markdown
    assert "http://[" not in markdown


def test_html_arbitrary_svg_data_image_degrades_to_alt_text() -> None:
    """验证来源 HTML 不能把可能含活动内容的任意 SVG data URI带入输出。"""
    svg = base64.b64encode(b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>').decode()
    payload = f'<html><body><h1>SVG</h1><img src="data:image/svg+xml;base64,{svg}" alt="Safe alt"></body></html>'.encode()

    middle = doc_analyze(payload, file_suffix="html")[0]

    assert not any(isinstance(block, ImageBlock) for block in middle.pages[0].blocks)
    assert "Safe alt" in render_markdown(middle)
    assert "alert(1)" not in middle.to_json()


def test_html_mineru_figure_keeps_real_caption_without_exposing_alt_as_caption() -> None:
    """验证 MinerU renderer 图片只恢复真实 caption，不重复显示用于无障碍的长 alt。"""
    payload = b"""<html><body><h1>Figure</h1><figure class="mineru-figure mineru-figure--image">
      <img src="https://example.com/image.png" alt="Long internal image description">
      <p class="mineru-caption">Visible figure caption</p></figure></body></html>"""

    markdown = render_markdown(doc_analyze(payload, file_suffix="html")[0])

    assert "Visible figure caption" in markdown
    assert "Long internal image description" not in markdown


def test_html_mineru_table_figure_rebinds_renderer_caption() -> None:
    """验证 MinerU table figure 的独立 caption 恢复为 table_caption 且只输出一次。"""
    payload = b"""<html><body><h1>Table figure</h1><figure class="mineru-figure mineru-figure--table">
      <table><tr><th>A</th></tr><tr><td>1</td></tr></table>
      <p class="mineru-caption">Visible table caption</p></figure></body></html>"""

    middle = doc_analyze(payload, file_suffix="html")[0]
    markdown = render_markdown(middle)

    table = next(block for block in middle.pages[0].blocks if block.type == BlockType.TABLE)
    assert any(child.type == BlockType.TABLE_CAPTION for child in table.content)  # type: ignore[union-attr]
    assert markdown.count("Visible table caption") == 1


def test_html_figure_preserves_direct_and_inline_text_around_visuals() -> None:
    """验证 figure 的直属文本、行内容器和 child tail 按 visual 前后顺序进入 raw blocks。"""
    payload = b"""<html><body><figure>Before<span>Inline</span>
      <a href="https://example.com/full"><img src="https://example.com/a.png"></a>After
      <figcaption>Cap</figcaption>Tail</figure></body></html>"""

    middle, model = doc_analyze(payload, file_suffix="html")
    raw_blocks = model.pages[0]
    markdown = render_markdown(middle)

    assert [block["type"] for block in raw_blocks] == [
        BlockType.TEXT,
        BlockType.IMAGE,
        BlockType.IMAGE_CAPTION,
        BlockType.TEXT,
    ]
    assert [block.get("content") for block in raw_blocks] == ["BeforeInline", "", "Cap", "After Tail"]
    assert [block.type for block in middle.pages[0].blocks] == [BlockType.TEXT, BlockType.IMAGE, BlockType.TEXT]
    assert markdown.index("BeforeInline") < markdown.index("Cap") < markdown.index("After Tail")


def test_html_colonized_office_svg_and_math_tags_preserve_visible_content() -> None:
    """验证 legacy HTML 冒号标签不会触发 QName 异常，并按本地名恢复正文、SVG 与公式。"""
    payload = b"""<html><body><o:p>Legacy Office text</o:p>
      <svg:svg><svg:text>Visible SVG text</svg:text></svg:svg>
      <m:math><m:mi>x</m:mi></m:math></body></html>"""

    middle, model = doc_analyze(payload, file_suffix="html")

    assert [(block["type"], block.get("content")) for block in model.pages[0]] == [
        (BlockType.TEXT, "Legacy Office text"),
        (BlockType.TEXT, "Visible SVG text"),
        (BlockType.EQUATION, "x"),
    ]
    markdown = render_markdown(middle)
    assert "Legacy Office text" in markdown and "Visible SVG text" in markdown
    assert "$$\nx\n$$" in markdown


def test_html_interleaved_figure_captions_bind_to_each_nearest_image() -> None:
    """验证同一 figure 的多张图片分别保留其相邻 caption，不会全部归到末图。"""
    payload = b"""<html><body><figure>
      <img src="https://example.com/a.png"><figcaption>Caption A</figcaption>
      <img src="https://example.com/b.png"><figcaption>Caption B</figcaption>
      </figure></body></html>"""

    middle, model = doc_analyze(payload, file_suffix="html")

    assert [(block["type"], block.get("content")) for block in model.pages[0]] == [
        (BlockType.IMAGE, ""),
        (BlockType.IMAGE_CAPTION, "Caption A"),
        (BlockType.IMAGE, ""),
        (BlockType.IMAGE_CAPTION, "Caption B"),
    ]
    images = [block for block in middle.pages[0].blocks if isinstance(block, ImageBlock)]
    assert len(images) == 2
    assert [child.content for child in images[0].content if child.type == BlockType.IMAGE_CAPTION] == ["Caption A"]
    assert [child.content for child in images[1].content if child.type == BlockType.IMAGE_CAPTION] == ["Caption B"]


@pytest.mark.parametrize(
    ("visual_markup", "parent_type", "body_type", "annotation_markup", "annotation_type"),
    [
        (
            '<img src="https://example.com/a.png">',
            BlockType.IMAGE,
            BlockType.IMAGE_BODY,
            "<figcaption><p>A</p><p>B</p></figcaption>",
            BlockType.IMAGE_CAPTION,
        ),
        (
            '<img src="https://example.com/a.png">',
            BlockType.IMAGE,
            BlockType.IMAGE_BODY,
            '<div class="footnote"><p>A</p><p>B</p></div>',
            BlockType.IMAGE_FOOTNOTE,
        ),
        (
            "<table><tr><td>X</td></tr></table>",
            BlockType.TABLE,
            BlockType.TABLE_BODY,
            "<figcaption><p>A</p><p>B</p></figcaption>",
            BlockType.TABLE_CAPTION,
        ),
        (
            "<table><tr><td>X</td></tr></table>",
            BlockType.TABLE,
            BlockType.TABLE_BODY,
            '<div class="footnote"><p>A</p><p>B</p></div>',
            BlockType.TABLE_FOOTNOTE,
        ),
    ],
)
def test_html_figure_block_children_keep_visual_annotation_relationship(
    visual_markup: str,
    parent_type: BlockType,
    body_type: BlockType,
    annotation_markup: str,
    annotation_type: BlockType,
) -> None:
    """验证 figure 中的块级说明文本仍按目标 visual 类型保留 caption/footnote 关系。"""
    payload = f"<html><body><figure>{visual_markup}{annotation_markup}</figure></body></html>".encode()

    middle, model = doc_analyze(payload, file_suffix="html")

    assert [block["type"] for block in model.pages[0]] == [parent_type, annotation_type, annotation_type]
    assert [block.get("content") for block in model.pages[0][1:]] == ["A", "B"]
    assert len(middle.pages[0].blocks) == 1
    visual = middle.pages[0].blocks[0]
    assert visual.type == parent_type
    assert [child.type for child in visual.content] == [body_type, annotation_type, annotation_type]  # type: ignore[union-attr]
    assert [child.content for child in visual.content[1:]] == ["A", "B"]  # type: ignore[union-attr]


def test_html_auto_selected_contextual_div_keeps_figure_caption_relation() -> None:
    """验证 auto 直接选中 visual wrapper div 时仍执行根节点 caption 关联。"""
    payload = b"""<html><body><div><figure><img src="https://example.com/a.png"></figure>
      <p class="caption">Div caption</p></div></body></html>"""

    middle, model = doc_analyze(payload, file_suffix="html")

    assert [(block["type"], block.get("content")) for block in model.pages[0]] == [
        (BlockType.IMAGE, ""),
        (BlockType.IMAGE_CAPTION, "Div caption"),
    ]
    image = next(block for block in middle.pages[0].blocks if isinstance(block, ImageBlock))
    assert [child.content for child in image.content if child.type == BlockType.IMAGE_CAPTION] == ["Div caption"]


def test_html_caption_does_not_cross_unrelated_parent_container() -> None:
    """验证 caption 与 visual 不共享明确语义父容器时不会跨容器猜测归属。"""
    detail = b"Useful main article text " * 20
    payload = (
        b'<html><body><main><div><figure><img src="https://example.com/a.png"></figure></div>'
        b'<p class="caption">Outside caption</p><p>' + detail + b"</p></main></body></html>"
    )

    middle = doc_analyze(payload, file_suffix="html")[0]
    image = next(block for block in middle.pages[0].blocks if isinstance(block, ImageBlock))

    assert all(child.type != BlockType.IMAGE_CAPTION for child in image.content)
    assert "Outside caption" in render_markdown(middle)


def test_html_alt_caption_fallback_policy_distinguishes_generic_and_mineru_figures() -> None:
    """验证普通图片可用 alt 兜底，但显式 caption 与 MinerU figure 不重复提升 alt。"""
    payload = b"""<html><body>
      <figure><img src="https://example.com/a.png" alt="Generic alt"></figure>
      <figure><img src="https://example.com/b.png" alt="Hidden alt"><figcaption>Explicit caption</figcaption></figure>
      <figure class="mineru-figure"><img src="https://example.com/c.png" alt="Accessibility alt"></figure>
      </body></html>"""

    middle = doc_analyze(payload, file_suffix="html")[0]
    images = [block for block in middle.pages[0].blocks if isinstance(block, ImageBlock)]
    captions = [[child.content for child in image.content if child.type == BlockType.IMAGE_CAPTION] for image in images]

    assert captions == [["Generic alt"], ["Explicit caption"], []]
    markdown = render_markdown(middle)
    assert "Generic alt" in markdown and "Explicit caption" in markdown
    assert "Hidden alt" not in markdown and "Accessibility alt" not in markdown


def test_html_inline_visual_splits_paragraph_text_in_dom_order() -> None:
    """验证段落内 visual 会切开前后文本，而不是把图片移到合并文本之后。"""
    payload = b'<html><body><p>Before<img src="https://example.com/a.png">After</p></body></html>'

    middle, model = doc_analyze(payload, file_suffix="html")

    assert [(block["type"], block.get("content")) for block in model.pages[0]] == [
        (BlockType.TEXT, "Before"),
        (BlockType.IMAGE, ""),
        (BlockType.TEXT, "After"),
    ]
    assert [block.type for block in middle.pages[0].blocks] == [BlockType.TEXT, BlockType.IMAGE, BlockType.TEXT]


def test_html_inline_visual_splits_ordered_list_without_renumbering_following_items() -> None:
    """验证列表项内 visual 提升为页面兄弟，并保持前后阅读顺序及后续有序编号。"""
    payload = b"""<html><body><ol start="3"><li>Before<img src="https://example.com/a.png">After</li>
      <li>Next</li></ol></body></html>"""

    middle, model = doc_analyze(payload, file_suffix="html")
    raw_blocks = model.pages[0]

    assert [block["type"] for block in raw_blocks] == [BlockType.LIST, BlockType.IMAGE, BlockType.TEXT, BlockType.LIST]
    assert raw_blocks[0]["start"] == 3 and raw_blocks[3]["start"] == 4
    assert raw_blocks[0]["content"] == [{"type": BlockType.TEXT, "content": "Before"}]
    assert raw_blocks[2]["content"] == "After"
    assert raw_blocks[3]["content"] == [{"type": BlockType.TEXT, "content": "Next"}]
    assert [block.type for block in middle.pages[0].blocks] == [
        BlockType.LIST,
        BlockType.IMAGE,
        BlockType.TEXT,
        BlockType.LIST,
    ]
    assert middle.pages[0].blocks[0].content[0].content == "3. Before"  # type: ignore[union-attr]
    assert middle.pages[0].blocks[3].content[0].content == "4. Next"  # type: ignore[union-attr]


def test_html_local_base_images_styles_and_escape_are_bounded(tmp_path: Path) -> None:
    """验证本地 base、CSS、栅格图可读取，但父目录逃逸图片只保留说明。"""
    assets = tmp_path / "assets"
    assets.mkdir()
    image_path = assets / "pixel.png"
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(image_path)
    (assets / "styles.css").write_text(".gone { display:none }", encoding="utf-8")
    outside = tmp_path.parent / "outside-html-image.png"
    Image.new("RGB", (1, 1), "blue").save(outside)
    source = tmp_path / "sample.htm"
    source.write_text(
        """<html><head><base href="assets/"><link rel="stylesheet" href="styles.css"></head><body>
        <h1>Local</h1><p class="gone">hidden css</p><img src="pixel.png" alt="Pixel">
        <img src="../outside-html-image.png" alt="Outside"></body></html>""",
        encoding="utf-8",
    )

    result = parse(source)
    async_result = asyncio.run(parse_async(source))

    assert result.middle_json.model_dump() == async_result.middle_json.model_dump()
    assert result.middle_json.file_suffix == "html"
    assert _image_body(result.middle_json).image_base64.startswith("data:image/png;base64,")
    markdown = result.markdown()
    assert "hidden css" not in markdown
    assert "Outside" in markdown
    assert outside.read_bytes() not in result.images().values()
    exported = result.middle_json.export(tmp_path / "export")
    assert len(exported.image_paths) == 1
    assert exported.image_paths[0].read_bytes() == image_path.read_bytes()
    assert _image_body(exported.middle_json).image_base64 is None
    assert _image_body(exported.middle_json).image_path is not None


def test_html_remote_source_resolves_relative_links_without_fetching_images() -> None:
    """验证 URL 来源只把相对链接与图片规范为绝对 URL，不下载远程图片。"""
    context = HtmlSourceContext(source_uri="https://example.com/news/page.html")
    payload = b'<html><body><h1 id="top">Remote</h1><p><a href="next.html">Next</a></p><img src="../img/a.png"></body></html>'

    middle, _ = doc_analyze(payload, file_suffix="html", source_context=context)
    markdown = render_markdown(middle)

    assert "https://example.com/news/next.html" in markdown
    assert _image_body(middle).image_url == "https://example.com/img/a.png"
    assert _image_body(middle).image_base64 is None


def test_html_local_image_symlink_cannot_escape_resource_root(tmp_path: Path) -> None:
    """验证本地图片 symlink resolve 到根目录外时只保留 alt 文本。"""
    outside = tmp_path.parent / "outside-symlink-image.png"
    Image.new("RGB", (1, 1), "black").save(outside)
    link = tmp_path / "linked.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink is unavailable on this platform")
    source = tmp_path / "sample.html"
    source.write_text('<html><body><h1>Link</h1><img src="linked.png" alt="Escaped"></body></html>', encoding="utf-8")

    middle = parse(source).middle_json

    assert not any(isinstance(block, ImageBlock) for block in middle.pages[0].blocks)
    assert "Escaped" in render_markdown(middle)


def test_html_model_empty_document_keeps_one_logical_page() -> None:
    """验证空 HTML 仍返回确定的一页，不制造伪 bbox 或标题。"""
    assert HtmlModel().predict(BytesIO(b"")) == [[]]
    middle, model = doc_analyze(b"", file_suffix="html")
    assert model.pages == [[]]
    assert len(middle.pages) == 1 and middle.pages[0].blocks == []


def test_html_parse_server_local_source_keeps_relative_assets(tmp_path: Path) -> None:
    """验证 parse-server 本地来源把原目录上下文传入 HTML 模型并输出严格结果。"""
    image_path = tmp_path / "pixel.png"
    Image.new("RGB", (2, 2), "green").save(image_path)
    source = tmp_path / "sample.html"
    source.write_text('<html><body><h1>API HTML</h1><img src="pixel.png" alt="Pixel"></body></html>', encoding="utf-8")
    file_store = FileStore(tmp_path / "api-files")
    request = CreateJobRequest.model_validate(
        {
            "files": [{"source": {"type": "local", "path": str(source)}}],
            "tier": "standard",
            "output_formats": ["markdown", "middle_json", "structured_content"],
        }
    )
    record = api_server.JobStore().create(request, file_store)

    asyncio.run(
        api_server._run_job(
            record,
            request,
            file_store,
            ocr_mode="auto",
            image_analysis=True,
            allow_local_source=True,
        )
    )

    parsed_file = record.files[0]
    assert parsed_file.status == "completed"
    assert parsed_file.output_files is not None and parsed_file.output_files.middle_json is not None
    middle_record = file_store.get_file(parsed_file.output_files.middle_json.file_id)
    assert middle_record.sha256sum is not None
    middle_payload = json.loads(file_store.read_blob(middle_record.sha256sum))
    assert middle_payload["file_suffix"] == "html"
    assert middle_payload["effort"] == "flash"
    image_body = middle_payload["pages"][0]["blocks"][1]["content"][0]
    assert image_body["image_base64"].startswith("data:image/png;base64,")


def test_html_doclib_local_bridge_uses_flash_parser(tmp_path: Path) -> None:
    """验证 Doclib 本地 Flash 桥接把 HTML 文件交给统一 MinerUParser。"""
    source = tmp_path / "doclib.html"
    source.write_text("<html><body><h1>Doclib HTML</h1><p>Body text.</p></body></html>", encoding="utf-8")
    service = object.__new__(ParseService)

    result = asyncio.run(
        service._parse_via_local(  # type: ignore[arg-type]
            {"path": str(source), "ext": "html"},
            "flash",
            "",
        )
    )

    assert result.middle_json.file_suffix == "html"
    assert result.middle_json.effort == "flash"
    assert "Doclib HTML" in result.markdown()


def test_html_rejects_page_range_and_resource_overflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 HTML 保持整本文档契约，并在输入预算超限时显式失败。"""
    source = tmp_path / "sample.html"
    source.write_text("<p>text</p>", encoding="utf-8")
    with pytest.raises(InvalidRequestError) as exc_info:
        parse(source, page_range="1")
    assert exc_info.value.code == "page_range_invalid"

    monkeypatch.setattr(html_converter_module, "MAX_HTML_BYTES", 4)
    with pytest.raises(HtmlResourceLimitError, match="max_html_bytes"):
        HtmlModel().predict(BytesIO(b"<p>x</p>"))


def test_html_comments_count_toward_dom_node_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 comment 等非元素 DOM 节点同样受总节点预算约束。"""
    monkeypatch.setattr(html_document_module, "MAX_HTML_NODES", 4)

    with pytest.raises(HtmlResourceLimitError, match="max_html_nodes"):
        doc_analyze(b"<html><body><!--1--><!--2--><!--3--></body></html>", file_suffix="html")


@pytest.mark.parametrize(
    ("styles", "single_limit", "total_limit", "expected_limit"),
    [
        ("<style>.large{display:none}</style>", 8, 128, "max_html_stylesheet_bytes"),
        (
            "<style>.a{display:none}</style><style>.b{display:none}</style>",
            16,
            24,
            "max_html_stylesheet_total_bytes",
        ),
    ],
)
def test_html_inline_stylesheets_enforce_resource_budgets(
    styles: str,
    single_limit: int,
    total_limit: int,
    expected_limit: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证内联 CSS 同时受单份和整文档 stylesheet 字节预算约束。"""
    monkeypatch.setattr(html_resources_module, "MAX_HTML_STYLESHEET_BYTES", single_limit)
    monkeypatch.setattr(html_resources_module, "MAX_HTML_STYLESHEET_TOTAL_BYTES", total_limit)
    payload = f"<html><head>{styles}</head><body><p>Visible</p></body></html>".encode()

    with pytest.raises(HtmlResourceLimitError, match=expected_limit):
        doc_analyze(payload, file_suffix="html")


def test_html_inline_and_local_stylesheets_share_total_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证内联与本地外链 CSS 按文档顺序共享累计 stylesheet 预算。"""
    (tmp_path / "styles.css").write_text(".a{display:none}", encoding="utf-8")
    monkeypatch.setattr(html_resources_module, "MAX_HTML_STYLESHEET_BYTES", 16)
    monkeypatch.setattr(html_resources_module, "MAX_HTML_STYLESHEET_TOTAL_BYTES", 24)
    payload = b"""<html><head><link rel="stylesheet" href="styles.css">
      <style>.b{display:none}</style></head><body><p>Visible</p></body></html>"""
    context = HtmlSourceContext(local_resource_root=tmp_path)

    with pytest.raises(HtmlResourceLimitError, match="max_html_stylesheet_total_bytes"):
        doc_analyze(payload, file_suffix="html", source_context=context)


@pytest.mark.parametrize(
    "image_url",
    [
        "javascript:alert(1)",
        "//example.com/a.png",
        "https://user:secret@example.com/a.png",
        "file:///tmp/a.png",
    ],
)
def test_html_remote_image_url_contract_rejects_unsafe_sources(image_url: str) -> None:
    """验证 image_url 公共字段只接受无凭据 HTTP(S) 绝对地址。"""
    with pytest.raises(ValueError):
        ImageBodyBlock(type=BlockType.IMAGE_BODY, content="", image_url=image_url)


def test_html_versioned_wire_roundtrips_empty_code_body() -> None:
    """验证空代码主体仍携带 wire marker，并在 HTML 往返后保留代码块元数据。"""
    source = MiddleJson.model_validate(
        {
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        {
                            "type": "code",
                            "index": 0,
                            "sub_type": "code",
                            "guess_lang": "python",
                            "content": [{"type": "code_body", "index": 0, "content": ""}],
                        }
                    ],
                }
            ],
            "is_full_document": True,
            "file_suffix": "html",
            "effort": "flash",
            "parse_mode": "txt",
            "mineru_version": "test",
        }
    )

    rendered = render_html(source, standalone=False)
    marker = BeautifulSoup(rendered, "html.parser").select_one('[data-block-type="code_body"]')
    roundtrip = doc_analyze(rendered.encode(), file_suffix="html")[0]
    code = next(block for block in roundtrip.pages[0].blocks if isinstance(block, CodeBlock))
    body = next(child for child in code.content if isinstance(child, CodeBodyBlock))

    assert marker is not None and marker.get("data-block-index") == "0"
    assert code.sub_type == BlockType.CODE and code.guess_lang == "python"
    assert body.content == ""


def test_html_versioned_wire_roundtrips_all_semantic_types() -> None:
    """验证新版 MinerU HTML 在 DEFAULT/FULL 中精确恢复公开类型和关键元数据。"""
    source = _wire_contract_middle()
    default_html = render_html(source, standalone=False)
    full_html = render_html(source, mode=RenderMode.FULL, standalone=False)
    default_root = BeautifulSoup(default_html, "html.parser").select_one(".mineru-document")
    full_root = BeautifulSoup(full_html, "html.parser").select_one(".mineru-document")

    assert default_root["data-mineru-html-version"] == "1"
    assert default_root["data-render-mode"] == "default"
    assert full_root["data-render-mode"] == "full"
    assert full_root.select_one('[data-block-type="chart_body"]') is not None
    assert full_root.select_one('[data-block-type="image_footnote"]') is not None
    assert full_root.select_one('[data-block-sub-type="algorithm"]') is not None
    assert "y^2\\tag{1}" in [element.get("data-mineru-latex") for element in full_root.select("[data-mineru-latex]")]

    default_middle = doc_analyze(default_html.encode(), file_suffix="html")[0]
    full_middle = doc_analyze(full_html.encode(), file_suffix="html")[0]
    auxiliary_types = {BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER, BlockType.ASIDE_TEXT}
    expected_default = [
        _semantic_block_signature(block) for block in source.pages[0].blocks if block.type not in auxiliary_types
    ]
    expected_full = [_semantic_block_signature(block) for block in source.pages[0].blocks]

    assert [_semantic_block_signature(block) for block in default_middle.pages[0].blocks] == expected_default
    assert [_semantic_block_signature(block) for block in full_middle.pages[0].blocks] == expected_full
    assert len(full_middle.pages) == 1 and full_middle.pages[0].page_idx == 0
    assert next(block for block in full_middle.pages[0].blocks if block.type == BlockType.EQUATION).content == "y^2\\tag{1}"  # type: ignore[union-attr]
    roundtrip_text = next(block for block in full_middle.pages[0].blocks if block.type == BlockType.TEXT).content  # type: ignore[union-attr]
    assert roundtrip_text == "Text & value < 3 <eq>x+1</eq> &lt;eq&gt;literal&lt;/eq&gt;"


def test_html_invalid_versioned_markers_fallback_without_partial_results() -> None:
    """验证未知版本和多类非法 marker 都整体回退，且可见正文不会重复。"""
    source = MiddleJson.model_validate(
        {
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        {"type": "text", "index": 0, "content": "WIRESENTINEL <eq>WIREFORMULA</eq>"},
                        {
                            "type": "image",
                            "index": 1,
                            "content": [
                                {
                                    "type": "image_body",
                                    "index": 1,
                                    "content": "Image body",
                                    "image_url": "https://example.com/wire.png",
                                },
                                {"type": "image_caption", "index": 2, "content": "Visible caption"},
                            ],
                        },
                    ],
                }
            ],
            "is_full_document": True,
            "file_suffix": "html",
            "effort": "flash",
            "parse_mode": "txt",
            "mineru_version": "test",
        }
    )
    base = render_html(source, standalone=False)
    variants: list[str] = []

    unknown = BeautifulSoup(base, "html.parser")
    unknown.select_one(".mineru-document")["data-mineru-html-version"] = "999"
    variants.append(str(unknown))

    illegal_type = BeautifulSoup(base, "html.parser")
    illegal_type.select_one(".mineru-block")["data-block-type"] = "not_a_block"
    variants.append(str(illegal_type))

    missing_body = BeautifulSoup(base, "html.parser")
    missing_body.select_one('[data-block-type="image_body"]').decompose()
    variants.append(str(missing_body))

    duplicate_body = BeautifulSoup(base, "html.parser")
    visual_body = duplicate_body.select_one('[data-block-type="image_body"]')
    visual_body.parent.append(copy(visual_body))
    variants.append(str(duplicate_body))

    parent_mismatch = BeautifulSoup(base, "html.parser")
    parent_mismatch.select_one('[data-block-type="image_caption"]')["data-block-type"] = "table_caption"
    variants.append(str(parent_mismatch))

    block_nested_formula = BeautifulSoup(base, "html.parser")
    block_nested_formula.select_one('[data-block-type="equation"][data-formula-display="inline"]')["data-formula-display"] = (
        "block"
    )
    variants.append(str(block_nested_formula))

    for variant in variants:
        markdown = render_markdown(doc_analyze(variant.encode(), file_suffix="html")[0])
        assert markdown.count("WIRESENTINEL") == 1
        assert markdown.count("WIREFORMULA") == 1
        assert markdown.count("Visible caption") == 1


def test_html_versioned_wire_visible_structural_text_falls_back_without_loss() -> None:
    """验证机器结构容器中新增的可见文本会整体回退，并保留编辑内容。"""
    source = MiddleJson.model_validate(
        {
            "pages": [{"page_idx": 0, "blocks": [{"type": "text", "index": 0, "content": "Original wire text"}]}],
            "is_full_document": True,
            "file_suffix": "html",
            "effort": "flash",
            "parse_mode": "txt",
            "mineru_version": "test",
        }
    )
    default_html = render_html(source, standalone=False)
    full_html = render_html(source, mode=RenderMode.FULL, standalone=False)
    variants: list[tuple[str, str]] = []

    root_text = BeautifulSoup(default_html, "html.parser")
    root_text.select_one(".mineru-document").insert(0, "ROOT EDIT ")
    variants.append((str(root_text), "ROOT EDIT"))

    section_text = BeautifulSoup(full_html, "html.parser")
    section_text.select_one(".mineru-page").insert(0, "SECTION EDIT ")
    variants.append((str(section_text), "SECTION EDIT"))

    wrapper_text = BeautifulSoup(default_html, "html.parser")
    wrapper_text.select_one(".mineru-block").insert(0, "WRAPPER EDIT ")
    variants.append((str(wrapper_text), "WRAPPER EDIT"))

    child_tail = BeautifulSoup(default_html, "html.parser")
    child_tail.select_one(".mineru-block > p").insert_after(" CHILD TAIL EDIT")
    variants.append((str(child_tail), "CHILD TAIL EDIT"))

    for variant, edited_text in variants:
        markdown = render_markdown(doc_analyze(variant.encode(), file_suffix="html")[0])
        assert "Original wire text" in markdown
        assert edited_text in markdown


def test_html_versioned_wire_markerless_block_child_falls_back_without_crash() -> None:
    """验证行内容器内新增的无 marker 块节点会事务式回退，而不是在物化阶段抛错。"""
    source = MiddleJson.model_validate(
        {
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        {
                            "type": "page_footnote",
                            "index": 0,
                            "anchor": "note",
                            "content": "Original note",
                        }
                    ],
                }
            ],
            "is_full_document": True,
            "file_suffix": "html",
            "effort": "flash",
            "parse_mode": "txt",
            "mineru_version": "test",
        }
    )
    soup = BeautifulSoup(render_html(source, standalone=False), "html.parser")
    paragraph = soup.new_tag("p")
    paragraph.string = "EXTRA NOTE"
    soup.select_one(".mineru-page-footnote").append(paragraph)

    markdown = render_markdown(doc_analyze(str(soup).encode(), file_suffix="html")[0])

    assert "Original note" in markdown
    assert "EXTRA NOTE" in markdown


def test_html_versioned_wire_edited_code_body_falls_back_without_loss() -> None:
    """验证普通代码 body 中新增可见节点会整体回退并保留代码与编辑内容。"""
    source = MiddleJson.model_validate(
        {
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        {
                            "type": "code",
                            "index": 0,
                            "sub_type": "code",
                            "guess_lang": "python",
                            "content": [{"type": "code_body", "index": 0, "content": "print(1)"}],
                        }
                    ],
                }
            ],
            "is_full_document": True,
            "file_suffix": "html",
            "effort": "flash",
            "parse_mode": "txt",
            "mineru_version": "test",
        }
    )
    soup = BeautifulSoup(render_html(source, standalone=False), "html.parser")
    paragraph = soup.new_tag("p")
    paragraph.string = "NEW NOTE"
    soup.select_one('[data-block-type="code_body"]').append(paragraph)

    markdown = render_markdown(doc_analyze(str(soup).encode(), file_suffix="html")[0])

    assert "print(1)" in markdown
    assert "NEW NOTE" in markdown


def test_html_versioned_wire_edited_table_body_falls_back_without_loss() -> None:
    """验证表格 body 中新增可见节点会整体回退并保留表格与编辑内容。"""
    source = MiddleJson.model_validate(
        {
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        {
                            "type": "table",
                            "index": 0,
                            "content": [
                                {
                                    "type": "table_body",
                                    "index": 0,
                                    "content": "<table><tr><td>A</td></tr></table>",
                                }
                            ],
                        }
                    ],
                }
            ],
            "is_full_document": True,
            "file_suffix": "html",
            "effort": "flash",
            "parse_mode": "txt",
            "mineru_version": "test",
        }
    )
    soup = BeautifulSoup(render_html(source, standalone=False), "html.parser")
    paragraph = soup.new_tag("p")
    paragraph.string = "NEW NOTE"
    soup.select_one('[data-block-type="table_body"]').append(paragraph)

    markdown = render_markdown(doc_analyze(str(soup).encode(), file_suffix="html")[0])

    assert "| A |" in markdown
    assert "NEW NOTE" in markdown


def test_html_marker_fallback_does_not_double_resolve_images(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证结构校验先于资源物化，非法 marker 回退只解析一次图片。"""
    source = MiddleJson.model_validate(
        {
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        {"type": "text", "index": 0, "content": "Fallback"},
                        {
                            "type": "image",
                            "index": 1,
                            "content": [
                                {
                                    "type": "image_body",
                                    "index": 1,
                                    "content": "",
                                    "image_url": "https://example.com/once.png",
                                }
                            ],
                        },
                    ],
                }
            ],
            "is_full_document": True,
            "file_suffix": "html",
            "effort": "flash",
            "parse_mode": "txt",
            "mineru_version": "test",
        }
    )
    soup = BeautifulSoup(render_html(source, standalone=False), "html.parser")
    soup.select_one('[data-block-type="text"]')["data-block-type"] = "invalid"
    original = HtmlResourceContext.resolve_image
    calls: list[str] = []

    def counted_resolve_image(self: HtmlResourceContext, image_source: str, *, alt: str = "") -> object:
        """记录图片解析次数后调用真实安全实现。"""
        calls.append(image_source)
        return original(self, image_source, alt=alt)

    monkeypatch.setattr(HtmlResourceContext, "resolve_image", counted_resolve_image)
    middle = doc_analyze(str(soup).encode(), file_suffix="html")[0]

    assert "Fallback" in render_markdown(middle)
    assert calls == ["https://example.com/once.png"]


def test_html_generic_div_soup_attaches_contextual_caption_and_footnote() -> None:
    """验证非标准 div visual 容器按完整 token 和父子上下文恢复 caption/footnote。"""
    payload = b"""<html><body><main><h1>Soup</h1>
      <div class="photo-card"><img src="https://example.com/a.png" alt="Alt only">
      <p id="caption">Context caption</p><div class="footnote">Context footnote</div></div>
      <div><img src="https://example.com/b.png"><p class="captionish">Not exact caption</p></div>
      </main></body></html>"""

    middle = doc_analyze(payload, file_suffix="html")[0]
    images = [block for block in middle.pages[0].blocks if isinstance(block, ImageBlock)]

    assert len(images) == 2
    assert [child.type for child in images[0].content] == [
        BlockType.IMAGE_BODY,
        BlockType.IMAGE_CAPTION,
        BlockType.IMAGE_FOOTNOTE,
    ]
    assert "Context caption" in render_markdown(middle)
    assert "Context footnote" in render_markdown(middle)
    assert all(child.type != BlockType.IMAGE_CAPTION for child in images[1].content)


def test_html_formula_priority_delimiters_and_supported_mathml_are_normalized() -> None:
    """验证所有受支持公式来源按统一优先级输出裸 LaTeX，并保留内部 tag。"""
    payload = rb"""<html><body><h1>Formula matrix</h1><p>
      <span class="formula"><span data-mineru-latex="\(producer\)" data-tex="data-low"><math alttext="alt-low">
      <annotation encoding="application/x-tex">annotation-low</annotation></math></span></span>
      <math data-tex="data-low"><semantics><mi>x</mi>
      <annotation encoding="application/x-tex">annotation-high</annotation></semantics></math>
      <span data-expr="$$z_3\tag{3}$$">fallback text</span>
      <math alttext="\[alt_value\]"><unknown>ignored</unknown></math>
      <span class="katex"><math><msup><mi>k</mi><mn>2</mn></msup></math><span>duplicate</span></span>
      <math><mfrac><mi>a</mi><mi>b</mi></mfrac></math></p>
      <script type="math/tex; mode=display">\[display_value\]</script></body></html>"""

    middle = doc_analyze(payload, file_suffix="html")[0]
    text = next(block for block in middle.pages[0].blocks if block.type == BlockType.TEXT).content  # type: ignore[union-attr]
    equations = [block.content for block in middle.pages[0].blocks if block.type == BlockType.EQUATION]  # type: ignore[union-attr]

    assert "<eq>producer</eq>" in text
    assert "<eq>annotation-high</eq>" in text
    assert "<eq>z_3\\tag{3}</eq>" in text
    assert "<eq>alt_value</eq>" in text
    assert "<eq>{k}^{2}</eq>" in text
    assert r"<eq>\frac{a}{b}</eq>" in text
    assert "data-low" not in text and "annotation-low" not in text and "duplicate" not in text
    assert equations == ["display_value"]


def test_html_block_formulas_nested_in_text_containers_preserve_dom_order() -> None:
    """验证文本容器内的 display 公式切成独立 Equation，并保留前后阅读顺序。"""
    payload = b"""<html><body><p>Before<script type="math/tex; mode=display">x</script>Between
      <span><math display="block" data-tex="y"></math></span>After</p>
      <ul><li>Item before<math display="block" data-tex="z"></math>Item after</li></ul></body></html>"""

    middle, model = doc_analyze(payload, file_suffix="html")
    raw = [(block["type"], block.get("content")) for block in model.pages[0]]

    assert raw[:5] == [
        (BlockType.TEXT, "Before"),
        (BlockType.EQUATION, "x"),
        (BlockType.TEXT, "Between"),
        (BlockType.EQUATION, "y"),
        (BlockType.TEXT, "After"),
    ]
    assert raw[5] == (BlockType.LIST, [{"type": BlockType.TEXT, "content": "Item before"}])
    assert raw[6:] == [(BlockType.EQUATION, "z"), (BlockType.TEXT, "Item after")]
    assert [block.type for block in middle.pages[0].blocks] == [
        BlockType.TEXT,
        BlockType.EQUATION,
        BlockType.TEXT,
        BlockType.EQUATION,
        BlockType.TEXT,
        BlockType.LIST,
        BlockType.EQUATION,
        BlockType.TEXT,
    ]


def test_html_invalid_mathml_and_asciimath_remain_visible_text() -> None:
    """验证未知 MathML 与本轮未支持 AsciiMath 不会伪装为 Equation 或被静默删除。"""
    payload = b"""<html><body><h1>Fallback math</h1>
      <math><unknown>not-latex</unknown></math>
      <script type="math/asciimath">sqrt(2)</script></body></html>"""

    middle = doc_analyze(payload, file_suffix="html")[0]
    markdown = render_markdown(middle)

    assert "not-latex" in markdown and "sqrt(2)" in markdown
    assert not any(block.type == BlockType.EQUATION for block in middle.pages[0].blocks)
