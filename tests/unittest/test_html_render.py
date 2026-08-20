from __future__ import annotations

from copy import deepcopy
from importlib import resources

from bs4 import BeautifulSoup
import pytest

from mineru.render import RenderMode, render_html
from mineru.types import (
    ChartAnnotationBlock,
    ChartBlock,
    ChartBodyBlock,
    CodeAnnotationBlock,
    CodeBlock,
    CodeBodyBlock,
    DocTitleBlock,
    EquationBlock,
    ImageBlock,
    IndexBlock,
    ListBlock,
    MiddleJson,
    PageAuxTextBlock,
    PageBlock,
    PageInfo,
    ParagraphTitleBlock,
    RefTextBlock,
    TableBlock,
    TableBodyBlock,
    TextBlock,
)

_PNG_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2l9sAAAAASUVORK5CYII="


def _middle(*pages: PageInfo) -> MiddleJson:
    """构造无需 PDF bbox 的严格 Office MiddleJson。"""
    return MiddleJson(
        pages=list(pages),
        file_suffix="docx",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )


def _page(page_idx: int, *blocks: PageBlock) -> PageInfo:
    """按调用方顺序构造测试页面。"""
    return PageInfo(page_idx=page_idx, blocks=list(blocks))


def _list(index: int, *items: str, sub_type: str | None = None) -> ListBlock:
    """构造普通或参考文献列表。"""
    child_class = RefTextBlock if sub_type == "ref_text" else TextBlock
    return ListBlock(
        type="list",
        index=index,
        sub_type=sub_type,
        content=[child_class(type=sub_type or "text", content=item) for item in items],
    )


def test_public_contract_fragment_standalone_title_and_input_immutability() -> None:
    """验证严格公共参数、双输出形态、标题回退和输入无副作用。"""
    middle = _middle(
        _page(
            0,
            DocTitleBlock(
                type="doc_title",
                index=0,
                level=1,
                content='<text style="bold">Demo</text> <unsafe>',
            ),
            TextBlock(type="text", index=1, content="body"),
        )
    )
    original = deepcopy(middle)

    fragment = render_html(middle, standalone=False)
    standalone = render_html(middle)

    assert fragment.startswith('<article class="mineru-document mineru-document--default">')
    assert fragment in standalone
    assert '<html lang="und">' in standalone
    assert "<title>Demo &lt;unsafe&gt;</title>" in standalone
    assert '<body class="mineru-html-body">' in standalone
    assert "<style>" in standalone
    assert "mathjax@" not in standalone
    assert "prismjs@" not in standalone
    assert middle == original

    explicit = render_html(middle, document_title="A </title><script>x</script>")
    assert "<title>A &lt;/title&gt;&lt;script&gt;x&lt;/script&gt;</title>" in explicit

    with pytest.raises(TypeError, match="MiddleJson"):
        render_html({})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RenderMode"):
        render_html(middle, mode="default")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="standalone"):
        render_html(middle, standalone=1)  # type: ignore[arg-type]


def test_default_and_full_modes_preserve_their_page_contracts() -> None:
    """验证 DEFAULT 连续阅读与 FULL 空页、辅助块和页面分隔。"""
    middle = _middle(
        _page(
            0,
            PageAuxTextBlock(type="header", index=0, content="HEADER"),
            TextBlock(type="text", index=1, content="inter-"),
        ),
        _page(5),
        _page(
            9,
            TextBlock(type="text", index=0, content="national", continues_prev=True),
            PageAuxTextBlock(type="footer", index=1, content="FOOTER"),
        ),
    )

    default = BeautifulSoup(render_html(middle, standalone=False), "html.parser")
    full = BeautifulSoup(render_html(middle, mode=RenderMode.FULL, standalone=False), "html.parser")

    assert default.select_one(".mineru-text").get_text() == "international"
    assert default.select_one('[data-block-type="text"]')["data-page-idx"] == "0"
    assert not default.select(".mineru-page")
    assert "HEADER" not in default.get_text()

    assert [section["data-page-idx"] for section in full.select(".mineru-page")] == ["0", "5", "9"]
    assert len(full.select(".mineru-page-break")) == 2
    assert [item.get_text() for item in full.select(".mineru-text")] == ["inter-", "national"]
    assert "HEADER" in full.get_text() and "FOOTER" in full.get_text()


def test_inline_html_escapes_plain_text_and_renders_styles_links_and_math() -> None:
    """验证普通尖括号、富样式、安全链接及 MathJax carrier。"""
    content = (
        'p <0.05 <local_dir> <text style="bold,italic,underline"> styled </text> '
        "<hyperlink>safe<url>https://example.test/a b</url></hyperlink> "
        "<hyperlink>bad<url>javascript:alert(1)</url></hyperlink> "
        "<eq>x < y & z</eq>"
    )
    result = render_html(_middle(_page(0, TextBlock(type="text", index=0, content=content))))
    soup = BeautifulSoup(result, "html.parser")

    paragraph = soup.select_one(".mineru-text")
    assert "p <0.05 <local_dir>" in paragraph.get_text()
    assert paragraph.select_one("em strong u") is not None
    assert paragraph.select_one(".mineru-preserve-whitespace") is not None
    assert paragraph.select_one('a[href="https://example.test/a%20b"]') is not None
    assert "bad" in paragraph.get_text()
    assert "javascript:" not in result
    assert paragraph.select_one(".mineru-math").get_text() == r"\(x < y & z\)"
    assert "mathjax@4.1.2/tex-chtml.js" in result
    assert "loader: {load: ['ui/safe']}" in result
    assert "ignoreHtmlClass: 'mineru-document'" in result
    assert "packages: {'[-]': ['require']}" in result


def test_formula_body_closing_delimiters_are_neutralized_before_mathjax_scanning() -> None:
    """验证公式体内的结束定界符改写为等价 TeX，不会提前闭合 carrier。"""
    middle = _middle(
        _page(
            0,
            TextBlock(type="text", index=0, content=r"inline <eq>x\)y</eq>"),
            EquationBlock(type="equation", index=1, content=r"x\]y"),
        )
    )
    soup = BeautifulSoup(render_html(middle, standalone=False), "html.parser")

    assert soup.select_one(".mineru-math--inline").get_text() == r"\(x\mathclose{)}y\)"
    assert soup.select_one(".mineru-math--block").get_text().replace("\n", "") == r"\[x\mathclose{]}y\]"


def test_lists_cover_native_explicit_reference_nested_and_orphan_shapes() -> None:
    """验证列表分类、非连续编号、显式 marker、参考文献和嵌套归属。"""
    ordered = _list(0, "1. first", "3. third")
    alpha = _list(1, "a. alpha", "b. beta")
    explicit = _list(2, "(1) one", "[x] done", "plain")
    mixed = _list(6, "1. one", "- two", "plain")
    reference = _list(3, "[1] first", "Author A", sub_type="ref_text")
    nested = ListBlock(
        type="list",
        index=4,
        content=[
            TextBlock(type="text", content="- parent"),
            ListBlock(type="list", content=[TextBlock(type="text", content="- child")]),
        ],
    )
    orphan = ListBlock(
        type="list",
        index=5,
        content=[ListBlock(type="list", content=[TextBlock(type="text", content="- orphan")])],
    )
    markerless_owner = ListBlock(
        type="list",
        index=7,
        content=[
            TextBlock(type="text", content="1. visible"),
            TextBlock(type="text", content=""),
            ListBlock(type="list", content=[TextBlock(type="text", content="- nested owner")]),
        ],
    )
    soup = BeautifulSoup(
        render_html(
            _middle(_page(0, ordered, alpha, explicit, reference, nested, orphan, mixed, markerless_owner)),
            standalone=False,
        ),
        "html.parser",
    )

    ordered_html = soup.select('[data-block-index="0"] ol')[0]
    assert ordered_html.find_all("li", recursive=False)[1]["value"] == "3"
    assert ordered_html.get_text(" ", strip=True) == "first third"
    assert soup.select_one('[data-block-index="1"] ol')["type"] == "a"
    assert [marker.get_text() for marker in soup.select('[data-block-index="2"] .mineru-list-marker')] == [
        "(1)",
        "[x]",
        "",
    ]
    assert "[1] first" in soup.select_one('[data-block-index="3"] li').get_text()
    parent_item = soup.select_one('[data-block-index="4"] > .mineru-list > li')
    assert parent_item.find("ul", recursive=False) is not None
    assert soup.select_one('[data-block-index="5"] .mineru-list-item--orphan') is not None
    assert [marker.get_text() for marker in soup.select('[data-block-index="6"] .mineru-list-marker')] == [
        "1.",
        "-",
        "",
    ]
    assert soup.select_one('[data-block-index="7"] .mineru-list-item--markerless > ul') is not None


def test_index_uses_real_forward_anchor_and_omits_duplicate_ids() -> None:
    """验证目录前向链接、页码尾清理、嵌套目录及重复 anchor 首项生效。"""
    index = IndexBlock(
        type="index",
        index=0,
        content=[
            ParagraphTitleBlock(type="paragraph_title", level=2, anchor="sec 1", content="Section\t3"),
            IndexBlock(
                type="index",
                content=[ParagraphTitleBlock(type="paragraph_title", level=3, anchor="missing", content="Missing\tiv")],
            ),
        ],
    )
    first = ParagraphTitleBlock(type="paragraph_title", index=1, level=2, anchor="sec 1", content="Section")
    duplicate = ParagraphTitleBlock(type="paragraph_title", index=2, level=9, anchor="sec 1", content="Duplicate")
    soup = BeautifulSoup(render_html(_middle(_page(0, index, first, duplicate)), standalone=False), "html.parser")

    assert soup.select_one('.mineru-index a[href="#sec-1"]').get_text() == "Section"
    assert "3" not in soup.select_one(".mineru-index").get_text()
    assert soup.select_one(".mineru-index li ul") is not None
    assert soup.select_one(".mineru-index li ul a") is None
    assert len(soup.select('[id="sec-1"]')) == 1
    assert soup.find("h6", attrs={"data-heading-level": "9"}).get_text() == "Duplicate"


def test_empty_title_does_not_create_a_broken_index_target_and_anchor_controls_are_normalized() -> None:
    """验证空正文标题不成为链接目标，控制字符在 id/href 两侧一致归一化。"""
    index = IndexBlock(
        type="index",
        index=0,
        content=[
            ParagraphTitleBlock(type="paragraph_title", level=2, anchor="empty", content="Empty"),
            ParagraphTitleBlock(type="paragraph_title", level=2, anchor="bad\x00id", content="Good"),
            ParagraphTitleBlock(type="paragraph_title", level=2, anchor="bad\ufffdid", content="Collision"),
        ],
    )
    empty = ParagraphTitleBlock(type="paragraph_title", index=1, level=2, anchor="empty", content="")
    good = ParagraphTitleBlock(type="paragraph_title", index=2, level=2, anchor="bad\x00id", content="Good")
    collision = ParagraphTitleBlock(
        type="paragraph_title",
        index=3,
        level=2,
        anchor="bad\ufffdid",
        content="Collision",
    )
    soup = BeautifulSoup(
        render_html(_middle(_page(0, index, empty, good, collision)), standalone=False),
        "html.parser",
    )

    links = soup.select(".mineru-index a")
    assert len(links) == 2
    assert links[0]["href"] == "#bad%EF%BF%BDid"
    assert links[1]["href"] == "#bad%EF%BF%BDid-2"
    assert soup.find(id="bad\ufffdid") is not None
    assert soup.find(id="bad\ufffdid-2") is not None


def test_empty_index_leaf_owns_its_following_nested_index() -> None:
    """验证空目录叶子的 nested ul 不会错误挂到更早的可见目录项。"""
    index = IndexBlock(
        type="index",
        index=0,
        content=[
            TextBlock(type="text", content="visible"),
            TextBlock(type="text", content=""),
            IndexBlock(type="index", content=[TextBlock(type="text", content="nested")]),
        ],
    )
    soup = BeautifulSoup(render_html(_middle(_page(0, index)), standalone=False), "html.parser")
    top_items = soup.select(".mineru-index > ul > li")

    assert len(top_items) == 2
    assert top_items[0].find("ul", recursive=False) is None
    assert top_items[1].find("ul", recursive=False).get_text(strip=True) == "nested"


def test_visual_child_order_image_details_and_asset_precedence() -> None:
    """验证视觉说明顺序、图片路径优先级、URL 编码与识别内容 details。"""
    image = ImageBlock.model_validate(
        {
            "type": "image",
            "index": 0,
            "sub_type": "diagram",
            "content": [
                {"type": "image_caption", "content": "before"},
                {
                    "type": "image_body",
                    "index": 0,
                    "content": "description <eq>x</eq>",
                    "image_path": "images/a b.png",
                    "image_base64": _PNG_URI,
                },
                {"type": "image_footnote", "content": "after"},
                {"type": "image_caption", "content": "again"},
            ],
        }
    )
    rendered = render_html(
        _middle(_page(0, image)),
        asset_base_url="https://cdn.example/doc",
        standalone=False,
    )
    soup = BeautifulSoup(rendered, "html.parser")

    assert rendered.index("before") < rendered.index("images/a%20b.png") < rendered.index("after") < rendered.index("again")
    assert soup.select_one("img")["src"] == "https://cdn.example/doc/images/a%20b.png"
    assert "data:image" not in rendered
    assert soup.select_one("details .mineru-math") is not None
    assert len(soup.select(".mineru-caption")) == 2


def test_table_keeps_safe_html_and_spatial_or_image_fallbacks() -> None:
    """验证原生表格、公式、危险属性清理、空间文本和不可用表格图片回退。"""
    html_table = TableBlock(
        type="table",
        index=0,
        content=[
            TableBodyBlock(
                type="table_body",
                index=0,
                content=(
                    '<table border="1"><tr><td rowspan="2" onclick="bad()">'
                    "<eq>x&lt;y</eq><table><tr><td>N</td></tr></table></td></tr></table>"
                ),
            )
        ],
    )
    spatial = TableBlock(
        type="table",
        index=1,
        content=[TableBodyBlock(type="table_body", index=1, content="A < B\nC   D")],
    )
    invalid = TableBlock(
        type="table",
        index=2,
        content=[TableBodyBlock(type="table_body", index=2, content="<table></table>", image_base64=_PNG_URI)],
    )
    soup = BeautifulSoup(render_html(_middle(_page(0, html_table, spatial, invalid)), standalone=False), "html.parser")

    assert len(soup.select('[data-block-index="0"] table')) == 2
    cell = soup.select_one('[data-block-index="0"] td')
    assert cell["rowspan"] == "2"
    assert "onclick" not in cell.attrs and "border" not in soup.select_one("table").attrs
    assert cell.select_one(".mineru-math") is not None
    assert soup.select_one('[data-block-index="1"] .mineru-table-text').get_text() == "A < B\nC   D"
    assert soup.select_one('[data-block-index="2"] img')["src"].startswith("data:image/png")


def test_discarded_invalid_table_math_does_not_load_mathjax() -> None:
    """验证无单元格表格回退图片后，已丢弃的 eq 不会误触发 MathJax。"""
    table = TableBlock(
        type="table",
        index=0,
        content=[
            TableBodyBlock(
                type="table_body",
                index=0,
                content="<table><eq>x</eq></table>",
                image_base64=_PNG_URI,
            )
        ],
    )
    rendered = render_html(_middle(_page(0, table)))

    assert BeautifulSoup(rendered, "html.parser").select_one("article .mineru-math") is None
    assert "mathjax@" not in rendered
    assert "data:image/png" in rendered


def test_chart_gfm_details_code_prism_and_algorithm_html() -> None:
    """验证严格 chart GFM、图片 details、Prism Autoloader 和算法转义。"""
    chart = ChartBlock(
        type="chart",
        index=0,
        sub_type="line chart",
        content=[
            ChartAnnotationBlock(type="chart_caption", content="chart before"),
            ChartBodyBlock(
                type="chart_body",
                index=0,
                image_path="images/chart.png",
                content="| A | B |\n| :--- | ---: |\n| 1 | 2 |",
            ),
        ],
    )
    code = CodeBlock(
        type="code",
        index=1,
        sub_type="code",
        guess_lang="shell",
        content=[
            CodeBodyBlock(type="code_body", index=1, content='echo "<x>"\n'),
            CodeAnnotationBlock(type="code_footnote", content="code after"),
        ],
    )
    algorithm = CodeBlock(
        type="code",
        index=2,
        sub_type="algorithm",
        content=[
            CodeBodyBlock(
                type="code_body",
                index=2,
                content="if a < b:\n  T<sub>q</sub> = <eq>x</eq><eq>y</eq>",
            )
        ],
    )
    result = render_html(_middle(_page(0, chart, code, algorithm)))
    soup = BeautifulSoup(result, "html.parser")

    assert soup.select_one("details .mineru-chart-table") is not None
    assert soup.select_one("th.mineru-align-left") is not None
    assert soup.select_one("th.mineru-align-right") is not None
    assert soup.select_one("pre.language-bash code.language-bash").get_text() == 'echo "<x>"\n'
    assert "prismjs@1.30.0/components/prism-core.min.js" in result
    assert "prismjs@1.30.0/plugins/autoloader/prism-autoloader.min.js" in result
    assert "Prism.highlightAllUnder(root)" in result
    assert "sha384-zLRFO4dw" in result and "sha384-Uq05+JLk" in result
    algorithm_html = soup.select_one(".mineru-algorithm")
    assert "if a < b:" in algorithm_html.get_text()
    assert algorithm_html.select_one("sub").get_text() == "q"
    assert len(algorithm_html.select(".mineru-math")) == 2
    assert r"\(x\) \(y\)" in algorithm_html.get_text()
    assert "<br" not in str(algorithm_html)
    assert "if a &lt; b:\n  " in str(algorithm_html)


@pytest.mark.parametrize("guess_lang", ["python bad/../../x", "constructor", "prototype", "__proto__"])
def test_invalid_code_language_remains_visible_without_loading_prism(guess_lang: str) -> None:
    """验证非法语言名不会进入 class 或 CDN 组件路径。"""
    code = CodeBlock(
        type="code",
        index=0,
        sub_type="code",
        guess_lang=guess_lang,
        content=[CodeBodyBlock(type="code_body", index=0, content="<script>x</script>")],
    )
    result = render_html(_middle(_page(0, code)))
    soup = BeautifulSoup(result, "html.parser")

    assert soup.select_one("code").get_text() == "<script>x</script>"
    assert not soup.select_one("code").get("class")
    assert "prismjs@" not in result
    assert "<script>x</script>" not in result


def test_empty_code_and_literal_class_text_do_not_load_external_runtimes() -> None:
    """验证空代码和普通文本中的 class 字样不会误触发 Prism 或 MathJax。"""
    middle = _middle(
        _page(
            0,
            TextBlock(type="text", index=0, content='class="mineru-math fake" class="language-python"'),
            CodeBlock(
                type="code",
                index=1,
                sub_type="code",
                guess_lang="python",
                content=[CodeBodyBlock(type="code_body", index=1, content="")],
            ),
        )
    )
    rendered = render_html(middle)

    assert "mathjax@" not in rendered
    assert "prismjs@" not in rendered
    assert '<pre class="mineru-code"><code></code></pre>' in rendered


def test_crlf_and_cr_are_normalized_to_visible_line_breaks() -> None:
    """验证 CRLF 与 CR 统一为两个可见 HTML 换行。"""
    rendered = render_html(
        _middle(_page(0, TextBlock(type="text", index=0, content="one\r\ntwo\rthree"))),
        standalone=False,
    )
    paragraph = BeautifulSoup(rendered, "html.parser").select_one(".mineru-text")

    assert len(paragraph.find_all("br")) == 2
    assert paragraph.get_text("|", strip=True) == "one|two|three"


def test_invalid_html_characters_are_replaced_and_output_remains_utf8_encodable() -> None:
    """验证普通文本、代码、anchor 和富 HTML 中的控制字符与 surrogate 可见降级。"""
    middle = _middle(
        _page(
            0,
            ParagraphTitleBlock(type="paragraph_title", index=0, level=2, anchor="a\ud800", content="T\x01"),
            TextBlock(type="text", index=1, content="body\udfff"),
            CodeBlock(
                type="code",
                index=2,
                sub_type="code",
                guess_lang="txt",
                content=[CodeBodyBlock(type="code_body", index=2, content="code\ud800\x02")],
            ),
            ChartBlock(
                type="chart",
                index=3,
                content=[ChartBodyBlock(type="chart_body", index=3, content="<p>chart\udfff\x03</p>")],
            ),
            TableBlock(
                type="table",
                index=4,
                content=[TableBodyBlock(type="table_body", index=4, content="table\ud800\x04")],
            ),
        )
    )
    rendered = render_html(middle)

    rendered.encode("utf-8")
    assert "\ud800" not in rendered and "\udfff" not in rendered
    assert rendered.count("\ufffd") >= 6


def test_setext_heading_shape_is_not_misclassified_as_a_gfm_chart_table() -> None:
    """验证缺少 pipe 的 Setext 标题形态按普通 chart 文本输出。"""
    chart = ChartBlock(
        type="chart",
        index=0,
        content=[ChartBodyBlock(type="chart_body", index=0, content="Title\n---\nbody")],
    )
    soup = BeautifulSoup(render_html(_middle(_page(0, chart)), standalone=False), "html.parser")

    assert soup.select_one(".mineru-chart-table") is None
    assert soup.select_one(".mineru-figure--chart").get_text(" ", strip=True) == "Title --- body"


@pytest.mark.parametrize(
    ("slash_count", "expected"),
    [(1, "|"), (2, "|"), (3, r"\|"), (4, r"\|"), (5, r"\\|")],
)
def test_chart_gfm_pipe_matches_markdown_it_backslash_decoding(slash_count: int, expected: str) -> None:
    """验证任意相邻反斜杠都保护列内 pipe，并与 markdown-it 解码结果一致。"""
    chart = ChartBlock(
        type="chart",
        index=0,
        content=[
            ChartBodyBlock(
                type="chart_body",
                index=0,
                content="| Formula |\n| --- |\n| " + "\\" * slash_count + "| |",
            )
        ],
    )
    soup = BeautifulSoup(render_html(_middle(_page(0, chart)), standalone=False), "html.parser")

    assert soup.select_one(".mineru-chart-table td").get_text() == expected


def test_mineru_styles_are_minified_scoped_and_inlined_byte_exact() -> None:
    """验证独立样式产物体积、作用域及 standalone 的逐字内联契约。"""
    root = resources.files("mineru").joinpath("resources", "html")
    source = root.joinpath("mineru.css").read_text(encoding="utf-8")
    minified = root.joinpath("mineru.min.css").read_text(encoding="utf-8")
    standalone = render_html(_middle())
    style = BeautifulSoup(standalone, "html.parser").style

    assert len(minified.encode("utf-8")) <= 10 * 1024
    assert "\n" not in minified and "/*" not in minified
    assert ".mineru-document" in source and ".mineru-html-body" in source
    assert style is not None and style.string == minified
    assert f"<style>{minified}</style>" in standalone
    assert not root.joinpath("crossnote").is_dir()


def test_empty_document_and_equation_image_do_not_load_external_scripts() -> None:
    """验证空文档和纯公式图片不会无谓加载 MathJax 或 Prism。"""
    empty = render_html(_middle())
    image_equation = render_html(_middle(_page(0, EquationBlock(type="equation", index=0, content="", image_base64=_PNG_URI))))

    assert '<article class="mineru-document mineru-document--default">\n\n</article>' in empty
    assert "mathjax@" not in empty and "prismjs@" not in empty
    assert "mathjax@" not in image_equation and "prismjs@" not in image_equation
    assert "data:image/png" in image_equation
