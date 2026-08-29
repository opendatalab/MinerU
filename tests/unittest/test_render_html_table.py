import markdown
from bs4 import BeautifulSoup

from mineru.config import LatexDelimitersConfig
from mineru.render._internal.markdown.table import format_embedded_html, render_html_table


DELIMITERS = LatexDelimitersConfig()


def test_render_html_table_uses_first_row_when_th_is_absent() -> None:
    """验证普通 td 首行可作为 GFM 表头。"""
    html = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"

    assert render_html_table(html, asset_base_url="", delimiters=DELIMITERS) == "\n".join(
        ["| A | B |", "| --- | --- |", "| 1 | 2 |"]
    )


def test_render_html_table_preserves_supported_inline_markup() -> None:
    """验证简单单元格保留链接、强调、代码、换行和上下标。"""
    html = (
        "<table><tr><th>Item</th><th>Note</th></tr>"
        "<tr><td><code>x|y</code></td>"
        '<td><a href="https://example.com">docs</a><br><strong>ready</strong><sup>2</sup></td></tr>'
        "</table>"
    )

    assert render_html_table(html, asset_base_url="", delimiters=DELIMITERS) == "\n".join(
        [
            "| Item | Note |",
            "| --- | --- |",
            r"| `x\|y` | [docs](https://example.com)<br>**ready**<sup>2</sup> |",
        ]
    )


def test_render_html_table_uses_markdown_only_for_simple_style_sets() -> None:
    """验证标准 HTML 标签按有效样式集合选择 Markdown 或完整 HTML wrapper。"""
    html = (
        "<table><tr><td><strong>bold</strong></td><td><em>italic</em></td>"
        "<td><s>strike</s></td><td><em><strong>both</strong></em></td>"
        "<td><strong><u>complex</u></strong></td>"
        "<td><s><strong>mixed</strong></s></td>"
        "<td><sup><strong>script</strong></sup></td></tr></table>"
    )

    assert render_html_table(html, asset_base_url="", delimiters=DELIMITERS) == "\n".join(
        [
            (
                "| **bold** | *italic* | ~~strike~~ | ***both*** | "
                "<strong><u>complex</u></strong> | <s><strong>mixed</strong></s> | "
                "<strong><sup>script</sup></strong> |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )


def test_render_html_table_escapes_angle_brackets_from_text_nodes() -> None:
    """验证实体解码后的文本尖括号不会重新变成可执行 Markdown HTML。"""
    html = (
        "<table><tr><th>Name</th><th>Value</th></tr>"
        "<tr><td>unsafe</td><td>&lt;script&gt;alert(1)&lt;/script&gt;</td></tr></table>"
    )

    rendered = render_html_table(html, asset_base_url="", delimiters=DELIMITERS)

    assert rendered == "\n".join(
        [
            "| Name | Value |",
            "| --- | --- |",
            "| unsafe | &lt;script>alert(1)&lt;/script> |",
        ]
    )
    parsed = BeautifulSoup(markdown.markdown(rendered, extensions=["tables"]), "html.parser")
    assert parsed.find("script") is None
    assert parsed.find_all("td")[1].get_text() == "<script>alert(1)</script>"


def test_render_html_table_escapes_formula_pipes_without_changing_latex() -> None:
    """验证 GFM 源码转义公式竖线，Markdown 解析后恢复原始 LaTeX。"""
    formulas = [
        r"\left|x\right|",
        r"\|x\|",
        r"\begin{array}{c|c}x\end{array}",
    ]
    rows = "".join(
        f"<tr><td>F{index}</td><td><eq>{formula}</eq></td><td>ok</td></tr>" for index, formula in enumerate(formulas, start=1)
    )
    html = f"<table><tr><th>Name</th><th>Formula</th><th>Note</th></tr>{rows}</table>"

    rendered = render_html_table(html, asset_base_url="", delimiters=DELIMITERS)
    assert rendered is not None
    parsed = BeautifulSoup(markdown.markdown(rendered, extensions=["tables"]), "html.parser")
    parsed_rows = parsed.find_all("tr")

    assert [len(row.find_all(["th", "td"], recursive=False)) for row in parsed_rows] == [3, 3, 3, 3]
    assert [row.find_all("td", recursive=False)[1].get_text() for row in parsed_rows[1:]] == [
        f"${formula}$" for formula in formulas
    ]


def test_render_html_table_falls_back_for_span_attribute_even_when_value_is_one() -> None:
    """验证只要显式出现 rowspan/colspan 就按复杂 HTML 输出。"""
    html = '<table><tr><td colspan="1">A</td></tr></table>'

    assert render_html_table(html, asset_base_url="", delimiters=DELIMITERS) == html


def test_format_embedded_html_rewrites_relative_images_and_equations() -> None:
    """验证复杂 HTML 的相对图片和行内公式统一改写。"""
    html = '<table><tr><td><img src="images/a.png"><eq>x&lt;y</eq></td></tr></table>'

    formatted = format_embedded_html(html, asset_base_url="https://cdn.example/doc", delimiters=DELIMITERS)

    assert 'src="https://cdn.example/doc/images/a.png"' in formatted
    assert "$x<y$" in formatted


def test_format_embedded_html_keeps_absolute_and_data_images() -> None:
    """验证已经可访问的绝对图片来源不会重复添加 base URL。"""
    html = '<table><tr><td><img src="https://example.com/a.png"><img src="data:image/png;base64,AAAA"></td></tr></table>'

    formatted = format_embedded_html(html, asset_base_url="https://cdn.example/doc", delimiters=DELIMITERS)

    assert "https://cdn.example/doc/https://" not in formatted
    assert 'src="data:image/png;base64,AAAA"' in formatted
