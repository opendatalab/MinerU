"""目录预览的书签、公式标签和注入边界回归。"""

from bs4 import BeautifulSoup

from mineru.kit.gradio.artifacts import _prepare_preview_links


def test_outline_preserves_bookmarks_and_prepares_unique_plain_labels() -> None:
    """真实 HTML 准备入口保留书签，避让全页 ID，并在 MathJax 执行前固定纯文本标签。"""
    document = """<html><head></head><body>
    <div id="mineru-outline-1"></div><div id="mineru-outline-1-toc"></div>
    <h1>页面控件标题</h1><article class="docvortex-document">
    <h2 class="docvortex-heading">中文 <strong>章节</strong></h2>
    <h4 class="docvortex-heading">中文 章节</h4>
    <h2 class="docvortex-heading" id="已有书签">公式 <span data-docvortex-latex="x^2">\\(x^2\\)</span></h2>
    <h3 class="docvortex-heading">  </h3>
    <a href="#已有书签">跳转</a><a href="https://example.test">外链</a>
    </article></body></html>"""
    soup = BeautifulSoup(_prepare_preview_links(document, include_outline=True), "html.parser")
    headings = soup.select("[data-mineru-outline-label]")
    assert [node["id"] for node in headings] == ["mineru-outline-1-toc-toc", "mineru-outline-2", "已有书签"]
    assert [node["data-mineru-outline-label"] for node in headings] == ["中文 章节", "中文 章节", "公式 x^2"]
    assert soup.find("a", string="跳转")["href"] == "#已有书签"
    assert soup.find("a", string="外链")["target"] == "_blank"
    assert len(soup.select("script[data-mineru-outline]")) == 1
    assert len(soup.select("style[data-mineru-outline]")) == 1
    assert not soup.h1.has_attr("id")
    assert not soup.h3.has_attr("id")
    assert "data-mineru-outline" not in _prepare_preview_links(document)


def test_empty_outline_has_no_controls_or_resources() -> None:
    """无标题或空标题的结果继续提供完整正文，且不注入目录界面。"""
    for body in ("<p>正文</p>", '<h2 class="docvortex-heading"> </h2>'):
        document = f'<html><head></head><body><article class="docvortex-document">{body}</article></body></html>'
        soup = BeautifulSoup(_prepare_preview_links(document, include_outline=True), "html.parser")
        assert not soup.select("[data-mineru-outline], [data-mineru-outline-label]")
        assert soup.article is not None
