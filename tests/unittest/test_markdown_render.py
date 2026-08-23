from __future__ import annotations

from copy import deepcopy
from typing import Literal

import pytest

from mineru.config import Config
from mineru.render import MarkdownRenderMode, render_markdown
from mineru.types import (
    ChartBlock,
    ChartBodyBlock,
    CodeBlock,
    CodeBodyBlock,
    EquationBlock,
    ImageBlock,
    ImageBodyBlock,
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


def _middle(*pages: PageInfo, file_suffix: str = "docx") -> MiddleJson:
    """构造最小严格 MiddleJson 测试对象。"""
    return MiddleJson(
        pages=list(pages),
        is_full_document=True,
        file_suffix=file_suffix,
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )


def _page(page_idx: int, *blocks: PageBlock) -> PageInfo:
    """构造一页并保留调用方给定的 block 顺序。"""
    return PageInfo(page_idx=page_idx, blocks=list(blocks))


def _image(index: int, path: str = "images/a.png") -> ImageBlock:
    """构造带 path 的图片父块。"""
    return ImageBlock(
        type="image",
        index=index,
        content=[ImageBodyBlock(type="image_body", index=index, content="", image_path=path)],
    )


def _table(index: int, content: str, *, continues_prev: bool | None = None) -> TableBlock:
    """构造无 bbox 的 Office 表格父块。"""
    return TableBlock(
        type="table",
        index=index,
        continues_prev=continues_prev,
        content=[TableBodyBlock(type="table_body", index=index, content=content)],
    )


def _pdf_table(
    index: int,
    content: str,
    *,
    continues_prev: bool | None = None,
) -> TableBlock:
    """构造带归一化 bbox 的 PDF 表格父块。"""
    bbox = (0.1, 0.1, 0.9, 0.9)
    return TableBlock(
        type="table",
        index=index,
        bbox=bbox,
        continues_prev=continues_prev,
        content=[TableBodyBlock(type="table_body", index=index, bbox=bbox, content=content)],
    )


def _list(
    index: int,
    *items: str,
    sub_type: Literal["text", "ref_text"] | None = None,
    continues_prev: bool | None = None,
) -> ListBlock:
    """构造带归一化 bbox 的列表父块，并按子类型生成文本叶子。"""
    child_type = sub_type or "text"
    child_class = RefTextBlock if child_type == "ref_text" else TextBlock
    return ListBlock(
        type="list",
        index=index,
        bbox=(0.1, 0.1, 0.9, 0.3),
        sub_type=sub_type,
        continues_prev=continues_prev,
        content=[child_class(type=child_type, content=item) for item in items],
    )


def _ref_text(index: int, content: str, *, continues_prev: bool | None = None) -> RefTextBlock:
    """构造可携带续段标记的顶层参考文献文本块。"""
    return RefTextBlock(
        type="ref_text",
        index=index,
        content=content,
        continues_prev=continues_prev,
    )


def test_render_modes_filter_merge_and_preserve_input() -> None:
    """验证两种模式的过滤、页内/跨页合并、分页线及无副作用。"""
    middle = _middle(
        _page(
            0,
            PageAuxTextBlock(type="header", index=0, content="HEADER"),
            TextBlock(type="text", index=1, content="Hello"),
            _image(2),
            TextBlock(type="text", index=3, content="world", continues_prev=True),
            PageAuxTextBlock(type="footer", index=4, content="FOOTER"),
        ),
        _page(
            1,
            PageAuxTextBlock(type="page_number", index=0, content="2"),
            TextBlock(type="text", index=1, content="again", continues_prev=True),
            PageAuxTextBlock(type="page_footnote", index=2, content="NOTE"),
        ),
    )
    before = middle.to_json(skip_defaults=False)

    default = render_markdown(middle)
    full = render_markdown(middle, mode=MarkdownRenderMode.FULL)

    assert default == "Hello world again\n\n![](images/a.png)"
    assert full == "\n\n---\n\n".join(
        [
            "HEADER\n\nHello world\n\n![](images/a.png)\n\nFOOTER",
            "2\n\nagain\n\nNOTE",
        ]
    )
    assert middle.to_json(skip_defaults=False) == before


def test_full_mode_preserves_empty_page_boundaries() -> None:
    """验证 FULL 对空白页仍保留相邻页分割线。"""
    middle = _middle(_page(0), _page(1, TextBlock(type="text", index=0, content="x")), _page(2))

    assert render_markdown(middle, mode=MarkdownRenderMode.FULL) == "\n\n---\n\nx\n\n---\n\n"
    assert render_markdown(middle) == "x"


def test_text_continuation_handles_hyphen_and_cjk_boundaries() -> None:
    """验证续写文本沿用西文断词与 CJK 直接连接规则。"""
    western = _middle(
        _page(
            0,
            TextBlock(type="text", index=0, content="inter-"),
            TextBlock(type="text", index=1, content="national", continues_prev=True),
        )
    )
    cjk = _middle(
        _page(
            0,
            TextBlock(type="text", index=0, content="中文"),
            TextBlock(type="text", index=1, content="继续", continues_prev=True),
        )
    )

    assert render_markdown(western) == "international"
    assert render_markdown(cjk) == "中文继续"


def test_ref_text_continuation_skips_page_auxiliary_blocks_by_mode() -> None:
    """验证 ref_text 默认跨页合并，FULL 只合并页内链且不修改输入。"""
    middle = _middle(
        _page(
            0,
            _ref_text(0, "inter-"),
            PageAuxTextBlock(type="page_footnote", index=1, content="NOTE"),
        ),
        _page(
            1,
            PageAuxTextBlock(type="header", index=0, content="HEADER"),
            _ref_text(1, "national", continues_prev=True),
            _ref_text(2, "continuation", continues_prev=True),
        ),
    )
    original = deepcopy(middle)

    assert render_markdown(middle) == "international continuation"
    assert render_markdown(middle, mode=MarkdownRenderMode.FULL) == (
        "inter-\n\nNOTE\n\n---\n\nHEADER\n\nnational continuation"
    )
    assert middle == original


def test_ref_text_continuation_keeps_semantic_barrier() -> None:
    """验证手工标记也不能让 ref_text 跨过正文等语义块。"""
    middle = _middle(
        _page(
            0,
            _ref_text(0, "first"),
            TextBlock(type="text", index=1, content="separator"),
            _ref_text(2, "second", continues_prev=True),
        )
    )

    assert render_markdown(middle) == "first\n\nseparator\n\nsecond"


def test_ref_text_continuation_reuses_url_boundary_joining() -> None:
    """验证 ref_text 续段复用正文的跨块 URL 连接规则。"""
    middle = _middle(
        _page(
            0,
            _ref_text(0, "See https://doi.o"),
            _ref_text(1, "rg/10.1016/example", continues_prev=True),
        )
    )

    assert render_markdown(middle) == "See https://doi.org/10.1016/example"


def test_list_continuation_merges_same_page_in_both_modes_without_mutating_input() -> None:
    """验证同页同子类型列表在两种模式下拼接，且不污染原始 MiddleJson。"""
    middle = _middle(
        _page(
            0,
            _list(0, "- first"),
            _list(1, "- second", continues_prev=True),
        ),
        file_suffix="pdf",
    )
    original = deepcopy(middle)

    assert render_markdown(middle) == "- first\n- second"
    assert render_markdown(middle, mode=MarkdownRenderMode.FULL) == "- first\n- second"
    assert middle == original


def test_list_continuation_merges_cross_page_chain_only_in_default_mode() -> None:
    """验证跨页列表链仅在默认模式整体拼接，完整模式保留页界并合并页内续段。"""
    middle = _middle(
        _page(0, _list(0, "[1] first", sub_type="ref_text")),
        _page(
            1,
            _list(0, "[2] second", sub_type="ref_text", continues_prev=True),
            _list(1, "[3] third", sub_type="ref_text", continues_prev=True),
        ),
        file_suffix="pdf",
    )

    assert render_markdown(middle) == "[1] first\n[2] second\n[3] third"
    assert render_markdown(middle, mode=MarkdownRenderMode.FULL) == (
        "[1] first\n\n---\n\n[2] second\n[3] third"
    )


def test_ref_list_continuation_skips_page_auxiliary_blocks_without_mutating_input() -> None:
    """验证默认模式跨页面辅助块合并参考文献，完整模式仍保留原始页界和顺序。"""
    middle = _middle(
        _page(
            0,
            _list(0, "[1] first", sub_type="ref_text"),
            PageAuxTextBlock(type="page_footnote", index=1, content="NOTE"),
        ),
        _page(
            1,
            PageAuxTextBlock(type="header", index=0, content="HEADER"),
            PageAuxTextBlock(type="page_number", index=1, content="2"),
            _list(2, "[2] second", sub_type="ref_text", continues_prev=True),
        ),
    )
    original = deepcopy(middle)

    assert render_markdown(middle) == "[1] first\n[2] second"
    assert render_markdown(middle, mode=MarkdownRenderMode.FULL) == (
        "[1] first\n\nNOTE\n\n---\n\nHEADER\n\n2\n\n[2] second"
    )
    assert middle == original


def test_ordinary_list_continuation_does_not_skip_page_auxiliary_blocks() -> None:
    """验证页面辅助块透明规则只作用于参考文献，普通列表仍要求物理相邻。"""
    middle = _middle(
        _page(
            0,
            _list(0, "- first"),
            PageAuxTextBlock(type="page_footnote", index=1, content="NOTE"),
        ),
        _page(
            1,
            PageAuxTextBlock(type="header", index=0, content="HEADER"),
            _list(1, "- second", continues_prev=True),
        ),
    )

    assert render_markdown(middle) == "- first\n\n- second"


def test_list_continuation_keeps_semantic_barrier_and_matching_subtype() -> None:
    """验证语义块仍会阻断列表续接，且子类型不一致时保持独立输出。"""
    non_adjacent = _middle(
        _page(
            0,
            _list(0, "[1] first", sub_type="ref_text"),
            TextBlock(type="text", index=1, bbox=(0.1, 0.4, 0.9, 0.5), content="separator"),
            _list(2, "[2] second", sub_type="ref_text", continues_prev=True),
        ),
        file_suffix="pdf",
    )
    mismatched = _middle(
        _page(
            0,
            _list(0, "[1] first", sub_type="ref_text"),
            _list(1, "- second", sub_type="text", continues_prev=True),
        ),
        file_suffix="pdf",
    )

    assert render_markdown(non_adjacent) == "[1] first\n\nseparator\n\n[2] second"
    assert render_markdown(mismatched) == "[1] first\n\n- second"


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ("[1] bracket", "[1] bracket"),
        ("1. dot", "1. dot"),
        ("(1) parenthesized", "(1) parenthesized"),
        ("1) closing parenthesis", "1) closing parenthesis"),
        ("1、cjk delimiter", "1、cjk delimiter"),
        ("［１］full width", "［１］full width"),
        ('<text style="bold">[1]</text> styled', "**[1]** styled"),
    ],
)
def test_reference_list_keeps_supported_numeric_prefix_styles(item: str, expected: str) -> None:
    """验证数字出现在前五个可见字符内时，单条参考文献保留原有编号。"""
    middle = _middle(_page(0, _list(0, item, sub_type="ref_text")), file_suffix="pdf")

    assert render_markdown(middle) == expected


def test_reference_list_uses_strict_numeric_prefix_majority() -> None:
    """验证参考文献按全部直属非空条目的严格多数决定是否补无序标记。"""
    numbered_majority = _middle(
        _page(0, _list(0, "[1] first", "missing marker", "3) third", sub_type="ref_text")),
        file_suffix="pdf",
    )
    unordered_majority = _middle(
        _page(0, _list(0, "1470–1480 continuation", "Author A", "Author B", sub_type="ref_text")),
        file_suffix="pdf",
    )
    tied = _middle(
        _page(0, _list(0, "[1] first", "Author A", sub_type="ref_text")),
        file_suffix="pdf",
    )

    assert render_markdown(numbered_majority) == "[1] first\nmissing marker\n3) third"
    expected_unordered = "- 1470–1480 continuation\n- Author A\n- Author B"
    assert render_markdown(unordered_majority) == expected_unordered
    assert render_markdown(unordered_majority, mode=MarkdownRenderMode.FULL) == expected_unordered
    assert render_markdown(tied) == "- [1] first\n- Author A"


def test_reference_list_bullets_mixed_children_without_duplication() -> None:
    """验证混合直属类型共同参与统计，已有短横线不重复且多行正文正确缩进。"""
    block = ListBlock(
        type="list",
        index=0,
        bbox=(0.1, 0.1, 0.9, 0.3),
        sub_type="ref_text",
        content=[
            TextBlock(type="text", content="- existing"),
            RefTextBlock(type="ref_text", content="Author A\ncontinued"),
        ],
    )

    assert render_markdown(_middle(_page(0, block), file_suffix="pdf")) == (
        "- existing\n- Author A\n  continued"
    )


def test_nested_reference_list_decides_bullets_independently() -> None:
    """验证嵌套参考文献依据自身子类型判定，外层普通列表行为保持不变。"""
    nested = ListBlock(
        type="list",
        sub_type="ref_text",
        content=[RefTextBlock(type="ref_text", content="Author A")],
    )
    outer = ListBlock(
        type="list",
        index=0,
        bbox=(0.1, 0.1, 0.9, 0.3),
        sub_type="text",
        content=[TextBlock(type="text", content="- outer"), nested],
    )

    assert render_markdown(_middle(_page(0, outer), file_suffix="pdf")) == "- outer\n    - Author A"


def test_text_continuation_joins_url_candidates_but_separates_independent_urls() -> None:
    """验证 Markdown 续写连接跨块 URL，同时保留两条独立 URL 的分隔。"""
    continued_url = _middle(
        _page(
            0,
            TextBlock(type="text", index=0, content="See https://doi.o"),
            TextBlock(type="text", index=1, content="rg/10.1016/example", continues_prev=True),
        )
    )
    independent_urls = _middle(
        _page(
            0,
            TextBlock(type="text", index=0, content="https://example.test/first"),
            TextBlock(type="text", index=1, content="https://example.test/second", continues_prev=True),
        )
    )

    assert render_markdown(continued_url) == "See https://doi.org/10.1016/example"
    assert render_markdown(independent_urls) == "https://example.test/first https://example.test/second"


def test_text_continuation_does_not_rewrite_formula_or_style_wrappers() -> None:
    """验证续写边界不会越过末尾公式或破坏样式节点。"""
    formula = _middle(
        _page(
            0,
            TextBlock(type="text", index=0, content="before <eq>x</eq>"),
            TextBlock(type="text", index=1, content="after", continues_prev=True),
        )
    )
    styled = _middle(
        _page(
            0,
            TextBlock(type="text", index=0, content='<text style="bold">inter-</text>'),
            TextBlock(type="text", index=1, content="national", continues_prev=True),
        )
    )

    assert render_markdown(formula) == "before $x$ after"
    assert render_markdown(styled) == "**inter**national"


def test_render_rejects_legacy_inputs_and_string_mode() -> None:
    """验证公共入口不兼容旧 dict/pages 输入或字符串模式。"""
    middle = _middle(_page(0))

    with pytest.raises(TypeError, match="MiddleJson"):
        render_markdown(middle.to_dict())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MarkdownRenderMode"):
        render_markdown(middle, mode="full")  # type: ignore[arg-type]


def test_inline_rich_text_unknown_tags_and_visible_spaces() -> None:
    """验证富文本、公式、链接、未知标签与可见空白。"""
    content = (
        'A <text style="bold">B</text> '
        '<hyperlink><text style="underline">link</text><url>https://example.com/a b</url></hyperlink> '
        '<eq>x_1</eq> <local_dir> p <0.05 and x > 0 <sup>2</sup><text style="underline">  </text>'
    )
    middle = _middle(_page(0, TextBlock(type="text", index=0, content=content)))

    rendered = render_markdown(middle)

    assert "**B**" in rendered
    assert '<a href="https://example.com/a b"><u>link</u></a>' in rendered
    assert "$x_1$" in rendered
    assert "<local_dir>" in rendered
    assert "p <0.05 and x > 0" in rendered
    assert "<sup>2</sup>" in rendered
    assert rendered.endswith("<sup>2</sup>__")


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("underline", "A___B"),
        ("strikethrough", "A---B"),
        ("underline,strikethrough", "A<s>___</s>B"),
        ("bold,underline", "A<strong>___</strong>B"),
    ],
)
def test_visible_style_ascii_spaces_use_dev_markers(style: str, expected: str) -> None:
    """验证纯 ASCII 样式空格使用 dev 的下划线或短横线 marker。"""
    content = f'A<text style="{style}">   </text>B'

    assert render_markdown(_middle(_page(0, TextBlock(type="text", index=0, content=content)))) == expected


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("underline", "<u>___广东__</u>"),
        ("strikethrough", "~~---广东--~~"),
        ("underline,strikethrough", "<s><u>___广东__</u></s>"),
    ],
)
def test_visible_style_edge_spaces_use_dev_markers(style: str, expected: str) -> None:
    """验证非空样式文本只替换首尾 ASCII 空格。"""
    content = f'<text style="{style}">   广东  </text>'

    assert render_markdown(_middle(_page(0, TextBlock(type="text", index=0, content=content)))) == expected


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("underline", r"\___"),
        ("strikethrough", r"\---"),
    ],
)
def test_standalone_visible_space_markers_are_escaped(style: str, expected: str) -> None:
    """验证整块 marker 会转义首字符，避免被当作 Markdown 分割线。"""
    content = f'<text style="{style}">   </text>'

    assert render_markdown(_middle(_page(0, TextBlock(type="text", index=0, content=content)))) == expected


def test_emphasis_only_spaces_keep_existing_html_behavior() -> None:
    """验证 emphasis-only 空格不进入 underline/strikethrough marker 规则。"""
    content = '<text style="emphasis">  </text>'

    rendered = render_markdown(_middle(_page(0, TextBlock(type="text", index=0, content=content))))

    assert "&nbsp;&nbsp;" in rendered


def test_text_block_escapes_markdown_prefix_and_malformed_tag() -> None:
    """验证普通 text 不会误变列表，损坏白名单标签按原文转义。"""
    middle = _middle(
        _page(
            0,
            TextBlock(type="text", index=0, content="- plain"),
            TextBlock(type="text", index=1, content='<text style="bold">broken'),
        )
    )

    assert render_markdown(middle) == '\\- plain\n\n<text style="bold">broken'


def test_title_and_index_render_anchor_links_without_heading_leaves() -> None:
    """验证标题 anchor 与递归目录 title leaf 的链接输出。"""
    index = IndexBlock(
        type="index",
        index=0,
        content=[
            ParagraphTitleBlock(
                type="paragraph_title",
                level=2,
                anchor="toc-a",
                content="1 Section\t12",
            ),
            IndexBlock(
                type="index",
                content=[TextBlock(type="text", content="Plain")],
            ),
        ],
    )
    title = ParagraphTitleBlock(
        type="paragraph_title",
        index=1,
        level=6,
        anchor="toc-a",
        content="Section",
    )

    rendered = render_markdown(_middle(_page(0, index, title)))

    assert rendered.startswith("- [1 Section](#toc-a)\n    - Plain")
    assert '<a id="toc-a"></a>\n###### Section' in rendered


def test_equation_uses_content_then_image_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证行间公式定界符配置及空公式图片回退。"""
    configured = Config(
        render={
            "latex_delimiters": {
                "display": {"left": "\\[", "right": "\\]"},
                "inline": {"left": "\\(", "right": "\\)"},
            }
        }
    )
    monkeypatch.setattr("mineru.render.markdown.config", configured)
    middle = _middle(
        _page(
            0,
            EquationBlock(type="equation", index=0, content="x=1"),
            EquationBlock(type="equation", index=1, content="", image_path="images/e.png"),
            TextBlock(type="text", index=2, content="inline <eq>y</eq>"),
        )
    )

    assert render_markdown(middle) == "\\[\nx=1\n\\]\n\n![](images/e.png)\n\ninline \\(y\\)"


@pytest.mark.parametrize(
    "html_table",
    [
        "<table><tr><td rowspan='2'>A</td></tr><tr><td>B</td></tr></table>",
        "<table><tr><td colspan='2'>A</td></tr></table>",
        "<table><tr><td><table><tr><td>N</td></tr></table></td></tr></table>",
        "<table><tr><td><img src='images/a.png'></td></tr></table>",
        "<table><tr><td><ul><li>A</li></ul></td></tr></table>",
        "<table><tr><td><p>A</p><p>B</p></td></tr></table>",
    ],
)
def test_complex_html_tables_fall_back_to_html(html_table: str) -> None:
    """验证不可无损转换的表格结构保持 HTML。"""
    rendered = render_markdown(_middle(_page(0, _table(0, html_table))), asset_base_url="assets")

    assert rendered.startswith("<table")
    assert "| --- |" not in rendered
    if "<img" in html_table:
        assert "assets/images/a.png" in rendered


def test_simple_html_table_converts_to_gfm_and_preserves_inline_formula() -> None:
    """验证简单 HTML table 转 GFM，并保留单元格公式反斜杠。"""
    content = (
        "<table><tr><th>Name</th><th>Value</th></tr>"
        "<tr><td>A|B</td><td><eq>\\frac{1}{2}</eq></td></tr></table>"
    )

    assert render_markdown(_middle(_page(0, _table(0, content)))) == "\n".join(
        [
            "| Name | Value |",
            "| --- | --- |",
            r"| A\|B | $\frac{1}{2}$ |",
        ]
    )


def test_spatial_table_uses_dynamic_fence_and_empty_table_uses_image() -> None:
    """验证空间投影文本保留空白，且空表回退图片。"""
    spatial = _table(0, "A   B\n```\n1   2")
    empty = TableBlock(
        type="table",
        index=1,
        content=[TableBodyBlock(type="table_body", index=1, content="", image_path="images/t.png")],
    )

    rendered = render_markdown(_middle(_page(0, spatial, empty)))

    assert rendered.startswith("````\nA   B\n```\n1   2\n````")
    assert rendered.endswith("![](images/t.png)")


def test_cross_page_table_merges_only_in_default_mode() -> None:
    """验证续表只在 DEFAULT 跨页合并，FULL 保持分页。"""
    previous_html = "<table><tr><th>H</th></tr><tr><td>A</td></tr></table>"
    current_html = "<table><tr><th>H</th></tr><tr><td>B</td></tr></table>"
    middle = _middle(
        _page(0, _pdf_table(0, previous_html)),
        _page(1, _pdf_table(0, current_html, continues_prev=True)),
        file_suffix="pdf",
    )
    original = deepcopy(middle)

    default = render_markdown(middle)
    full = render_markdown(middle, mode=MarkdownRenderMode.FULL)

    assert default == "\n".join(["| H |", "| --- |", "| A |", "| B |"])
    assert full.count("| --- |") == 2
    assert "\n\n---\n\n" in full
    assert middle == original


def test_code_uses_language_and_dynamic_fence() -> None:
    """验证普通代码使用 guess_lang 和足够长的 fenced block。"""
    code = CodeBlock(
        type="code",
        index=0,
        sub_type="code",
        guess_lang="python",
        content=[CodeBodyBlock(type="code_body", index=0, content="print('x')\n```")],
    )
    invalid_language = CodeBlock(
        type="code",
        index=1,
        sub_type="code",
        guess_lang="python bad",
        content=[CodeBodyBlock(type="code_body", index=1, content="x")],
    )

    rendered = render_markdown(_middle(_page(0, code, invalid_language)))

    assert rendered.startswith("````python\nprint('x')\n```\n````")
    assert rendered.endswith("```txt\nx\n```")


def test_algorithm_preserves_whitespace_comparisons_scripts_and_formula() -> None:
    """验证算法 HTML 原样保留缩进、比较符、上下标并支持相邻公式。"""
    algorithm = CodeBlock(
        type="code",
        index=0,
        sub_type="algorithm",
        content=[
            CodeBodyBlock(
                type="code_body",
                index=0,
                content=(
                    "if a < b and c > d:\n  T<sub>queue</sub><sup>2</sup> = "
                    "<eq>a < b</eq><eq>c > d</eq>"
                ),
            )
        ],
    )

    rendered = render_markdown(_middle(_page(0, algorithm)))

    assert 'class="mineru-algorithm"' in rendered
    assert "if a < b and c > d:\n  T<sub>queue</sub><sup>2</sup> = $a < b$ $c > d$" in rendered
    assert "&lt;" not in rendered
    assert "&gt;" not in rendered


def test_image_path_precedes_base64_and_visual_child_order_is_preserved() -> None:
    """验证图片资源优先级、base URL 和视觉子块原始顺序。"""
    image = ImageBlock.model_validate(
        {
            "type": "image",
            "index": 0,
            "content": [
                {"type": "image_caption", "content": "before"},
                {
                    "type": "image_body",
                    "index": 0,
                    "content": "description",
                    "image_path": "images/a b.png",
                    "image_base64": "data:image/png;base64,AAAA",
                },
                {"type": "image_footnote", "content": "after"},
            ],
        }
    )

    rendered = render_markdown(_middle(_page(0, image)), asset_base_url="https://cdn.example/doc")

    assert rendered.index("before") < rendered.index("https://cdn.example/doc/images/a%20b.png")
    assert rendered.index("description") < rendered.index("after")
    assert "data:image" not in rendered


def test_chart_without_image_outputs_existing_gfm_content() -> None:
    """验证无图片图表直接输出已有 GFM 内容。"""
    chart = ChartBlock(
        type="chart",
        index=0,
        content=[ChartBodyBlock(type="chart_body", index=0, content="| A |\n| --- |\n| 1 |")],
    )

    assert render_markdown(_middle(_page(0, chart))) == "| A |\n| --- |\n| 1 |"


def test_chart_without_image_converts_simple_html_table_to_gfm() -> None:
    """验证无图片 chart 的简单 HTML table 直接转换为 GFM。"""
    chart = ChartBlock(
        type="chart",
        index=0,
        content=[
            ChartBodyBlock(
                type="chart_body",
                index=0,
                content="<table><tr><th>A</th></tr><tr><td>1</td></tr></table>",
            )
        ],
    )

    assert render_markdown(_middle(_page(0, chart))) == "| A |\n| --- |\n| 1 |"


def test_chart_with_image_places_converted_gfm_in_details() -> None:
    """验证有图片 chart 保留图片，并把简单表格 GFM 放入 details。"""
    chart = ChartBlock(
        type="chart",
        index=0,
        sub_type="line chart",
        content=[
            ChartBodyBlock(
                type="chart_body",
                index=0,
                content="<table><tr><th>A</th></tr><tr><td>1</td></tr></table>",
                image_path="images/chart.png",
            )
        ],
    )

    rendered = render_markdown(_middle(_page(0, chart)))

    assert rendered.startswith("![](images/chart.png)\n\n<details>")
    assert "| A |\n| --- |\n| 1 |" in rendered
    assert "<table>" not in rendered


def test_chart_complex_html_table_remains_html() -> None:
    """验证 chart 的复杂 HTML table 不会被有损转换为 GFM。"""
    chart = ChartBlock(
        type="chart",
        index=0,
        content=[
            ChartBodyBlock(
                type="chart_body",
                index=0,
                content='<table><tr><td colspan="2">A</td></tr></table>',
            )
        ],
    )

    rendered = render_markdown(_middle(_page(0, chart)))

    assert rendered.startswith("<table>")
    assert 'colspan="2"' in rendered


def test_direct_base64_image_fallback() -> None:
    """验证未外置图片时 Markdown 直接使用 data URI。"""
    image = ImageBlock(
        type="image",
        index=0,
        content=[
            ImageBodyBlock(
                type="image_body",
                index=0,
                content="",
                image_base64="data:image/png;base64,AAAA",
            )
        ],
    )

    assert render_markdown(_middle(_page(0, image))) == "![](data:image/png;base64,AAAA)"
