from mineru.render import render_markdown
from mineru.render.markdown import blocks_to_markdown
from mineru.render.markdown_table import to_markdown_table
from mineru.types import Block, BlockType, ContentType, Line, PageInfo, Span


def test_table_span_ignores_legacy_html_payload_alias() -> None:
    """表格 span 只接受 content 承载 HTML，不再兼容旧 html 别名。"""
    span = Span.from_dict(
        {
            "type": ContentType.TABLE,
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "html": "<table><tr><td>A</td></tr></table>",
        }
    )

    assert span.content == ""
    assert span.to_dict() == {
        "type": ContentType.TABLE,
        "bbox": (0.0, 0.0, 10.0, 10.0),
    }


def test_to_markdown_table_renders_simple_table() -> None:
    html = """
    <table>
      <tr><th>Name</th><th>Score</th></tr>
      <tr><td>Alice</td><td>90</td></tr>
      <tr><td>Bob</td><td>85</td></tr>
    </table>
    """

    assert to_markdown_table(html) == "\n".join(
        [
            "| Name | Score |",
            "| --- | --- |",
            "| Alice | 90 |",
            "| Bob | 85 |",
        ]
    )


def test_to_markdown_table_falls_back_for_colspan_cells() -> None:
    html = """
    <table>
      <tr><th colspan="2">User</th><th>Score</th></tr>
      <tr><td>Alice</td><td>Math</td><td>90</td></tr>
    </table>
    """

    assert to_markdown_table(html) == html.strip()


def test_to_markdown_table_falls_back_for_rowspan_cells() -> None:
    html = """
    <table>
      <tr><th>Name</th><th>Subject</th><th>Score</th></tr>
      <tr><td rowspan="2">Alice</td><td>Math</td><td>90</td></tr>
      <tr><td>English</td><td>95</td></tr>
    </table>
    """

    assert to_markdown_table(html) == html.strip()


def test_to_markdown_table_preserves_simple_inline_markup() -> None:
    html = """
    <table>
      <tr><th>Item</th><th>Note</th></tr>
      <tr><td><code>x|y</code></td><td><a href="https://example.com">docs</a><br>ready</td></tr>
    </table>
    """

    assert to_markdown_table(html) == "\n".join(
        [
            "| Item | Note |",
            "| --- | --- |",
            r"| `x\|y` | [docs](https://example.com)<br>ready |",
        ]
    )


def test_to_markdown_table_falls_back_for_nested_tables() -> None:
    html = """
    <table>
      <tr>
        <td>
          <table><tr><td>nested</td></tr></table>
        </td>
      </tr>
    </table>
    """

    assert to_markdown_table(html) == html.strip()


def test_to_markdown_table_falls_back_for_complex_block_content() -> None:
    html = """
    <table>
      <tr><th>Item</th><th>Details</th></tr>
      <tr><td>A</td><td><ul><li>one</li><li>two</li></ul></td></tr>
    </table>
    """

    assert to_markdown_table(html) == html.strip()


def test_blocks_to_markdown_prefers_markdown_table_when_enabled() -> None:
    html = """
    <table>
      <tr><th>Name</th><th>Score</th></tr>
      <tr><td>Alice</td><td>90</td></tr>
    </table>
    """.strip()
    table_block = Block(
        index=0,
        type=BlockType.TABLE,
        bbox=(0.0, 0.0, 10.0, 10.0),
        blocks=[
            Block(
                index=0,
                type=BlockType.TABLE_BODY,
                bbox=(0.0, 0.0, 10.0, 10.0),
                lines=[
                    Line(
                        bbox=(0.0, 0.0, 10.0, 10.0),
                        spans=[Span(type=ContentType.TABLE, bbox=(0.0, 0.0, 10.0, 10.0), content=html)],
                    )
                ],
            )
        ],
    )

    assert blocks_to_markdown([table_block], prefer_markdown_table=False) == [html]
    assert blocks_to_markdown([table_block], prefer_markdown_table=True) == [
        "\n".join(
            [
                "| Name | Score |",
                "| --- | --- |",
                "| Alice | 90 |",
            ]
        )
    ]


def test_blocks_to_markdown_keeps_merged_cell_tables_as_html() -> None:
    html = """
    <table>
      <tr><th colspan="2">User</th><th>Score</th></tr>
      <tr><td>Alice</td><td>Math</td><td>90</td></tr>
    </table>
    """.strip()
    table_block = Block(
        index=0,
        type=BlockType.TABLE,
        bbox=(0.0, 0.0, 10.0, 10.0),
        blocks=[
            Block(
                index=0,
                type=BlockType.TABLE_BODY,
                bbox=(0.0, 0.0, 10.0, 10.0),
                lines=[
                    Line(
                        bbox=(0.0, 0.0, 10.0, 10.0),
                        spans=[Span(type=ContentType.TABLE, bbox=(0.0, 0.0, 10.0, 10.0), content=html)],
                    )
                ],
            )
        ],
    )

    assert blocks_to_markdown([table_block], prefer_markdown_table=True) == [html]


def test_render_markdown_prefers_markdown_table_when_enabled() -> None:
    html = """
    <table>
      <tr><th>Name</th><th>Score</th></tr>
      <tr><td>Alice</td><td>90</td></tr>
    </table>
    """.strip()
    table_block = Block(
        index=0,
        type=BlockType.TABLE,
        bbox=(0.0, 0.0, 10.0, 10.0),
        blocks=[
            Block(
                index=0,
                type=BlockType.TABLE_BODY,
                bbox=(0.0, 0.0, 10.0, 10.0),
                lines=[
                    Line(
                        bbox=(0.0, 0.0, 10.0, 10.0),
                        spans=[Span(type=ContentType.TABLE, bbox=(0.0, 0.0, 10.0, 10.0), content=html)],
                    )
                ],
            )
        ],
    )
    page = PageInfo(page_idx=0, para_blocks=[table_block], _backend="pipeline")

    assert render_markdown([page], prefer_markdown_table=False) == html
    assert render_markdown([page], prefer_markdown_table=True) == "\n".join(
        [
            "| Name | Score |",
            "| --- | --- |",
            "| Alice | 90 |",
        ]
    )
