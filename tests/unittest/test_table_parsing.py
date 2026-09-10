# Copyright (c) Opendatalab. All rights reserved.
from mineru.utils.table_parsing import parse_table_html, parse_table_html_to_dict


SIMPLE_TABLE = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"

SPAN_TABLE = (
    '<table><thead><tr><th>Name</th><th colspan="2">Scores</th></tr></thead>'
    '<tbody><tr><td rowspan="2">Alice</td><td>90</td><td>85</td></tr>'
    "<tr><td>70</td><td>75</td></tr></tbody></table>"
)


def test_simple_table_grid() -> None:
    structure = parse_table_html(SIMPLE_TABLE)

    assert structure.row_count == 2
    assert structure.column_count == 2
    assert structure.has_merged_cells is False
    assert [cell.text for cell in structure.rows[0].cells] == ["a", "b"]
    assert [(c.row_index, c.col_index) for c in structure.rows[1].cells] == [(1, 0), (1, 1)]


def test_colspan_expands_column_count() -> None:
    structure = parse_table_html(SPAN_TABLE)

    assert structure.column_count == 3
    assert structure.has_merged_cells is True
    scores = structure.rows[0].cells[1]
    assert scores.colspan == 2
    assert scores.col_index == 1


def test_rowspan_shifts_following_row_columns() -> None:
    """被上一行 rowspan 占用的列不应重复分配。"""
    structure = parse_table_html(SPAN_TABLE)

    # "Alice" spans rows 1-2, so row 2's first cell starts at column 1, not 0.
    assert [(c.text, c.col_index) for c in structure.rows[2].cells] == [("70", 1), ("75", 2)]


def test_cell_at_resolves_spanned_coordinates() -> None:
    structure = parse_table_html(SPAN_TABLE)

    assert structure.cell_at(1, 0).text == "Alice"
    assert structure.cell_at(2, 0).text == "Alice"
    assert structure.cell_at(0, 2).text == "Scores"
    assert structure.cell_at(99, 99) is None


def test_header_detection_from_thead_and_th() -> None:
    structure = parse_table_html(SPAN_TABLE)

    assert structure.rows[0].is_header_row is True
    assert structure.header_row_count == 1
    assert structure.rows[1].is_header_row is False
    assert all(cell.is_header for cell in structure.rows[0].cells)


def test_returns_none_without_rows() -> None:
    assert parse_table_html("") is None
    assert parse_table_html("   ") is None
    assert parse_table_html("<p>no table here</p>") is None


def test_malformed_span_attributes_fall_back_to_one() -> None:
    """模型输出的 colspan 可能非法，需回退为 1 而不是抛异常。"""
    structure = parse_table_html(
        '<table><tr><td colspan="abc">x</td><td colspan="0">y</td>'
        '<td rowspan="-3">z</td></tr></table>'
    )

    assert structure.column_count == 3
    assert [c.colspan for c in structure.rows[0].cells] == [1, 1, 1]
    assert structure.rows[0].cells[2].rowspan == 1


def test_nested_table_is_captured_but_not_flattened() -> None:
    structure = parse_table_html(
        "<table><tr><td><table><tr><td>inner</td></tr></table></td>"
        "<td>outer</td></tr></table>"
    )

    assert structure.row_count == 1
    assert structure.column_count == 2
    assert structure.has_nested_tables is True
    assert "inner" in structure.rows[0].cells[0].nested_html
    assert structure.rows[0].cells[1].nested_html is None


def test_to_dict_shape_is_json_serializable() -> None:
    data = parse_table_html_to_dict(SPAN_TABLE)

    assert data["row_count"] == 3
    assert data["column_count"] == 3
    assert data["header_row_count"] == 1
    assert data["rows"][0]["cells"][0] == {
        "text": "Name",
        "row_index": 0,
        "col_index": 0,
        "colspan": 1,
        "rowspan": 1,
        "is_header": True,
    }
    assert parse_table_html_to_dict("") is None


def _middle_json(html: str) -> list[dict]:
    """构造与真实 middle.json 一致的 table block 结构。"""
    return [
        {
            "para_blocks": [
                {
                    "type": "table",
                    "blocks": [
                        {
                            "type": "table_body",
                            "lines": [{"spans": [{"type": "table", "html": html}]}],
                        },
                        {
                            "type": "table_caption",
                            "lines": [{"spans": [{"type": "text", "content": "Cap"}]}],
                        },
                    ],
                }
            ]
        }
    ]


def _body_span(pdf_info: list[dict]) -> dict:
    return pdf_info[0]["para_blocks"][0]["blocks"][0]["lines"][0]["spans"][0]


def test_attach_table_structure_populates_nested_table_body_span() -> None:
    from mineru.backend.utils.runtime_utils import attach_table_structure

    pdf_info = _middle_json(SPAN_TABLE)
    attach_table_structure(pdf_info)

    structure = _body_span(pdf_info)["table_structure"]
    assert structure["row_count"] == 3
    assert structure["column_count"] == 3
    assert structure["rows"][2]["cells"][0]["col_index"] == 1


def test_attach_table_structure_handles_pipeline_html_wrapper() -> None:
    """pipeline 表格识别模型输出带 <html><body> 包装与 <style>。"""
    from mineru.backend.utils.runtime_utils import attach_table_structure

    wrapped = (
        "<html><style>table{border:1px solid black;}</style><body>"
        "<table><tr><td>A</td><td>B</td></tr></table></body></html>"
    )
    pdf_info = _middle_json(wrapped)
    attach_table_structure(pdf_info)

    structure = _body_span(pdf_info)["table_structure"]
    assert structure["row_count"] == 1
    assert structure["column_count"] == 2
    assert [c["text"] for c in structure["rows"][0]["cells"]] == ["A", "B"]


def test_attach_table_structure_preserves_original_html() -> None:
    from mineru.backend.utils.runtime_utils import attach_table_structure

    pdf_info = _middle_json(SIMPLE_TABLE)
    attach_table_structure(pdf_info)

    assert _body_span(pdf_info)["html"] == SIMPLE_TABLE


def test_attach_table_structure_skips_non_table_spans() -> None:
    from mineru.backend.utils.runtime_utils import attach_table_structure

    pdf_info = _middle_json(SIMPLE_TABLE)
    attach_table_structure(pdf_info)

    caption_span = pdf_info[0]["para_blocks"][0]["blocks"][1]["lines"][0]["spans"][0]
    assert "table_structure" not in caption_span


def test_attach_table_structure_omits_key_for_unparsable_html() -> None:
    from mineru.backend.utils.runtime_utils import attach_table_structure

    pdf_info = _middle_json("")
    attach_table_structure(pdf_info)

    assert "table_structure" not in _body_span(pdf_info)


def test_attach_table_structure_disabled_by_env_flag(monkeypatch) -> None:
    from mineru.backend.utils.runtime_utils import attach_table_structure

    monkeypatch.setenv("MINERU_TABLE_STRUCTURE_ENABLE", "false")
    pdf_info = _middle_json(SPAN_TABLE)
    attach_table_structure(pdf_info)

    assert "table_structure" not in _body_span(pdf_info)


def test_attach_table_structure_unknown_flag_value_defaults_on(monkeypatch) -> None:
    from mineru.backend.utils.runtime_utils import attach_table_structure

    monkeypatch.setenv("MINERU_TABLE_STRUCTURE_ENABLE", "maybe")
    pdf_info = _middle_json(SPAN_TABLE)
    attach_table_structure(pdf_info)

    assert "table_structure" in _body_span(pdf_info)
