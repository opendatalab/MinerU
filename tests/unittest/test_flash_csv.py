from __future__ import annotations

import asyncio
import codecs
import json
from io import BytesIO
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.model.flash import CsvModel
from mineru.model.flash import csv as csv_module
from mineru.parser import parse, parse_async
from mineru.parser import api_server
from mineru.parser.api_server import CreateJobRequest, FileStore
from mineru.parser.file_type import guess_suffix_by_bytes, guess_suffix_by_path
from mineru.render.html import render_html
from mineru.render.markdown import render_markdown
from mineru.types import BlockType, TableBlock, TableBodyBlock


def _raw_table_html(payload: bytes) -> str:
    """解析 CSV 并返回 model-list 中唯一表格的 HTML。"""
    pages = CsvModel().predict(BytesIO(payload))
    assert len(pages) == 1
    assert len(pages[0]) == 1
    assert pages[0][0]["type"] == BlockType.TABLE
    return pages[0][0]["content"]


def _html_rows(payload: bytes) -> list[list[str]]:
    """把 CSV 投影 HTML 还原为逐行单元格纯文本，方便断言语义。"""
    soup = BeautifulSoup(_raw_table_html(payload), "html.parser")
    return [[cell.get_text("\n") for cell in row.find_all(["th", "td"])] for row in soup.find_all("tr")]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (codecs.BOM_UTF8 + "姓名,年龄\n张三,30\n".encode(), [["姓名", "年龄"], ["张三", "30"]]),
        (codecs.BOM_UTF16_LE + "姓名;年龄\n张三;30\n".encode("utf-16le"), [["姓名", "年龄"], ["张三", "30"]]),
        ("姓名|年龄\n张三|30\n".encode("gb18030"), [["姓名", "年龄"], ["张三", "30"]]),
        ("name\tcity\nAndré\tZürich\n".encode("cp1252"), [["name", "city"], ["André", "Zürich"]]),
    ],
)
def test_csv_decodes_common_encodings_and_delimiters(payload: bytes, expected: list[list[str]]) -> None:
    """验证常见中西文编码和四种分隔符都进入同一表格语义。"""
    assert _html_rows(payload) == expected


def test_csv_sep_directive_overrides_delimiter_sniffing() -> None:
    """验证 Excel sep 指令只控制分隔符，不作为数据行输出。"""
    payload = 'sep=;\r\nname;note\r\nAlice;"1,2"\r\n'.encode()

    assert _html_rows(payload) == [["name", "note"], ["Alice", "1,2"]]


def test_csv_preserves_multiline_quotes_padding_ragged_rows_and_safe_text() -> None:
    """验证多行字段、引号、前导零、首尾空格、短行补齐和 HTML 转义。"""
    payload = (
        'name,note,code,markup\n'
        'Alice,"line 1\nline 2",001,"<script>alert(1)</script>"\n'
        'Bob,"  say ""hi""  ",02\n'
    ).encode()

    table_html = _raw_table_html(payload)
    assert "line 1<br>line 2" in table_html
    assert "<script>" not in table_html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in table_html
    assert _html_rows(payload) == [
        ["name", "note", "code", "markup"],
        ["Alice", "line 1\nline 2", "001", "<script>alert(1)</script>"],
        ["Bob", '  say "hi"  ', "02", ""],
    ]

    middle, _ = doc_analyze(payload, file_suffix="csv")
    markdown = render_markdown(middle)
    standalone_html = render_html(middle)
    assert "<script>alert(1)</script>" not in markdown
    assert "&lt;script>alert(1)&lt;/script>" in markdown
    assert "<script>alert(1)</script>" not in standalone_html


@pytest.mark.parametrize(
    ("payload", "expected_header"),
    [
        (b"name,age\nAlice,30\nBob,40\n", True),
        (b"name,city\nAlice,London\nBob,Paris\n", True),
        (b"1,10\n2,20\n3,30\n", False),
        (b"only,one,row\n", False),
    ],
)
def test_csv_header_inference_is_deterministic(payload: bytes, expected_header: bool) -> None:
    """验证有类型证据、纯文本标签、纯数据和单行文件的表头边界。"""
    soup = BeautifulSoup(_raw_table_html(payload), "html.parser")
    assert bool(soup.find("th")) is expected_header


def test_csv_empty_single_column_and_blank_records_keep_one_logical_page() -> None:
    """验证空 CSV、单列 CSV 和空记录都保留确定的一页语义。"""
    assert CsvModel().predict(BytesIO(b"")) == [[]]
    assert _html_rows(b"value\none\n\ntwo\n") == [["value"], ["one"], [""], ["two"]]


@pytest.mark.parametrize(
    "payload",
    [
        b'a,b\n1,"unterminated\n',
        b"a,b\n1,\x81\n",
    ],
)
def test_csv_rejects_malformed_syntax_and_unsupported_encoding(payload: bytes) -> None:
    """验证损坏引号或无法严格解码的字节会让整份 CSV 失败。"""
    with pytest.raises(ValueError):
        CsvModel().predict(BytesIO(payload))


@pytest.mark.parametrize(
    ("constant", "limit", "payload", "message"),
    [
        ("MAX_CSV_BYTES", 3, b"a,b\n", "max_bytes"),
        ("MAX_CSV_ROWS", 1, b"a\nb\n", "max_rows"),
        ("MAX_CSV_COLUMNS", 1, b"a,b\n", "max_columns"),
        ("MAX_CSV_GRID_SLOTS", 3, b"a,b\n1,2\n", "max_grid_slots"),
    ],
)
def test_csv_enforces_resource_limits(
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    limit: int,
    payload: bytes,
    message: str,
) -> None:
    """验证输入、行、列和规则化网格限制都采用显式失败。"""
    monkeypatch.setattr(csv_module, constant, limit)

    with pytest.raises(ValueError, match=message):
        CsvModel().predict(BytesIO(payload))


def test_csv_doc_analyze_sync_async_and_render_contracts_match() -> None:
    """验证同步异步 Analyze 元数据和既有表格渲染链保持一致。"""
    payload = "姓名,年龄\n张三,30\n".encode()
    middle, model = doc_analyze(payload, file_suffix="csv")
    async_middle, async_model = asyncio.run(aio_doc_analyze(payload, file_suffix="csv"))

    assert model.file_suffix == async_model.file_suffix == "csv"
    assert model.effort == async_model.effort == "flash"
    assert model.parse_mode == async_model.parse_mode == "txt"
    assert middle.model_dump() == async_middle.model_dump()
    assert len(middle.pages) == 1
    table = middle.pages[0].blocks[0]
    assert isinstance(table, TableBlock)
    assert isinstance(table.content[0], TableBodyBlock)
    assert render_markdown(middle) == "| 姓名 | 年龄 |\n| --- | --- |\n| 张三 | 30 |"
    assert "<table>" in render_html(middle)


def test_csv_path_parsing_and_signatureless_detection(tmp_path: Path) -> None:
    """验证 .csv 扩展名兜底、无路径字节不猜 CSV，并阻止 .txt 自动升级。"""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("name,age\nAlice,30\n", encoding="utf-8")
    text_path = tmp_path / "sample.txt"
    text_path.write_text("name,age\nAlice,30\n", encoding="utf-8")
    fake_pdf_path = tmp_path / "fake.csv"
    fake_pdf_path.write_bytes(b"%PDF-1.7\n")

    assert guess_suffix_by_bytes(csv_path.read_bytes()) == "txt"
    assert guess_suffix_by_bytes(csv_path.read_bytes(), str(csv_path)) == "csv"
    assert guess_suffix_by_path(csv_path) == "csv"
    assert guess_suffix_by_path(text_path) == "txt"
    assert guess_suffix_by_path(fake_pdf_path) == "pdf"

    result = parse(csv_path)
    async_result = asyncio.run(parse_async(csv_path))
    assert result.middle_json.file_suffix == async_result.middle_json.file_suffix == "csv"
    assert result.markdown() == async_result.markdown()
    with pytest.raises(ValueError, match="Unsupported file type: txt"):
        parse(text_path)


def test_csv_parse_server_job_emits_structured_outputs_with_flash_metadata(tmp_path: Path) -> None:
    """验证 parse-server 实际解析 CSV，并输出 Markdown、Middle JSON 与结构化内容。"""
    source = tmp_path / "sample.csv"
    source.write_text("name,age\nAlice,30\n", encoding="utf-8")
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
    assert parsed_file.output_files is not None
    assert parsed_file.output_files.markdown is not None
    assert parsed_file.output_files.middle_json is not None
    assert parsed_file.output_files.structured_content is not None

    markdown_record = file_store.get_file(parsed_file.output_files.markdown.file_id)
    assert markdown_record.sha256sum is not None
    markdown = file_store.read_blob(markdown_record.sha256sum).decode()
    assert "| name | age |" in markdown

    middle_record = file_store.get_file(parsed_file.output_files.middle_json.file_id)
    assert middle_record.sha256sum is not None
    middle_payload = json.loads(file_store.read_blob(middle_record.sha256sum))
    assert middle_payload["file_suffix"] == "csv"
    assert middle_payload["effort"] == "flash"
    assert middle_payload["parse_mode"] == "txt"

    structured_record = file_store.get_file(parsed_file.output_files.structured_content.file_id)
    assert structured_record.sha256sum is not None
    structured_payload = json.loads(file_store.read_blob(structured_record.sha256sum))
    assert structured_payload["file_suffix"] == "csv"
    assert structured_payload["pages"][0]["blocks"][0]["type"] == "table"
