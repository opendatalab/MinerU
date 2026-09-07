from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from docvortex.document.detection import guess_suffix_by_bytes, guess_suffix_by_path

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.parser import api_server, parse, parse_async
from mineru.parser.api_server import CreateJobRequest, FileStore
from mineru.render.html import render_html
from mineru.render.markdown import render_markdown
from mineru.types import TableBlock, TableBodyBlock


def test_csv_doc_analyze_sync_async_and_render_contracts_match() -> None:
    """验证同步异步 Analyze 元数据和既有表格渲染链保持一致。"""
    payload = "姓名,年龄\n张三,30\n".encode()
    middle, model = doc_analyze(payload, file_suffix="csv")
    async_middle, async_model = asyncio.run(aio_doc_analyze(payload, file_suffix="csv"))

    assert model.metadata.file_suffix == async_model.metadata.file_suffix == "csv"
    assert model.extensions["mineru"]["tier"] == async_model.extensions["mineru"]["tier"] == "flash"
    assert model.extensions["mineru"]["parse_mode"] == async_model.extensions["mineru"]["parse_mode"] == "txt"
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
    assert result.middle_json.metadata.file_suffix == async_result.middle_json.metadata.file_suffix == "csv"
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
    assert middle_payload["metadata"]["file_suffix"] == "csv"
    assert middle_payload["extensions"]["mineru"]["tier"] == "flash"
    assert middle_payload["extensions"]["mineru"]["parse_mode"] == "txt"

    structured_record = file_store.get_file(parsed_file.output_files.structured_content.file_id)
    assert structured_record.sha256sum is not None
    structured_payload = json.loads(file_store.read_blob(structured_record.sha256sum))
    assert structured_payload["metadata"]["file_suffix"] == "csv"
    assert structured_payload["pages"][0]["blocks"][0]["type"] == "table"
