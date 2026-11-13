# Copyright (c) Opendatalab. All rights reserved.
"""验证共享资源限制在本地解析及 Tasks/Router 边界保持输入错误语义。"""

from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path
from typing import Literal, NoReturn
from zipfile import ZIP_DEFLATED, ZipFile

import docvortex
import pytest
from docvortex.document import detection
from docvortex.errors import DocumentError
from docvortex.schema import DocumentProperties
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mineru.errors import InvalidRequestError, error_type_for, http_status_for
from mineru.integrations.docvortex import read_source_properties, translate_document_errors
from mineru.parser import MinerUParser, mineru_parser

import test_task_api

service = test_task_api.service
_Stage = Literal["detection", "metadata", "prepare", "analysis"]
_STAGES: tuple[_Stage, ...] = ("detection", "metadata", "prepare", "analysis")


def _reject_stage(monkeypatch: pytest.MonkeyPatch, stage: _Stage, error: DocumentError) -> None:
    """只替换指定共享阶段，让其余输入准备和服务生命周期走真实代码。"""

    def reject(*args: object, **kwargs: object) -> NoReturn:
        """模拟共享文档层在分配超限资源前拒绝输入。"""
        raise error

    async def reject_async(*args: object, **kwargs: object) -> NoReturn:
        """模拟原生异步分析入口传出相同的共享错误。"""
        raise error

    if stage == "detection":
        monkeypatch.setattr(detection, "guess_suffix_by_path", reject)
    elif stage == "metadata":
        monkeypatch.setattr(docvortex, "extract_metadata", reject)
    elif stage == "prepare":
        monkeypatch.setattr(MinerUParser, "_maybe_adjust_pdf_bytes", reject)
    else:
        monkeypatch.setattr(mineru_parser, "doc_analyze", reject)
        monkeypatch.setattr(mineru_parser, "aio_doc_analyze", reject_async)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stage", _STAGES)
def test_parser_preserves_resource_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: _Stage, asynchronous: bool
) -> None:
    """同步异步入口对检测、属性、准备和分析阶段统一保留资源限制信息。"""
    source = tmp_path / "limited.html"
    source.write_bytes(test_task_api.HTML)
    shared_error = DocumentError("resource_limit", "Document exceeds the resource budget", "document")
    _reject_stage(monkeypatch, stage, shared_error)
    parser = MinerUParser(tier="flash")

    with pytest.raises(InvalidRequestError) as raised:
        if asynchronous:
            asyncio.run(parser.parse_async(source))
        else:
            parser.parse(source)

    assert raised.value.code == "resource_limit"
    assert raised.value.message == shared_error.message
    assert raised.value.param == "document"
    assert raised.value.type == "invalid_request_error"
    assert raised.value.__cause__ is shared_error


def test_metadata_resource_limit_is_not_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    """单独读取属性也必须拒绝超限输入，不能返回空属性继续解析。"""
    _reject_stage(monkeypatch, "metadata", DocumentError("resource_limit", "Archive is too large"))
    with pytest.raises(InvalidRequestError, match="Archive is too large") as raised:
        read_source_properties(test_task_api.HTML, "html")
    assert raised.value.code == "resource_limit"
    assert error_type_for("resource_limit") == "invalid_request_error"
    assert http_status_for("resource_limit") == 400


def test_metadata_other_document_errors_stay_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    """普通属性提取失败继续返回空属性，由正文引擎判断输入是否有效。"""
    _reject_stage(monkeypatch, "metadata", DocumentError("file_corrupted", "Metadata is unavailable"))
    assert read_source_properties(test_task_api.HTML, "html") == DocumentProperties()


def test_parser_preserves_recognized_document_error_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """明确输入错误转换为产品异常，保留代码、消息与原始异常链。"""
    source = tmp_path / "document.html"
    source.write_bytes(test_task_api.HTML)
    shared_error = DocumentError("file_corrupted", "Document is malformed")
    _reject_stage(monkeypatch, "analysis", shared_error)
    with pytest.raises(InvalidRequestError) as raised:
        MinerUParser(tier="flash").parse(source)
    assert raised.value.code == "file_corrupted"
    assert raised.value.__cause__ is shared_error


def test_xml_failure_outside_native_input_is_not_reclassified() -> None:
    """同类底层异常若发生在模型阶段，不得归入可跳过的输入错误。"""
    from lxml.etree import XMLSyntaxError

    original = XMLSyntaxError("model configuration", 201, 1, 1)
    with pytest.raises(XMLSyntaxError) as raised:
        with translate_document_errors():
            raise original
    assert raised.value is original


def test_parser_does_not_reclassify_unknown_document_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """未约定的共享错误不能被猜测为可跳过的单文档失败。"""
    source = tmp_path / "document.html"
    source.write_bytes(test_task_api.HTML)
    shared_error = DocumentError("unknown_runtime_contract", "Unexpected state")
    _reject_stage(monkeypatch, "analysis", shared_error)
    with pytest.raises(DocumentError) as raised:
        MinerUParser(tier="flash").parse(source)
    assert raised.value is shared_error


@pytest.mark.parametrize("stage", _STAGES)
def test_task_resource_limit_isolated_to_file(
    service: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch, stage: _Stage
) -> None:
    """直接 API 和 Router 保留逐文件错误码，拒绝输入后仍能处理正常任务。"""
    client, _ = service
    inputs = [("limited.html", test_task_api.HTML)]
    with monkeypatch.context() as context:
        _reject_stage(context, stage, DocumentError("resource_limit", "Document exceeds the resource budget", "document"))
        task = test_task_api._submit(client, inputs)
        test_task_api._upload(client, task, inputs)
        result = test_task_api._terminal(client, task)

    assert result["status"] == "failed"
    assert result["files"][0]["status"] == "failed"
    assert result["files"][0]["error"] == {
        "code": "resource_limit",
        "message": "Document exceeds the resource budget",
        "param": "document",
    }
    assert result["files"][0].get("result_url") is None

    recovered = test_task_api._submit(client, inputs)
    test_task_api._upload(client, recovered, inputs)
    assert test_task_api._terminal(client, recovered)["status"] == "completed"


def _office_document(suffix: str) -> bytes:
    """创建真实 Office 输入，供正式依赖包与 Tasks 全链路验收使用。"""
    output = BytesIO()
    if suffix == "docx":
        from docx import Document

        document = Document()
        document.add_paragraph("Office budget acceptance")
    elif suffix == "pptx":
        from pptx import Presentation

        document = Presentation()
        document.slides.add_slide(document.slide_layouts[0]).shapes.title.text = "Office budget acceptance"
    else:
        from openpyxl import Workbook

        document = Workbook()
        document.active["A1"] = "Office budget acceptance"
    document.save(output)
    return output.getvalue()


@pytest.mark.parametrize("suffix", ["docx", "pptx", "xlsx"])
@pytest.mark.parametrize("budget", ["zip", "merge"])
def test_real_office_budget_and_artifact_recovery(
    service: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch, suffix: str, budget: str
) -> None:
    """真实超限文件经 API/Router 返回稳定错误，随后正常 Office 仍产出可下载正文。"""
    from docvortex.foundation import ooxml_zip

    client, _ = service
    source = _office_document("xlsx")
    rejected = BytesIO()
    with ZipFile(BytesIO(source)) as original, ZipFile(rejected, "w", ZIP_DEFLATED) as target:
        for info in original.infolist():
            data = original.read(info)
            if budget == "merge" and info.filename == "xl/worksheets/sheet1.xml":
                data = data.replace(b"</worksheet>", b'<mergeCells><mergeCell ref="A1:A100001"/></mergeCells></worksheet>')
            target.writestr(info, data)
        if budget == "zip":
            target.writestr("unreferenced-padding.bin", b"x" * (512 * 1024 + 1))

    with monkeypatch.context() as context:
        if budget == "zip":
            context.setattr(ooxml_zip, "MAX_ENTRY_BYTES", 512 * 1024)
        inputs = [("limited.xlsx", rejected.getvalue())]
        task = test_task_api._submit(client, inputs)
        test_task_api._upload(client, task, inputs)
        failed = test_task_api._terminal(client, task)
    assert failed["status"] == "failed"
    assert failed["files"][0]["error"]["code"] == "resource_limit"
    assert failed["files"][0].get("result_url") is None

    inputs = [(f"normal.{suffix}", _office_document(suffix))]
    task = test_task_api._submit(client, inputs)
    test_task_api._upload(client, task, inputs)
    completed = test_task_api._terminal(client, task)
    assert completed["status"] == "completed"
    response = client.get(completed["files"][0]["result_url"])
    assert response.status_code == 200
    with ZipFile(BytesIO(response.content)) as artifact:
        markdown = [artifact.read(name).decode() for name in artifact.namelist() if name.endswith(".md")]
    assert any("Office budget acceptance" in text for text in markdown)
