"""验证 Doclib 只通过 DocVortex 公共接口读取并映射属性。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import docvortex
import pytest
from docvortex.errors import DocumentError
from docvortex.result import Diagnostic, MetadataResult
from docvortex.schema import DocumentMetadata, DocumentProperties, Producer

from mineru.doclib.core import file_io


def metadata_result() -> MetadataResult:
    """构造超长和多值属性，检查数据库映射不反向修改源对象。"""
    return MetadataResult(
        DocumentMetadata(
            file_suffix="pdf",
            producer=Producer(name="docvortex", version="0.2.5"),
            document=DocumentProperties(
                title="标题" * 400,
                authors=["Alice", "Bob"],
                subject="Subject",
                keywords=["alpha", "beta"],
                languages=["zh-CN", "en"],
                page_count=7,
                page_count_kind="physical",
            ),
        )
    )


def test_extract_metadata_delegates_and_limits_only_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """真实的提取职责交给 DocVortex，字符串截断只在 Doclib 映射发生。"""
    extracted = metadata_result()
    calls: list[str] = []

    def extract(source: str) -> MetadataResult:
        """替换公共接口，确认调用原始路径并传递完整结果。"""
        calls.append(source)
        return extracted

    monkeypatch.setattr(docvortex, "extract_metadata", extract)
    result = asyncio.run(file_io.extract_metadata("sample.pdf"))
    assert calls == ["sample.pdf"]
    assert result["page_count"] == 7
    assert result["author"] == "Alice; Bob"
    assert result["keywords"] == "alpha, beta"
    assert result["language"] == "zh-CN"
    assert len(result["title"]) == 500
    assert len(extracted.metadata.document.title) == 800
    assert not hasattr(file_io, "PDFDocument")


def test_extract_metadata_open_error_preserves_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入错误映射到 Doclib 稳定错误，不把失败伪装成空属性。"""

    def extract(source: str) -> MetadataResult:
        """模拟共享引擎无法打开文件。"""
        raise DocumentError("open_failed", "cannot open document")

    monkeypatch.setattr(docvortex, "extract_metadata", extract)
    with pytest.raises(file_io.MetadataExtractionError) as exc:
        asyncio.run(file_io.extract_metadata("sample.pdf"))
    assert exc.value.code == "open_failed"


def test_partial_metadata_keeps_values_and_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    """可选属性失败时保留页数和已有标题，错误由入库层记录。"""
    extracted = metadata_result()
    extracted.diagnostics = (Diagnostic("read_metadata_failed", "bad XMP"),)

    def extract(source: str) -> MetadataResult:
        """返回部分成功的属性提取结果。"""
        return extracted

    monkeypatch.setattr(docvortex, "extract_metadata", extract)
    result = asyncio.run(file_io.extract_metadata("sample.pdf"))
    assert result["page_count"] == 7
    assert result["title"]
    assert result["error_code"] == "read_metadata_failed"
    assert result["error_msg"] == "bad XMP"


@pytest.mark.parametrize("suffix", ["doc", "docx", "rtf"])
def test_declared_pages_do_not_change_reflow_scheduling(suffix: str) -> None:
    """源软件声明页数仍随 JSON 保留，调度维持单逻辑文档口径。"""
    extracted = metadata_result()
    extracted.metadata.file_suffix = suffix
    extracted.metadata.document.page_count_kind = "declared"
    assert file_io.metadata_to_doclib(extracted.metadata)["page_count"] == 1
    assert extracted.metadata.document.page_count == 7


def test_text_ingest_does_not_claim_document_metadata(tmp_path: Path) -> None:
    """纯文本沿用 Doclib 原有能力，不扩大 DocVortex 输入类型。"""
    source = tmp_path / "file.txt"
    source.write_text("body")
    assert asyncio.run(file_io.extract_metadata(str(source)))["title"] is None
