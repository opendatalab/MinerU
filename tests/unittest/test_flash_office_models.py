import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from mineru.backend.office.docx_analyze import office_docx_analyze
from mineru.backend.office.pptx_analyze import office_pptx_analyze
from mineru.backend.office.xlsx_analyze import office_xlsx_analyze
from mineru.model.flash.docx.main import convert_binary as convert_docx
from mineru.model.flash.pptx.main import convert_binary as convert_pptx
from mineru.model.flash.xlsx.main import convert_binary as convert_xlsx
from mineru.parser.office import DocxParser, PptxParser, XlsxParser


_OFFICE_SAMPLE_DIR = Path(__file__).resolve().parents[2] / "demo" / "office_docs"


@pytest.mark.parametrize(
    ("suffix", "converter", "expected_pages", "expected_module"),
    [
        ("docx", convert_docx, 3, "mineru.model.flash.docx.main"),
        ("pptx", convert_pptx, 6, "mineru.model.flash.pptx.main"),
        ("xlsx", convert_xlsx, 3, "mineru.model.flash.xlsx.main"),
    ],
)
def test_flash_office_model_conversion(
    suffix: str,
    converter: Callable[[BinaryIO], list[dict[str, Any]]],
    expected_pages: int,
    expected_module: str,
) -> None:
    """验证迁移后的 Office 模型入口与真实样例转换结果保持不变。"""

    sample_path = _OFFICE_SAMPLE_DIR / f"{suffix}_01.{suffix}"
    with sample_path.open("rb") as stream:
        pages = converter(stream)

    assert converter.__module__ == expected_module
    assert len(pages) == expected_pages


def test_office_parsers_bind_existing_analyzers() -> None:
    """验证三个 Office Parser 仍绑定原有的后端分析入口。"""

    assert DocxParser()._analyze_fn is office_docx_analyze
    assert PptxParser()._analyze_fn is office_pptx_analyze
    assert XlsxParser()._analyze_fn is office_xlsx_analyze


@pytest.mark.parametrize(
    "module_name",
    ["mineru.model.docx", "mineru.model.pptx", "mineru.model.xlsx"],
)
def test_legacy_office_model_paths_are_removed(module_name: str) -> None:
    """验证迁移后不再暴露旧的 Office 模型包路径。"""

    assert importlib.util.find_spec(module_name) is None
