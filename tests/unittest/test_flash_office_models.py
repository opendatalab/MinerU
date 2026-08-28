import importlib.util
import subprocess
import sys
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO
from unittest.mock import MagicMock

import pytest

import mineru.model.flash as flash_models
import mineru.model.flash.models as flat_model_module
from mineru.model.flash import (
    DocModel,
    DocxModel,
    EpubModel,
    OdpModel,
    OdsModel,
    OdtModel,
    OfdModel,
    PdfModel,
    PptModel,
    PptxModel,
    RtfModel,
    XlsModel,
    XlsxModel,
)
from mineru.model.flash.office.doc import doc_converter as doc_converter_module
from mineru.model.flash.office.docx import docx_converter as docx_converter_module
from mineru.model.flash.office.docx import main as docx_main
from mineru.model.flash.office.odf import converters as odf_converter_module
from mineru.model.flash.office.pptx import main as pptx_main
from mineru.model.flash.office.pptx import pptx_converter as pptx_converter_module
from mineru.model.flash.office.ppt import ppt_converter as ppt_converter_module
from mineru.model.flash.office.rtf import converter as rtf_converter_module
from mineru.model.flash.office.xls import xls_converter as xls_converter_module
from mineru.model.flash.office.xlsx import main as xlsx_main
from mineru.model.flash.office.xlsx import xlsx_converter as xlsx_converter_module


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OFFICE_SAMPLE_DIR = _PROJECT_ROOT / "demo" / "office_docs"


@pytest.mark.parametrize(
    ("suffix", "model_class", "convert_binary", "expected_pages"),
    [
        ("docx", DocxModel, docx_main.convert_binary, 3),
        ("pptx", PptxModel, pptx_main.convert_binary, 6),
        ("xlsx", XlsxModel, xlsx_main.convert_binary, 3),
    ],
)
def test_flash_office_model_conversion(
    suffix: str,
    model_class: type[Any],
    convert_binary: Callable[[BinaryIO], list[list[dict[str, Any]]]],
    expected_pages: int,
) -> None:
    """验证独立 Office 模型与兼容函数的真实样例结果保持不变。"""

    sample_path = _OFFICE_SAMPLE_DIR / f"{suffix}_01.{suffix}"
    with sample_path.open("rb") as stream:
        model_pages = model_class().predict(stream)
        assert not stream.closed

    with sample_path.open("rb") as stream:
        compatible_pages = convert_binary(stream)
        assert not stream.closed

    assert len(model_pages) == expected_pages
    assert len(compatible_pages) == expected_pages


@pytest.mark.parametrize(
    ("model_class", "converter_module", "converter_name"),
    [
        (DocModel, doc_converter_module, "DocConverter"),
        (DocxModel, docx_converter_module, "DocxConverter"),
        (PptxModel, pptx_converter_module, "PptxConverter"),
        (PptModel, ppt_converter_module, "PptConverter"),
        (XlsModel, xls_converter_module, "XlsConverter"),
        (XlsxModel, xlsx_converter_module, "XlsxConverter"),
        (RtfModel, rtf_converter_module, "RtfConverter"),
        (OdtModel, odf_converter_module, "OdtConverter"),
        (OdsModel, odf_converter_module, "OdsConverter"),
        (OdpModel, odf_converter_module, "OdpConverter"),
    ],
)
def test_office_model_creates_converter_per_prediction(
    monkeypatch: pytest.MonkeyPatch,
    model_class: type[Any],
    converter_module: ModuleType,
    converter_name: str,
) -> None:
    """验证 Office 模型每次预测使用独立 Converter，且不关闭输入流。"""

    first_converter = MagicMock()
    first_converter.pages = [[{"content": "first"}]]
    second_converter = MagicMock()
    second_converter.pages = [[{"content": "second"}]]
    converter_factory = MagicMock(side_effect=[first_converter, second_converter])
    monkeypatch.setattr(converter_module, converter_name, converter_factory)

    model = model_class()
    first_stream = BytesIO(b"first")
    second_stream = BytesIO(b"second")

    assert model.predict(first_stream) is first_converter.pages
    assert model.predict(second_stream) is second_converter.pages
    assert converter_factory.call_count == 2
    first_converter.convert.assert_called_once_with(first_stream)
    second_converter.convert.assert_called_once_with(second_stream)
    assert not first_stream.closed
    assert not second_stream.closed


@pytest.mark.parametrize(
    ("main_module", "model_name"),
    [
        (docx_main, "DocxModel"),
        (pptx_main, "PptxModel"),
        (xlsx_main, "XlsxModel"),
    ],
)
def test_convert_binary_delegates_to_document_model(
    monkeypatch: pytest.MonkeyPatch,
    main_module: ModuleType,
    model_name: str,
) -> None:
    """验证旧 convert_binary 函数仅负责转发给对应文档模型。"""

    expected_pages = [[{"content": "model"}]]
    model = MagicMock()
    model.predict.return_value = expected_pages
    model_factory = MagicMock(return_value=model)
    monkeypatch.setattr(main_module, model_name, model_factory)
    stream = BytesIO(b"office")

    assert main_module.convert_binary(stream) is expected_pages
    model_factory.assert_called_once_with()
    model.predict.assert_called_once_with(stream)


@pytest.mark.parametrize(
    ("main_module", "suffix"),
    [
        (docx_main, "docx"),
        (pptx_main, "pptx"),
        (xlsx_main, "xlsx"),
    ],
)
def test_convert_path_delegates_to_binary_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    main_module: ModuleType,
    suffix: str,
) -> None:
    """验证旧 convert_path 函数打开文件后转发给二进制兼容入口。"""

    sample_path = tmp_path / f"sample.{suffix}"
    sample_path.write_bytes(b"office")
    expected_pages = [[{"content": "model"}]]
    binary_helper = MagicMock(return_value=expected_pages)
    monkeypatch.setattr(main_module, "convert_binary", binary_helper)

    assert main_module.convert_path(str(sample_path)) is expected_pages
    file_stream = binary_helper.call_args.args[0]
    assert file_stream.closed


def test_models_are_exported_from_flash_root() -> None:
    """验证全部模型统一由 Flash 根包公开。"""

    assert flash_models.__all__ == [
        "PdfModel",
        "CsvModel",
        "EpubModel",
        "HtmlModel",
        "OfdModel",
        "RtfModel",
        "DocModel",
        "DocxModel",
        "PptModel",
        "PptxModel",
        "XlsModel",
        "XlsxModel",
        "OdtModel",
        "OdsModel",
        "OdpModel",
    ]
    assert PdfModel is flat_model_module.PdfModel
    assert RtfModel is flat_model_module.RtfModel
    assert DocModel is flat_model_module.DocModel
    assert DocxModel is flat_model_module.DocxModel
    assert EpubModel is flat_model_module.EpubModel
    assert OfdModel is flat_model_module.OfdModel
    assert PptxModel is flat_model_module.PptxModel
    assert PptModel is flat_model_module.PptModel
    assert XlsModel is flat_model_module.XlsModel
    assert XlsxModel is flat_model_module.XlsxModel
    assert OdtModel is flat_model_module.OdtModel
    assert OdsModel is flat_model_module.OdsModel
    assert OdpModel is flat_model_module.OdpModel


@pytest.mark.parametrize(
    ("package_name", "model_name"),
    [
        ("mineru.model.flash.office.doc", "DocModel"),
        ("mineru.model.flash.office.docx", "DocxModel"),
        ("mineru.model.flash.office.pptx", "PptxModel"),
        ("mineru.model.flash.office.ppt", "PptModel"),
        ("mineru.model.flash.office.xls", "XlsModel"),
        ("mineru.model.flash.office.xlsx", "XlsxModel"),
        ("mineru.model.flash.office.rtf", "RtfModel"),
        ("mineru.model.flash.office.odf", "OdtModel"),
        ("mineru.model.flash.office.odf", "OdsModel"),
        ("mineru.model.flash.office.odf", "OdpModel"),
    ],
)
def test_office_subpackages_do_not_export_models(package_name: str, model_name: str) -> None:
    """验证 Office 子包不再导出模型类或保留独立 model 模块。"""

    package = importlib.import_module(package_name)
    assert not hasattr(package, model_name)
    assert importlib.util.find_spec(f"{package_name}.model") is None


def test_importing_pdf_model_does_not_load_office_converters() -> None:
    """验证纯 PDF 模型导入不会提前加载任何 Office Converter。"""

    script = "\n".join(
        [
            "import sys",
            "from mineru.model.flash import PdfModel",
            "assert PdfModel.__name__ == 'PdfModel'",
            "assert 'mineru.model.flash.office.docx.docx_converter' not in sys.modules",
            "assert 'mineru.model.flash.office.doc.doc_converter' not in sys.modules",
            "assert 'mineru.model.flash.office.pptx.pptx_converter' not in sys.modules",
            "assert 'mineru.model.flash.office.ppt.ppt_converter' not in sys.modules",
            "assert 'mineru.model.flash.office.xls.xls_converter' not in sys.modules",
            "assert 'olefile' not in sys.modules",
            "assert 'mineru.model.flash.office.xlsx.xlsx_converter' not in sys.modules",
            "assert 'mineru.model.flash.office.rtf.converter' not in sys.modules",
            "assert 'mineru.model.flash.office.odf.converters' not in sys.modules",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "module_name",
    [
        "mineru.model.doc",
        "mineru.model.docx",
        "mineru.model.ppt",
        "mineru.model.pptx",
        "mineru.model.xls",
        "mineru.model.xlsx",
        "mineru.model.odt",
        "mineru.model.ods",
        "mineru.model.odp",
    ],
)
def test_legacy_office_model_paths_are_removed(module_name: str) -> None:
    """验证迁移后不再暴露旧的 Office 模型包路径。"""

    assert importlib.util.find_spec(module_name) is None
