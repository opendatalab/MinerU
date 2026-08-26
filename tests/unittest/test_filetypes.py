from __future__ import annotations

from pathlib import Path

from mineru.filetypes import (
    CSV_EXTENSIONS,
    FILE_TYPE_BY_EXTENSION,
    FLASH_ONLY_PARSE_EXTENSIONS,
    INGESTIBLE_EXTENSIONS,
    MIME_TYPE_BY_EXTENSION,
    OFFICE_EXTENSIONS,
    PARSEABLE_EXTENSIONS,
    TEXT_EXTENSIONS,
    is_flash_only_parse_extension,
)
from mineru.kit.common import ensure_supported_inputs, expand_input_paths


def test_office_extensions_includes_legacy_binary_formats() -> None:
    assert {"doc", "docx", "ppt", "pptx", "xls", "xlsx", "rtf"} <= OFFICE_EXTENSIONS


def test_legacy_office_extensions_are_parseable_and_ingestible() -> None:
    for ext in ("doc", "ppt", "xls"):
        assert ext in PARSEABLE_EXTENSIONS
        assert ext in INGESTIBLE_EXTENSIONS
        assert is_flash_only_parse_extension(ext)


def test_file_type_by_extension_maps_legacy_office_to_self() -> None:
    assert FILE_TYPE_BY_EXTENSION["doc"] == "doc"
    assert FILE_TYPE_BY_EXTENSION["ppt"] == "ppt"
    assert FILE_TYPE_BY_EXTENSION["xls"] == "xls"


def test_mime_type_by_extension_covers_legacy_office() -> None:
    assert MIME_TYPE_BY_EXTENSION["doc"] == "application/msword"
    assert MIME_TYPE_BY_EXTENSION["ppt"] == "application/vnd.ms-powerpoint"
    assert MIME_TYPE_BY_EXTENSION["xls"] == "application/vnd.ms-excel"
    assert MIME_TYPE_BY_EXTENSION["rtf"] == "application/rtf"


def test_rtf_is_a_flash_only_parseable_office_type() -> None:
    """验证 RTF 自动进入所有本地 Flash 与 doclib 文件集合。"""
    assert "rtf" in OFFICE_EXTENSIONS
    assert "rtf" in FLASH_ONLY_PARSE_EXTENSIONS
    assert "rtf" in PARSEABLE_EXTENSIONS
    assert "rtf" in INGESTIBLE_EXTENSIONS
    assert FILE_TYPE_BY_EXTENSION["rtf"] == "rtf"
    assert is_flash_only_parse_extension("rtf")


def test_csv_is_a_flash_only_parseable_type_instead_of_plain_text() -> None:
    """验证 CSV 进入结构化 flash 解析集合并保留独立 MIME 和文件类型。"""
    assert CSV_EXTENSIONS == frozenset({"csv"})
    assert "csv" in FLASH_ONLY_PARSE_EXTENSIONS
    assert "csv" in PARSEABLE_EXTENSIONS
    assert "csv" in INGESTIBLE_EXTENSIONS
    assert "csv" not in TEXT_EXTENSIONS
    assert FILE_TYPE_BY_EXTENSION["csv"] == "csv"
    assert MIME_TYPE_BY_EXTENSION["csv"] == "text/csv"
    assert is_flash_only_parse_extension("csv")


def test_mineru_kit_discovers_and_accepts_csv_inputs(tmp_path: Path) -> None:
    """验证 mineru-kit 的目录展开和输入校验自动继承 CSV 解析集合。"""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("name,age\nAlice,30\n", encoding="utf-8")

    expanded = expand_input_paths([str(tmp_path)])

    assert expanded == [csv_path]
    ensure_supported_inputs(expanded)
