from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from mineru.errors import InvalidRequestError
from mineru.filetypes import (
    CSV_EXTENSIONS,
    EPUB_EXTENSIONS,
    FILE_TYPE_BY_EXTENSION,
    FLASH_ONLY_PARSE_EXTENSIONS,
    HTML_EXTENSIONS,
    INGESTIBLE_EXTENSIONS,
    MIME_TYPE_BY_EXTENSION,
    ODF_EXTENSIONS,
    OFD_EXTENSIONS,
    OFFICE_EXTENSIONS,
    PAGE_RANGE_PARSE_EXTENSIONS,
    PARSEABLE_EXTENSIONS,
    TEXT_EXTENSIONS,
    is_flash_only_parse_extension,
    is_page_range_parse_extension,
    is_tiered_parse_extension,
    mime_type_for_extension,
    normalize_parse_extension,
    validate_parse_filename,
)
from mineru.kit.common import ensure_supported_inputs, expand_input_paths
from mineru.parser import parse


@pytest.mark.parametrize("as_path", [False, True])
@pytest.mark.parametrize(
    "template",
    [
        "{ext}",
        ".{ext}",
        "in/t.{ext}",
        "../in/t.{ext}",
        "./in/t.{ext}",
        ".data/t.{ext}",
        ".t.{ext}",
        "/absolute/t.{ext}",
        r"..\in\t.{ext}",
    ],
)
@pytest.mark.parametrize(
    "ext,mime,flash",
    [
        ("CSV", "text/csv", True),
        ("DOCX", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", True),
        ("PDF", "application/pdf", False),
        ("JPG", "image/jpeg", False),
    ],
)
def test_path_extension_controls_format_capabilities(as_path: bool, template: str, ext: str, mime: str, flash: bool) -> None:
    """相对路径、隐藏名称和裸扩展名共享后缀、档位、分页与 MIME 判定。"""
    text = template.format(ext=ext)
    value = Path(text) if as_path else text
    assert normalize_parse_extension(value) == ext.lower()
    assert is_flash_only_parse_extension(value) is flash
    assert is_tiered_parse_extension(value) is (not flash)
    assert is_page_range_parse_extension(value) is (ext == "PDF")
    assert mime_type_for_extension(value) == mime


@pytest.mark.parametrize(
    "name",
    ["report.csv", ".report.csv", "report..csv", "报告 2026.CSV", " report.csv ", "%2e%2e%2freport.csv", "README", "a" * 1024],
)
def test_parse_filename_preserves_valid_basename(name: str) -> None:
    """纯文件名校验保留中文、空格、隐藏名称和百分号，不改写输入或限制扩展名。"""
    assert validate_parse_filename(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "a" * 1025,
        "../t.csv",
        "in/t.csv",
        "/t.csv",
        r"..\t.csv",
        r"C:\in\t.csv",
        "C:t.csv",
        r"\\server\share\t.csv",
        "t.csv:stream",
    ],
)
def test_parse_filename_rejects_paths_and_invalid_lengths(name: str) -> None:
    """API 文件名独立拒绝两种路径分隔符、盘符、备用数据流和非法长度。"""
    with pytest.raises(ValueError):
        validate_parse_filename(name)


def test_parse_filename_rejects_all_c0_and_c1_controls() -> None:
    """覆盖 C0、DEL 和 C1 控制字符，避免仅拦截 NUL 或换行。"""
    for codepoint in (*range(0x20), *range(0x7F, 0xA0)):
        with pytest.raises(ValueError):
            validate_parse_filename(f"report{chr(codepoint)}.csv")


def test_office_extensions_includes_legacy_binary_formats() -> None:
    assert {"doc", "docx", "ppt", "pptx", "xls", "xlsx", "rtf"} <= OFFICE_EXTENSIONS


def test_only_pdf_supports_page_range_parsing() -> None:
    """验证分页能力与图片的质量 tier/remote 能力相互独立。"""
    assert PAGE_RANGE_PARSE_EXTENSIONS == frozenset({"pdf"})
    assert is_page_range_parse_extension("sample.pdf")
    assert not is_page_range_parse_extension("sample.png")
    assert not is_page_range_parse_extension("sample.epub")


def test_image_parser_rejects_explicit_page_range(tmp_path: Path) -> None:
    """验证图片在转为内部 PDF 之前按整文件契约拒绝 page_range。"""
    source = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), "white").save(source)

    with pytest.raises(InvalidRequestError) as exc_info:
        parse(source, tier="flash", page_range="1")

    assert exc_info.value.code == "page_range_invalid"
    assert exc_info.value.param == "page_range"


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


def test_odf_extensions_are_flash_only_office_types() -> None:
    """验证三个 ODF 后缀进入本地 Office、CLI 和 doclib 派生集合。"""
    assert ODF_EXTENSIONS == frozenset({"odt", "ods", "odp"})
    for ext in ODF_EXTENSIONS:
        assert ext in OFFICE_EXTENSIONS
        assert ext in FLASH_ONLY_PARSE_EXTENSIONS
        assert ext in PARSEABLE_EXTENSIONS
        assert ext in INGESTIBLE_EXTENSIONS
        assert FILE_TYPE_BY_EXTENSION[ext] == ext
        assert is_flash_only_parse_extension(ext)
    assert MIME_TYPE_BY_EXTENSION["odt"] == "application/vnd.oasis.opendocument.text"
    assert MIME_TYPE_BY_EXTENSION["ods"] == "application/vnd.oasis.opendocument.spreadsheet"
    assert MIME_TYPE_BY_EXTENSION["odp"] == "application/vnd.oasis.opendocument.presentation"


def test_csv_is_a_flash_only_parseable_type_instead_of_plain_text() -> None:
    """验证 CSV/TSV 进入结构化 flash 解析集合并保留独立 MIME 和文件类型。"""
    assert CSV_EXTENSIONS == frozenset({"csv", "tsv"})
    for ext in CSV_EXTENSIONS:
        assert ext in FLASH_ONLY_PARSE_EXTENSIONS
        assert ext in PARSEABLE_EXTENSIONS
        assert ext in INGESTIBLE_EXTENSIONS
        assert ext not in TEXT_EXTENSIONS
        assert FILE_TYPE_BY_EXTENSION[ext] == ext
        assert is_flash_only_parse_extension(ext)
    assert MIME_TYPE_BY_EXTENSION["csv"] == "text/csv"
    assert MIME_TYPE_BY_EXTENSION["tsv"] == "text/tab-separated-values"


def test_epub_is_a_flash_only_parseable_e_book_type() -> None:
    """验证 EPUB 进入本地解析、doclib 与 MIME 派生集合，但不伪装成 Office。"""
    assert EPUB_EXTENSIONS == frozenset({"epub"})
    assert "epub" not in OFFICE_EXTENSIONS
    assert "epub" in FLASH_ONLY_PARSE_EXTENSIONS
    assert "epub" in PARSEABLE_EXTENSIONS
    assert "epub" in INGESTIBLE_EXTENSIONS
    assert FILE_TYPE_BY_EXTENSION["epub"] == "epub"
    assert MIME_TYPE_BY_EXTENSION["epub"] == "application/epub+zip"
    assert is_flash_only_parse_extension("epub")


def test_html_extensions_are_canonical_flash_only_inputs() -> None:
    """验证 HTML/HTM 在所有入口归一为单一 html 文件类型。"""
    assert HTML_EXTENSIONS == frozenset({"html", "htm", "shtml"})
    for ext in HTML_EXTENSIONS:
        assert ext in FLASH_ONLY_PARSE_EXTENSIONS
        assert ext in PARSEABLE_EXTENSIONS
        assert ext in INGESTIBLE_EXTENSIONS
        assert FILE_TYPE_BY_EXTENSION[ext] == "html"
        assert MIME_TYPE_BY_EXTENSION[ext] == "text/html"
        assert is_flash_only_parse_extension(ext)


def test_ofd_is_an_independent_fixed_layout_flash_type() -> None:
    """验证 OFD 进入现代 Flash/Doclib 集合但不归入 Office。"""
    assert OFD_EXTENSIONS == frozenset({"ofd"})
    assert "ofd" not in OFFICE_EXTENSIONS
    assert "ofd" in FLASH_ONLY_PARSE_EXTENSIONS
    assert "ofd" in PARSEABLE_EXTENSIONS
    assert "ofd" in INGESTIBLE_EXTENSIONS
    assert FILE_TYPE_BY_EXTENSION["ofd"] == "ofd"
    assert MIME_TYPE_BY_EXTENSION["ofd"] == "application/ofd"
    assert is_flash_only_parse_extension("ofd")


def test_additional_text_extensions_are_ingestible_but_not_parseable() -> None:
    """新增文本扩展进入 doclib ingest/扫描，但不进入结构化解析集合。"""
    for ext in ("mdx", "latex", "adoc", "asciidoc"):
        assert ext in TEXT_EXTENSIONS
        assert ext in INGESTIBLE_EXTENSIONS
        assert ext not in PARSEABLE_EXTENSIONS
    assert FILE_TYPE_BY_EXTENSION["mdx"] == "markdown"
    assert FILE_TYPE_BY_EXTENSION["latex"] == "tex"
    assert FILE_TYPE_BY_EXTENSION["adoc"] == "asciidoc"
    assert FILE_TYPE_BY_EXTENSION["asciidoc"] == "asciidoc"


def test_mineru_kit_discovers_and_accepts_csv_inputs(tmp_path: Path) -> None:
    """验证 mineru-kit 的目录展开和输入校验自动继承 CSV 解析集合。"""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("name,age\nAlice,30\n", encoding="utf-8")

    expanded = expand_input_paths([str(tmp_path)])

    assert expanded == [csv_path]
    ensure_supported_inputs(expanded)
