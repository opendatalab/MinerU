from __future__ import annotations

from mineru.filetypes import (
    FILE_TYPE_BY_EXTENSION,
    INGESTIBLE_EXTENSIONS,
    MIME_TYPE_BY_EXTENSION,
    OFFICE_EXTENSIONS,
    PARSEABLE_EXTENSIONS,
    is_flash_only_parse_extension,
)


def test_office_extensions_includes_legacy_binary_formats() -> None:
    assert {"doc", "docx", "ppt", "pptx", "xls", "xlsx"} <= OFFICE_EXTENSIONS


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
