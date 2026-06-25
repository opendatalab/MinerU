from pathlib import Path

import pytest

from mineru.parser import _build_parser
from mineru.parser.html import HtmlParser
from mineru.parser.office import XlsxParser
from mineru.parser.pdf import PdfFlashParser


def test_build_parser_prefers_html_extension_when_content_guess_is_plain_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html_file = tmp_path / "sample.html"
    html_file.write_text("<div>hello</div>", encoding="utf-8")

    monkeypatch.setattr("mineru.utils.guess_suffix_or_lang.guess_suffix_by_path", lambda path: "txt")

    parser = _build_parser(html_file, tier="flash")

    assert isinstance(parser, HtmlParser)


def test_build_parser_prefers_xlsx_extension_when_content_guess_is_legacy_doc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xlsx_file = tmp_path / "legacy.xlsx"
    xlsx_file.write_bytes(b"\xd0\xcf\x11\xe0")

    monkeypatch.setattr("mineru.utils.guess_suffix_or_lang.guess_suffix_by_path", lambda path: "doc")

    parser = _build_parser(xlsx_file, tier="flash")

    assert isinstance(parser, XlsxParser)


def test_build_parser_rejects_unsupported_suffix_instead_of_falling_back_to_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_file = tmp_path / "sample.txt"
    text_file.write_text("hello", encoding="utf-8")

    monkeypatch.setattr("mineru.utils.guess_suffix_or_lang.guess_suffix_by_path", lambda path: "txt")

    with pytest.raises(ValueError, match="Unsupported file type"):
        _build_parser(text_file, tier="flash")


def test_build_parser_keeps_pdf_extension_routing_when_content_guess_is_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF")

    monkeypatch.setattr("mineru.utils.guess_suffix_or_lang.guess_suffix_by_path", lambda path: "html")

    parser = _build_parser(pdf_file, tier="flash")

    assert isinstance(parser, PdfFlashParser)
