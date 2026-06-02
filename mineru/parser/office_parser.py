# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any

from .base import DocumentParser
from .parse_result import ParseResult
from .types import PageInfo


class OfficeBaseParser(DocumentParser, ABC):
    """Abstract base for DOCX, PPTX, XLSX parsers.

    Subclasses supply the analyze function and file suffix.
    """

    _analyze_fn: Any

    _suffix: str

    def parse(self, path: str | Path) -> ParseResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        file_name = path.stem
        file_bytes = path.read_bytes()
        middle_json, model_output = self._analyze_fn(file_bytes, image_writer=None)
        return self._build_result(middle_json, file_name, model_output)

    def _build_result(
        self,
        middle_json: dict,
        file_name: str,
        model_output: Any = None,
    ) -> ParseResult:
        from ..version import __version__

        pages = [PageInfo.from_dict(p) for p in middle_json["pdf_info"]]
        return ParseResult(
            pages=pages,
            _backend=middle_json.get("_backend", "office"),
            _version_name=__version__,
            _file_name=file_name,
            _model_output=model_output if self.return_model_output else None,
        )


class DocxParser(OfficeBaseParser):
    _suffix = "docx"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from ..backend.office.docx_analyze import office_docx_analyze

        self._analyze_fn = office_docx_analyze


class PptxParser(OfficeBaseParser):
    _suffix = "pptx"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from ..backend.office.pptx_analyze import office_pptx_analyze

        self._analyze_fn = office_pptx_analyze


class XlsxParser(OfficeBaseParser):
    _suffix = "xlsx"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from ..backend.office.xlsx_analyze import office_xlsx_analyze

        self._analyze_fn = office_xlsx_analyze
