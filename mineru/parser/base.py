# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parse_result import ParseResult


class DocumentParser(ABC):
    """Abstract base class for all document parsers.

    Subclasses implement ``parse()`` for a specific document category (PDF, DOCX, PPTX, XLSX).
    """

    def __init__(
        self,
        *,
        backend: str = "hybrid-auto-engine",
        method: str = "auto",
        lang: str = "ch",
        formula_enable: bool = True,
        table_enable: bool = True,
        image_analysis: bool = True,
        server_url: str | None = None,
        start_page_id: int = 0,
        end_page_id: int | None = None,
        return_md: bool = True,
        return_middle_json: bool = True,
        return_model_output: bool = True,
        return_content_list: bool = True,
        return_images: bool = True,
        output_dir: str | Path = "./output",
    ):
        self.backend = backend
        self.method = method
        self.lang = lang
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self.image_analysis = image_analysis
        self.server_url = server_url
        self.start_page_id = start_page_id
        self.end_page_id = end_page_id
        self.return_md = return_md
        self.return_middle_json = return_middle_json
        self.return_model_output = return_model_output
        self.return_content_list = return_content_list
        self.return_images = return_images
        self.output_dir = Path(output_dir)
        self._closed = False

    @abstractmethod
    def parse(self, path: str | Path) -> ParseResult:
        """Parse a document and return structured results."""

    async def parse_async(self, path: str | Path) -> ParseResult:
        """Asynchronously parse a document.

        The default implementation delegates to ``parse()`` via ``asyncio.to_thread``.
        Subclasses may override for native async support.
        """
        import asyncio

        return await asyncio.to_thread(self.parse, path)

    def parse_batch(self, paths: list[str | Path]) -> list[ParseResult]:
        """Parse multiple documents synchronously.

        The default implementation calls ``parse()`` for each path in order.
        Subclasses may override for batch-optimized execution.
        """
        return [self.parse(p) for p in paths]

    async def parse_batch_async(self, paths: list[str | Path]) -> list[ParseResult]:
        """Parse multiple documents asynchronously.

        The default implementation calls ``parse_async()`` concurrently for all paths.
        Subclasses may override for batch-optimized execution.
        """
        import asyncio

        return await asyncio.gather(*(self.parse_async(p) for p in paths))

    def close(self) -> None:
        """Release resources held by this parser instance.

        After ``close()``, the instance must not be reused.
        The default implementation is a no-op; subclasses may override.
        """
        self._closed = True

    def __enter__(self) -> "DocumentParser":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
