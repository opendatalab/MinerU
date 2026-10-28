"""Regression coverage for isolated ZIP export staging directories."""

from __future__ import annotations

import base64
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Literal

import pytest
from PIL import Image

from mineru.kit.common import save_parse_result
from mineru.parser.base import ParseResult
from mineru.parser.writer import DataWriter
from mineru.types import ImageBlock, ImageBodyBlock, MiddleJson, PageInfo


@pytest.fixture
def image_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (2, 2), "red").save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def parse_result(image_bytes: bytes) -> ParseResult:
    return ParseResult(
        MiddleJson(
            pages=[
                PageInfo(
                    page_idx=0,
                    blocks=[
                        ImageBlock(
                            type="image",
                            index=0,
                            content=[
                                ImageBodyBlock(
                                    type="image_body",
                                    index=0,
                                    content="",
                                    image_base64="data:image/png;base64," + base64.b64encode(image_bytes).decode(),
                                )
                            ],
                        )
                    ],
                )
            ],
            is_full_document=True,
            metadata={"file_suffix": "docx", "producer": {"name": "mineru", "version": "test"}},
        )
    )


@pytest.mark.parametrize("sibling_kind", ["file", "directory"])
def test_zip_export_preserves_existing_sibling(
    tmp_path: Path,
    parse_result: ParseResult,
    image_bytes: bytes,
    sibling_kind: Literal["file", "directory"],
) -> None:
    """A same-stem hidden sibling must neither enter the ZIP nor be removed."""
    dest = tmp_path / "report.zip"
    sibling = tmp_path / ".report"
    if sibling_kind == "directory":
        sibling.mkdir()
        sentinel = sibling / "private.txt"
    else:
        sentinel = sibling
    sentinel.write_bytes(b"unrelated user data")

    save_parse_result(parse_result, dest, "zip")

    assert sentinel.read_bytes() == b"unrelated user data"
    assert set(tmp_path.iterdir()) == {sibling, dest}
    with zipfile.ZipFile(dest) as archive:
        names = set(archive.namelist())
        images = [name for name in names if name.startswith("images/")]
        assert len(images) == 1
        assert names == {"markdown.md", "middle_json.json", "structured_content.json", *images}
        assert archive.read(images[0]) == image_bytes
        assert images[0] in archive.read("markdown.md").decode()


def test_zip_export_cleans_up_after_save_failure(
    tmp_path: Path, parse_result: ParseResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial result files must be removed when result serialization fails."""
    dest = tmp_path / "report.zip"

    def fail_save(self: ParseResult, writer: DataWriter) -> None:
        writer.write("images/partial.png", b"partial image")
        raise RuntimeError("save failed")

    monkeypatch.setattr(ParseResult, "save", fail_save)

    with pytest.raises(RuntimeError, match="save failed"):
        save_parse_result(parse_result, dest, "zip")

    assert list(tmp_path.iterdir()) == []


def test_zip_export_cleans_up_after_archive_failure(
    tmp_path: Path, parse_result: ParseResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An archive write error must not leave the staged document or images behind."""
    dest = tmp_path / "report.zip"

    def fail_write(self: zipfile.ZipFile, filename: Path, arcname: str) -> None:
        raise OSError("archive write failed")

    monkeypatch.setattr(zipfile.ZipFile, "write", fail_write)

    with pytest.raises(OSError, match="archive write failed"):
        save_parse_result(parse_result, dest, "zip")

    assert not any(path != dest for path in tmp_path.iterdir())
