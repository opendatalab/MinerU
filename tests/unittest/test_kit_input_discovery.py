"""Keep Office lock sidecars out of directory-based kit parsing."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile
from typing import Any

import pytest
from typer.testing import CliRunner

from mineru.filetypes import OFFICE_EXTENSIONS
from mineru.kit.commands import parse
from mineru.kit.common import expand_input_paths
from mineru.kit.main import app


@pytest.mark.parametrize("extension", sorted(OFFICE_EXTENSIONS))
@pytest.mark.parametrize("uppercase", [False, True])
def test_directory_discovery_skips_office_lock_files(tmp_path: Path, extension: str, uppercase: bool) -> None:
    suffix = extension.upper() if uppercase else extension
    document = tmp_path / f"report.{suffix}"
    lock = tmp_path / f"~$report.{suffix}"
    document.touch()
    lock.touch()

    assert expand_input_paths([str(tmp_path)]) == [document]
    assert lock.exists()


def test_directory_discovery_preserves_other_inputs_and_order(tmp_path: Path) -> None:
    """Do not treat every tilde-prefixed or hidden file as a lock, or recurse."""
    directory = tmp_path / "~$documents"
    directory.mkdir()
    names = ["z.docx", "a.csv", ".hidden.docx", "~$data.csv", "~$figure.pdf", "~notes.docx", "draft~$report.docx"]
    documents = [directory / name for name in names]
    for document in documents:
        document.touch()
    (directory / "~$report.docx").touch()
    (directory / "unsupported.bin").touch()
    nested = directory / "nested.docx"
    nested.mkdir()
    (nested / "child.docx").touch()
    explicit = tmp_path / "first.docx"
    explicit.touch()

    assert expand_input_paths([str(explicit), str(directory)]) == [explicit, *sorted(documents)]


def test_explicit_lock_paths_are_not_silently_discarded(tmp_path: Path) -> None:
    lock = tmp_path / "~$report.docx"
    lock.touch()
    missing = tmp_path / "~$missing.docx"

    assert expand_input_paths([str(lock), str(missing)]) == [lock, missing]


@pytest.mark.parametrize("with_lock", [False, True])
def test_parse_directory_with_real_docx_files(tmp_path: Path, with_lock: bool) -> None:
    """Exercise the public CLI and native Office parser without model inference."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures/pr5389/table-links.docx"
    source = tmp_path / "documents"
    source.mkdir()
    # The lock sorts between these documents, so reaching the second is observable.
    for name in ("a.docx", "报告.docx"):
        copyfile(fixture, source / name)
    if with_lock:
        (source / "~$报告.docx").write_bytes(b"simulated Office owner file, not a document")
    output = tmp_path / "output"

    result = CliRunner().invoke(app, ["parse", str(source), "-o", str(output)])

    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in output.iterdir()) == ["a.md", "报告.md"]
    assert (output / "a.md").read_bytes() == (output / "报告.md").read_bytes()
    assert "https://example.org/one" in (output / "报告.md").read_text(encoding="utf-8")


def test_remote_directory_parse_does_not_send_office_lock_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "documents"
    source.mkdir()
    document = source / "report.docx"
    document.touch()
    (source / "~$report.docx").touch()
    received: list[Path] = []

    class Result:
        def markdown(self) -> str:
            return "document content"

    class Parser:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def parse(self, path: Path, *, page_range: str) -> Result:
            received.append(path)
            return Result()

    monkeypatch.setattr(parse, "MinerUApiParser", Parser)
    output = tmp_path / "output"
    result = CliRunner().invoke(app, ["parse", str(source), "-o", str(output), "--remote-url", "https://example.invalid"])

    assert result.exit_code == 0, result.output
    assert received == [document]
    assert sorted(path.name for path in output.iterdir()) == ["report.md"]


@pytest.mark.parametrize("with_lock", [False, True])
def test_empty_directory_keeps_no_inputs_error(tmp_path: Path, with_lock: bool) -> None:
    source = tmp_path / "documents"
    source.mkdir()
    if with_lock:
        (source / "~$report.docx").touch()
    output = tmp_path / "output"

    result = CliRunner().invoke(app, ["parse", str(source), "-o", str(output)])

    assert result.exit_code == 1
    assert "No input files found." in " ".join(result.output.split())
    assert not output.exists()


def test_directory_parse_still_stops_on_an_ordinary_corrupt_document(tmp_path: Path) -> None:
    source = tmp_path / "documents"
    source.mkdir()
    (source / "a.docx").write_bytes(b"invalid document")
    fixture = Path(__file__).resolve().parents[1] / "fixtures/pr5389/table-links.docx"
    copyfile(fixture, source / "b.docx")
    output = tmp_path / "output"

    result = CliRunner().invoke(app, ["parse", str(source), "-o", str(output)])

    assert result.exit_code == 1
    assert not output.exists()
