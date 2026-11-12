"""Exercise output path validation through the CLI with real native CSV inputs."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from mineru.kit.commands import parse
from mineru.kit.common import OUTPUT_FILE_SUFFIXES
from mineru.kit.main import app
from mineru.parser.base import ParseResult


def _csv_inputs(tmp_path: Path, input_kind: str) -> tuple[list[str], list[Path]]:
    """Prepare explicit files or directories containing one or two inputs."""
    source_dir = tmp_path / "documents"
    source_dir.mkdir()
    count = 1 if input_kind in {"single_file", "directory_one"} else 2
    sources = [source_dir / f"document{index}.csv" for index in range(count)]
    for source in sources:
        source.write_text(f"name,value\n{source.stem},1\n", encoding="utf-8")
    inputs = [str(source_dir)] if input_kind.startswith("directory") else [str(source) for source in sources]
    return inputs, sources


@pytest.mark.parametrize("input_kind", ["single_file", "multiple_files", "directory_one", "directory_many"])
@pytest.mark.parametrize("directory_name", ["results", "results.v1", "results.md"])
@pytest.mark.parametrize("output_format", ["markdown", "middle_json", "zip"])
def test_parse_accepts_existing_output_directory(
    tmp_path: Path, input_kind: str, directory_name: str, output_format: str
) -> None:
    """An existing directory accepts every format regardless of its suffix."""
    inputs, sources = _csv_inputs(tmp_path, input_kind)
    original_inputs = {source: source.read_bytes() for source in sources}
    output = tmp_path / directory_name
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("unrelated content", encoding="utf-8")
    destinations = [output / f"{source.stem}{OUTPUT_FILE_SUFFIXES[output_format]}" for source in sources]
    for destination in destinations:
        destination.write_text("stale output", encoding="utf-8")

    result = CliRunner().invoke(app, ["parse", *inputs, "-o", str(output), "-f", output_format, "--tier", "flash"])

    assert result.exit_code == 0, result.output
    assert set(output.iterdir()) == {sentinel, *destinations}
    assert sentinel.read_text(encoding="utf-8") == "unrelated content"
    assert {source: source.read_bytes() for source in sources} == original_inputs
    for source, destination in zip(sources, destinations):
        if output_format == "zip":
            with zipfile.ZipFile(destination) as archive:
                assert source.stem in archive.read("markdown.md").decode("utf-8")
                assert json.loads(archive.read("middle_json.json"))["pages"]
        elif output_format == "middle_json":
            payload = json.loads(destination.read_text(encoding="utf-8"))
            assert source.stem in json.dumps(payload["pages"])
        else:
            assert source.stem in destination.read_text(encoding="utf-8")


@pytest.mark.parametrize("input_kind", ["multiple_files", "directory_one", "directory_many"])
@pytest.mark.parametrize("output_name", ["results.v1", "results.md"])
@pytest.mark.parametrize("existing_file", [False, True])
def test_parse_rejects_suffixed_non_directory_batch_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, input_kind: str, output_name: str, existing_file: bool
) -> None:
    """Batch validation must still reject suffixed paths that are not directories."""
    inputs, sources = _csv_inputs(tmp_path, input_kind)
    original_inputs = {source: source.read_bytes() for source in sources}
    output = tmp_path / output_name
    if existing_file:
        output.write_text("existing output file", encoding="utf-8")

    def unexpected_parse(*args: Any, **kwargs: Any) -> ParseResult:
        """Fail if invalid output paths reach the parser."""
        pytest.fail("invalid output paths must be rejected before parsing")

    monkeypatch.setattr(parse, "local_parse", unexpected_parse)
    result = CliRunner().invoke(app, ["parse", *inputs, "-o", str(output), "--tier", "flash"])

    assert result.exit_code == 1
    assert "directory path" in " ".join(result.output.split())
    assert {source: source.read_bytes() for source in sources} == original_inputs
    if existing_file:
        assert output.read_text(encoding="utf-8") == "existing output file"
    else:
        assert not output.exists()


@pytest.mark.parametrize("input_kind", ["multiple_files", "directory_one", "directory_many"])
@pytest.mark.parametrize("output_format", ["markdown", "middle_json", "zip"])
def test_parse_rejects_existing_suffixless_file_for_batch_input(tmp_path: Path, input_kind: str, output_format: str) -> None:
    """Even a directory containing one input must not overwrite a regular file."""
    inputs, sources = _csv_inputs(tmp_path, input_kind)
    original_inputs = {source: source.read_bytes() for source in sources}
    output = tmp_path / "results"
    output.write_bytes(b"existing output file")

    result = CliRunner().invoke(app, ["parse", *inputs, "-o", str(output), "-f", output_format, "--tier", "flash"])

    assert result.exit_code == 1
    assert "directory path" in " ".join(result.output.split())
    assert output.read_bytes() == b"existing output file"
    assert {source: source.read_bytes() for source in sources} == original_inputs


@pytest.mark.parametrize(
    ("input_kind", "existing_file"),
    [("multiple_files", False), ("directory_one", False), ("directory_many", False), ("single_file", True)],
)
@pytest.mark.parametrize("output_format", ["markdown", "middle_json", "zip"])
def test_parse_preserves_valid_suffixless_output_paths(
    tmp_path: Path, input_kind: str, existing_file: bool, output_format: str
) -> None:
    """New directories and explicit single-file overwrites keep their original behavior."""
    inputs, sources = _csv_inputs(tmp_path, input_kind)
    original_inputs = {source: source.read_bytes() for source in sources}
    output = tmp_path / "results"
    if existing_file:
        output.write_bytes(b"stale output")

    result = CliRunner().invoke(app, ["parse", *inputs, "-o", str(output), "-f", output_format, "--tier", "flash"])

    assert result.exit_code == 0, result.output
    assert {source: source.read_bytes() for source in sources} == original_inputs
    if existing_file:
        destinations = [output]
    else:
        assert output.is_dir()
        destinations = [output / f"{source.stem}{OUTPUT_FILE_SUFFIXES[output_format]}" for source in sources]
        assert set(output.iterdir()) == set(destinations)
    for source, destination in zip(sources, destinations):
        if output_format == "zip":
            with zipfile.ZipFile(destination) as archive:
                assert source.stem in archive.read("markdown.md").decode("utf-8")
        elif output_format == "middle_json":
            payload = json.loads(destination.read_text(encoding="utf-8"))
            assert source.stem in json.dumps(payload["pages"])
        else:
            assert source.stem in destination.read_text(encoding="utf-8")
