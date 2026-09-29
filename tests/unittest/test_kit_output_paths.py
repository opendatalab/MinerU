# Copyright (c) Opendatalab. All rights reserved.
"""Regression tests for mineru-kit output path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from mineru.kit.common import resolve_batch_output_paths


def _make_pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\n")
    return path


def test_single_input_existing_output_file_is_accepted(tmp_path: Path) -> None:
    """`mineru-kit parse a.pdf -o a.md` must still work when a.md already exists."""
    source = _make_pdf(tmp_path / "sample.pdf")
    output = tmp_path / "document.md"
    output.write_text("stale")

    assert resolve_batch_output_paths([source], output, "markdown") == {source: output}


def test_single_input_file_like_output_is_accepted(tmp_path: Path) -> None:
    source = _make_pdf(tmp_path / "sample.pdf")
    output = tmp_path / "document.md"

    assert resolve_batch_output_paths([source], output, "markdown") == {source: output}


def test_multi_input_file_like_output_is_rejected(tmp_path: Path) -> None:
    first = _make_pdf(tmp_path / "a.pdf")
    second = _make_pdf(tmp_path / "b.pdf")

    with pytest.raises(ValueError):
        resolve_batch_output_paths([first, second], tmp_path / "out.md", "markdown")


def test_multi_input_existing_output_file_is_rejected(tmp_path: Path) -> None:
    first = _make_pdf(tmp_path / "a.pdf")
    second = _make_pdf(tmp_path / "b.pdf")
    output = tmp_path / "document.md"
    output.write_text("stale")

    with pytest.raises(ValueError):
        resolve_batch_output_paths([first, second], output, "markdown")


def test_single_input_directory_output_gets_stem_appended(tmp_path: Path) -> None:
    source = _make_pdf(tmp_path / "sample.pdf")
    outdir = tmp_path / "out"
    outdir.mkdir()

    assert resolve_batch_output_paths([source], outdir, "markdown") == {source: outdir / "sample.md"}
