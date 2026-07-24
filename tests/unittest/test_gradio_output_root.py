import asyncio
from pathlib import Path

import pytest

from mineru.cli import gradio_app
from mineru.cli.gradio_app import (
    create_gradio_run_paths,
    normalize_gradio_output_root,
    register_gradio_allowed_path,
)


def test_normalize_gradio_output_root_creates_directory(tmp_path):
    output_root = tmp_path / "nested" / "results"

    normalized_root = normalize_gradio_output_root(output_root)

    assert normalized_root == str(output_root.resolve())
    assert output_root.is_dir()
    assert not list(output_root.glob(".mineru-write-test-*"))


@pytest.mark.parametrize("output_root", ["", "   ", None])
def test_normalize_gradio_output_root_rejects_empty_value(output_root):
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_gradio_output_root(output_root)


def test_normalize_gradio_output_root_rejects_file(tmp_path):
    output_file = tmp_path / "result.txt"
    output_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="not writable"):
        normalize_gradio_output_root(output_file)


def test_register_gradio_allowed_path_normalizes_and_deduplicates(tmp_path):
    allowed_paths = []

    normalized_root = register_gradio_allowed_path(allowed_paths, tmp_path)
    register_gradio_allowed_path(allowed_paths, tmp_path / ".")

    assert normalized_root == str(tmp_path.resolve())
    assert allowed_paths == [normalized_root]


def test_create_gradio_run_paths_uses_custom_output_root(tmp_path):
    run_root, extract_root, archive_zip_path = create_gradio_run_paths(
        "sample.pdf",
        output_root=tmp_path,
    )

    assert run_root.parent == tmp_path / "gradio"
    assert extract_root == run_root / "result"
    assert archive_zip_path.parent == run_root
    assert archive_zip_path.name == "sample.zip"


def test_create_gradio_run_paths_keeps_default_output_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    run_root, _, _ = create_gradio_run_paths("sample.pdf")

    assert run_root.parent == Path("./output/gradio")


def test_stream_to_markdown_forwards_output_root(monkeypatch, tmp_path):
    captured = {}

    async def fake_run_to_markdown_job(**kwargs):
        captured.update(kwargs)
        return "rendered", "source", "{}", "result.zip", "preview.pdf"

    monkeypatch.setattr(
        gradio_app,
        "_run_to_markdown_job",
        fake_run_to_markdown_job,
    )

    async def collect_updates():
        return [
            update
            async for update in gradio_app.stream_to_markdown(
                file_path="sample.pdf",
                output_root=str(tmp_path),
            )
        ]

    updates = asyncio.run(collect_updates())

    assert captured["output_root"] == str(tmp_path)
    assert updates[-1][1:] == (
        "result.zip",
        "rendered",
        "source",
        "{}",
        "preview.pdf",
    )
