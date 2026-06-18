import json
from pathlib import Path

from PIL import Image, JpegImagePlugin

from benchmarks.mineru_diffusion.compare_results import parse_layout_blocks
from benchmarks.mineru_diffusion.end2end_suite import (
    PageCase,
    PdfSource,
    blocks_to_layout_text,
    blocks_to_markdown,
    compare_suite_rows,
    parse_page_spec,
    render_pdf_sources,
    run_dllm_throughput_suite,
    summarize_suite_rows,
)


def test_parse_page_spec_supports_ranges_and_shortcuts():
    assert parse_page_spec("0,2-4,last", page_count=6) == [0, 2, 3, 4, 5]
    assert parse_page_spec("first:3,last:2", page_count=6) == [0, 1, 2, 4, 5]
    assert parse_page_spec("all", page_count=3) == [0, 1, 2]


def test_render_pdf_sources_writes_images_and_manifest(tmp_path: Path):
    assert JpegImagePlugin is not None
    pdf_path = tmp_path / "sample.pdf"
    first = Image.new("RGB", (80, 100), "white")
    second = Image.new("RGB", (90, 110), "white")
    first.save(pdf_path, save_all=True, append_images=[second])

    cases = render_pdf_sources(
        [PdfSource(doc_id="sample", pdf_path=pdf_path, pages="0,last")],
        output_dir=tmp_path / "rendered",
        scale=1.0,
    )

    assert [case.case_id for case in cases] == ["sample_p0001", "sample_p0002"]
    assert all(case.image_path.exists() for case in cases)
    manifest = json.loads((tmp_path / "rendered" / "manifest.json").read_text())
    assert [row["page_index"] for row in manifest["cases"]] == [0, 1]


def test_blocks_to_markdown_and_layout_text_normalize_suite_payload():
    blocks = [
        {
            "type": "title",
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "angle": 0,
            "content": "Title",
        },
        {
            "type": "table",
            "bbox": [0.4, 0.5, 0.8, 0.9],
            "angle": 90,
            "content": "<fcel>A<nl>",
        },
    ]

    markdown = blocks_to_markdown(blocks)
    layout_text = blocks_to_layout_text(blocks)

    assert "Title" in markdown
    assert "<table>" in markdown
    parsed = parse_layout_blocks(layout_text)
    assert [(block.label, block.bbox) for block in parsed] == [
        ("title", (100.0, 200.0, 300.0, 400.0)),
        ("table", (400.0, 500.0, 800.0, 900.0)),
    ]
    assert "<|rotate_right|>" in layout_text


def test_compare_suite_rows_aggregates_latency_and_quality():
    baseline_rows = [
        {
            "case_id": "doc_p0001",
            "ok": True,
            "metrics": {"total_elapsed": 2.0},
            "markdown": "same text",
            "layout_output": (
                "<|box_start|>000 000 500 500<|box_end|>"
                "<|ref_start|>text<|ref_end|><|rotate_up|>"
            ),
            "block_type_counts": {"text": 1},
        }
    ]
    candidate_rows = [
        {
            "case_id": "doc_p0001",
            "ok": True,
            "metrics": {"total_elapsed": 1.0},
            "markdown": "same text",
            "layout_output": (
                "<|box_start|>010 010 490 490<|box_end|>"
                "<|ref_start|>text<|ref_end|><|rotate_up|>"
            ),
            "block_type_counts": {"text": 1},
        }
    ]

    payload = compare_suite_rows(baseline_rows, candidate_rows)

    assert payload["summary"]["num_cases"] == 1
    assert payload["summary"]["total_speedup"] == 2.0
    assert payload["summary"]["mean_markdown_similarity"] == 1.0
    assert payload["summary"]["mean_layout_f1"] == 1.0
    assert payload["cases"][0]["layout_f1"] == 1.0


def test_summarize_suite_rows_carries_common_content_concurrency():
    rows = [
        {
            "ok": True,
            "metrics": {
                "total_elapsed": 2.0,
                "content_concurrency": 4,
            },
            "markdown": "abcd",
            "block_type_counts": {"text": 1},
        },
        {
            "ok": True,
            "metrics": {
                "total_elapsed": 3.0,
                "content_concurrency": 4,
            },
            "markdown": "ef",
            "block_type_counts": {"table": 1},
        },
    ]

    summary = summarize_suite_rows(rows)

    assert summary["content_concurrency"] == 4


def test_summarize_suite_rows_reports_throughput_wall_clock():
    rows = [
        {
            "ok": True,
            "metrics": {
                "throughput_wall_elapsed_s": 5.0,
                "throughput_layout_wall_elapsed_s": 2.0,
                "throughput_extract_wall_elapsed_s": 3.0,
                "layout_concurrency": 2,
                "content_concurrency": 4,
            },
            "markdown": "abcd",
            "block_type_counts": {"text": 1},
        },
        {
            "ok": True,
            "metrics": {
                "throughput_wall_elapsed_s": 5.0,
                "throughput_layout_wall_elapsed_s": 2.0,
                "throughput_extract_wall_elapsed_s": 3.0,
                "layout_concurrency": 2,
                "content_concurrency": 4,
            },
            "markdown": "ef",
            "block_type_counts": {"table": 1},
        },
    ]

    summary = summarize_suite_rows(rows)

    assert summary["throughput_wall_elapsed_s"] == 5.0
    assert summary["throughput_layout_wall_elapsed_s"] == 2.0
    assert summary["throughput_extract_wall_elapsed_s"] == 3.0
    assert summary["throughput_pages_per_s"] == 0.4
    assert summary["throughput_markdown_chars_per_s"] == 1.2
    assert summary["layout_concurrency"] == 2
    assert summary["content_concurrency"] == 4


def test_run_dllm_throughput_suite_returns_compare_compatible_rows(tmp_path: Path):
    image_paths = []
    cases = []
    for index in range(2):
        image_path = tmp_path / f"page_{index}.png"
        Image.new("RGB", (100, 100), "white").save(image_path)
        image_paths.append(image_path)
        cases.append(
            PageCase(
                case_id=f"doc_p{index + 1:04d}",
                doc_id="doc",
                pdf_path=tmp_path / "doc.pdf",
                page_index=index,
                image_path=image_path,
                width=100,
                height=100,
            )
        )

    class FakeClient:
        def infer(self, image_path, prompt, max_tokens):
            if prompt == "\nLayout Detection:":
                return (
                    "<|box_start|>000 000 500 500<|box_end|>"
                    "<|ref_start|>text<|ref_end|><|rotate_up|>"
                )
            return "content"

    rows = run_dllm_throughput_suite(
        cases,
        output_dir=tmp_path / "out",
        endpoint="http://unused/v1/chat/completions",
        model="mineru-diffusion",
        timeout=10,
        layout_max_tokens=64,
        content_max_tokens=64,
        table_max_tokens=64,
        formula_max_tokens=64,
        block_size=32,
        dynamic_threshold=0.9,
        layout_concurrency=2,
        content_concurrency=2,
        client=FakeClient(),
    )

    summary = summarize_suite_rows(rows)

    assert [row["case_id"] for row in rows] == ["doc_p0001", "doc_p0002"]
    assert all(row["ok"] for row in rows)
    assert all(row["markdown"] == "content" for row in rows)
    assert summary["throughput_wall_elapsed_s"] > 0
    assert summary["layout_concurrency"] == 2
    assert summary["content_concurrency"] == 2
