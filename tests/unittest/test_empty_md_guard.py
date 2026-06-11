# Copyright (c) Opendatalab. All rights reserved.
"""Regression tests for the silent 0-byte .md write bug.

Repro for opendatalab/MinerU#5113-style failure mode: when the upstream
model output cannot be assembled into markdown (e.g. format markers absent,
parser bug, edge case), `_process_output` previously wrote a 0-byte .md file
silently. /file_parse then returned HTTP 200 + md_content="" — indistinguishable
from success. These tests pin the new "fail loudly" contract.
"""
import os
import tempfile
from unittest import mock

import pytest

from mineru.cli.common import _process_output
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.data.utils.exceptions import EmptyData


def _fake_pdf_info_with_pages():
    """Minimal pdf_info shape: list of page dicts. Non-empty so the guard
    triggers — represents a PDF that DID get pages, but extraction produced
    nothing useful from them."""
    return [
        {
            "para_blocks": [],
            "discarded_blocks": [],
            "page_idx": 0,
            "page_size": [612, 792],
        }
    ]


def test_process_output_raises_on_empty_md_with_pages(tmp_path):
    """When pdf_info has pages but the extractor produced no markdown, refuse
    to write a 0-byte .md file. Raise EmptyData instead."""
    image_dir = tmp_path / "images"
    md_dir = tmp_path / "md"
    image_dir.mkdir()
    md_dir.mkdir()
    md_writer = FileBasedDataWriter(str(md_dir))

    # Patch vlm_union_make to simulate "model output didn't match MinerU
    # format markers" — returns empty string. This is the exact failure
    # observed in #5113 with vlm-http-client + a non-MinerU upstream model.
    with mock.patch(
        "mineru.cli.common.vlm_union_make", return_value=""
    ):
        # Make sure the opt-out env var is NOT set
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MINERU_ALLOW_EMPTY_MD", None)
            with pytest.raises(EmptyData) as excinfo:
                _process_output(
                    pdf_info=_fake_pdf_info_with_pages(),
                    pdf_bytes=b"",
                    pdf_file_name="dummy",
                    local_md_dir=str(md_dir),
                    local_image_dir=str(image_dir),
                    md_writer=md_writer,
                    f_draw_layout_bbox=False,
                    f_draw_span_bbox=False,
                    f_dump_orig_pdf=False,
                    f_dump_md=True,
                    f_dump_content_list=False,
                    f_dump_middle_json=False,
                    f_dump_model_output=False,
                    f_make_md_mode="mm_md",
                    middle_json={"pdf_info": _fake_pdf_info_with_pages()},
                    model_output=None,
                    process_mode="vlm",
                )
    assert "0-byte" in str(excinfo.value) or "No markdown" in str(excinfo.value)
    # And critically: NO .md file should have been written.
    assert not (md_dir / "dummy.md").exists(), (
        "guard must refuse to write — caller should not see a 0-byte .md"
    )


def test_process_output_writes_when_md_nonempty(tmp_path):
    """Sanity: real extraction output still writes normally."""
    image_dir = tmp_path / "images"
    md_dir = tmp_path / "md"
    image_dir.mkdir()
    md_dir.mkdir()
    md_writer = FileBasedDataWriter(str(md_dir))

    with mock.patch(
        "mineru.cli.common.vlm_union_make",
        return_value="# Real heading\n\nSome real content.",
    ):
        _process_output(
            pdf_info=_fake_pdf_info_with_pages(),
            pdf_bytes=b"",
            pdf_file_name="dummy",
            local_md_dir=str(md_dir),
            local_image_dir=str(image_dir),
            md_writer=md_writer,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_orig_pdf=False,
            f_dump_md=True,
            f_dump_content_list=False,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_make_md_mode="mm_md",
            middle_json={"pdf_info": _fake_pdf_info_with_pages()},
            model_output=None,
            process_mode="vlm",
        )
    md_path = md_dir / "dummy.md"
    assert md_path.exists()
    assert md_path.read_text(encoding="utf-8").strip() != ""


def test_process_output_opt_out_via_env(tmp_path):
    """MINERU_ALLOW_EMPTY_MD=true preserves legacy behavior for callers who
    legitimately expect empty markdown (e.g. zero-content / image-only docs
    in pipelines that handle this downstream)."""
    image_dir = tmp_path / "images"
    md_dir = tmp_path / "md"
    image_dir.mkdir()
    md_dir.mkdir()
    md_writer = FileBasedDataWriter(str(md_dir))

    with mock.patch(
        "mineru.cli.common.vlm_union_make", return_value=""
    ):
        with mock.patch.dict(os.environ, {"MINERU_ALLOW_EMPTY_MD": "true"}):
            # Should NOT raise — env opt-out honored.
            _process_output(
                pdf_info=_fake_pdf_info_with_pages(),
                pdf_bytes=b"",
                pdf_file_name="dummy",
                local_md_dir=str(md_dir),
                local_image_dir=str(image_dir),
                md_writer=md_writer,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_orig_pdf=False,
                f_dump_md=True,
                f_dump_content_list=False,
                f_dump_middle_json=False,
                f_dump_model_output=False,
                f_make_md_mode="mm_md",
                middle_json={"pdf_info": _fake_pdf_info_with_pages()},
                model_output=None,
                process_mode="vlm",
            )
    md_path = md_dir / "dummy.md"
    assert md_path.exists()
    # Legacy behavior preserved: 0-byte file IS allowed when opted in.
    assert md_path.read_text(encoding="utf-8") == ""


def test_process_output_allows_empty_pdf_info(tmp_path):
    """Edge case: pdf_info itself is empty (zero pages). Guard should NOT
    fire — there's nothing to extract from, so empty markdown is correct."""
    image_dir = tmp_path / "images"
    md_dir = tmp_path / "md"
    image_dir.mkdir()
    md_dir.mkdir()
    md_writer = FileBasedDataWriter(str(md_dir))

    with mock.patch(
        "mineru.cli.common.vlm_union_make", return_value=""
    ):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MINERU_ALLOW_EMPTY_MD", None)
            # Empty pdf_info — empty md is a legitimate outcome, no raise.
            _process_output(
                pdf_info=[],
                pdf_bytes=b"",
                pdf_file_name="dummy",
                local_md_dir=str(md_dir),
                local_image_dir=str(image_dir),
                md_writer=md_writer,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_orig_pdf=False,
                f_dump_md=True,
                f_dump_content_list=False,
                f_dump_middle_json=False,
                f_dump_model_output=False,
                f_make_md_mode="mm_md",
                middle_json={"pdf_info": []},
                model_output=None,
                process_mode="vlm",
            )
    assert (md_dir / "dummy.md").exists()
