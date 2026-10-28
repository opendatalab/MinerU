"""复放真实 layout/OCR 检测结果，隔离推理随机性验证原生文本修复。"""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from docvortex.document.pdf import PDFDocument
from docvortex.postprocess.document import model_json_to_middle_json
from docvortex.schema import ModelJson

from mineru.backend.analysis.pdf.normalization import _normalize_pdf_model_list
from mineru.backend.analysis.pdf.text.content import _fill_window_block_content_and_lines
from mineru.backend.analysis.pdf.constants import VLM_TXT_DET_TYPE
from mineru.backend.analysis.pdf.snapshots import create_page_snapshot_cache
from mineru.parser.base import ParseResult

FIXTURE = Path(__file__).parents[1] / "fixtures/pdf_text_5635_5636"


@pytest.mark.parametrize("effort", ["medium", "high", "xhigh"])
@pytest.mark.parametrize("owned", [False, True])
def test_frozen_native_text_pipeline_restores_source_words(effort: str, owned: bool) -> None:
    """同一三页输入在回填、模型协议、中间协议和 Markdown 均保留完整词语。"""
    layout = json.loads((FIXTURE / "layout.json").read_text())
    detection = json.loads((FIXTURE / "ocr_detection.json").read_text())
    images = []
    context = MagicMock()
    with PDFDocument((FIXTURE / "source.pdf").read_bytes()) as document:
        try:
            for index in range(document.page_count):
                rendered = document.render_page(index, scale=200 / 72)
                images.append({"img_pil": rendered.pil_image, "scale": 200 / 72})
            pages = _fill_window_block_content_and_lines(
                images,
                [document[index] for index in range(document.page_count)],
                layout,
                [[], [], []],
                detection,
                "txt",
                effort,
                VLM_TXT_DET_TYPE,
                context,
                page_snapshots=create_page_snapshot_cache(
                    [document[index] for index in range(document.page_count)], enabled=owned
                ),
            )
            _normalize_pdf_model_list(pages)
        finally:
            for image in images:
                image["img_pil"].close()
    context.ocr_model.ocr.assert_not_called()
    model_dict = json.loads((FIXTURE / "before_model.json").read_text())
    model_dict["pages"] = pages
    model = ModelJson.from_dict(model_dict)
    middle = model_json_to_middle_json(model)
    markdown = ParseResult(middle_json=middle).markdown()
    for value in [json.dumps(model.to_dict(), ensure_ascii=False), json.dumps(middle.to_dict(), ensure_ascii=False), markdown]:
        assert "This book is a practical" in value
        assert value.count("financial") == 5
        assert "fi nancial" not in value
        assert "mo mentum" not in value
        assert "tech niques" not in value
    first = next(block for block in pages[0] if block["type"] == "text")
    assert first["lines"][0]["bbox"][0] <= 54 / 432
    assert first["lines"][0]["bbox"][3] >= 262.36 / 648


def test_drop_cap_recovery_guards() -> None:
    """不把正常字号、标题、别栏字符或已认领首字重复补入段落。"""
    from mineru.backend.analysis.pdf.text.content import _recover_native_drop_caps
    from mineru.backend.analysis.pdf.text.models import _AnalyzeLine, _AnalyzeSpan
    from docvortex.document.pdf import PDFPageTextGeometry

    chars: list[dict[str, Any]] = [
        {"char": "T", "char_idx": 0, "bbox": (10.0, 10.0, 20.0, 32.0), "writing_angle": 0.0},
        {"char": "h", "char_idx": 1, "bbox": (20.5, 10.0, 25.0, 18.0), "writing_angle": 0.0},
        {"char": "i", "char_idx": 2, "bbox": (25.0, 10.0, 28.0, 18.0), "writing_angle": 0.0},
    ]
    for case in ["drop_cap", "ordinary", "title", "other_column", "already_present", "formula", "code", "owned_elsewhere"]:
        current = deepcopy(chars)
        content = "This" if case == "already_present" else "his"
        bbox = (10.0, 10.0, 80.0, 60.0)
        if case == "ordinary":
            current[0]["bbox"] = (10.0, 10.0, 20.0, 18.0)
        if case == "other_column":
            current[0]["bbox"] = (0.0, 10.0, 5.0, 32.0)
        geometry = PDFPageTextGeometry(chars=current, tight_bboxes={i: c["bbox"] for i, c in enumerate(current)}, origins={})
        span = _AnalyzeSpan("text", (20.5, 10.0, 75.0, 18.0), content=content)
        lines = {0: [_AnalyzeLine(span.bbox, [span])]}
        kind = "paragraph_title" if case == "title" else "code" if case == "code" else "text"
        blocks = [{"type": kind, "bbox": [v / 100 for v in bbox]}]
        if case == "owned_elsewhere":
            owned = _AnalyzeSpan("text", current[0]["bbox"], content="T")
            lines[1] = [_AnalyzeLine(owned.bbox, [owned])]
            blocks.append({"type": "paragraph_title", "bbox": [v / 100 for v in bbox]})
        _recover_native_drop_caps(blocks, lines, (100.0, 100.0), geometry, [current[0]["bbox"]] if case == "formula" else [])
        assert span.content == ("This" if case in {"drop_cap", "already_present"} else "his")
        if case == "drop_cap":
            _recover_native_drop_caps(blocks, lines, (100.0, 100.0), geometry, [])
            assert span.content == "This"


@pytest.mark.parametrize("tight_spacing", [False, True])
def test_owned_ligature_text_matches_reference_and_preserves_code(tight_spacing: bool) -> None:
    """真实快照的 Rust/Python 组装结果一致，代码区域保留源空格。"""
    from dataclasses import asdict
    from docvortex import _native
    from mineru.backend.analysis.pdf.text.native import fill_char_in_spans
    from mineru.backend.analysis.pdf.text.models import _AnalyzeSpan

    with PDFDocument((FIXTURE / "source.pdf").read_bytes()) as document:
        owner = document[0].get_text_snapshot()
    if owner is None:
        pytest.skip("Rust snapshot unavailable")
    results = []
    before = _native.text_snapshot_stats()[2]
    for current_owner in [None, owner]:
        geometry = owner.materialize_geometry()
        span = _AnalyzeSpan(
            "text",
            (0.0, 0.0, 432.0, 648.0),
            metadata={
                "chars": [],
                "height": 648.0,
                "width": 432.0,
                "_native_tight_spacing": tight_spacing,
            },
        )
        pending = fill_char_in_spans(
            [span],
            geometry.chars,
            12.0,
            tight_bboxes=geometry.tight_bboxes,
            origins=geometry.origins,
            detect_scripts=False,
            _native_text=current_owner,
        )
        assert not pending
        assert span.content.count("financial" if tight_spacing else "fi nancial") == 2
        results.append(asdict(span))
    assert results[0] == results[1]
    assert _native.text_snapshot_stats()[2] == before + 1


@pytest.mark.parametrize("text", ["a\ufffeb", "a\uffffb"])
def test_noncharacters_are_not_unconditionally_rewritten(text: str) -> None:
    """有原生字符归属的非字符保持原值，不把任意 U+FFFE/U+FFFF 当作断词符。"""
    from mineru.backend.analysis.pdf.text.native import chars_to_content
    from mineru.backend.analysis.pdf.text.models import _AnalyzeSpan

    chars = [
        {"char": value, "char_idx": index, "bbox": (index * 5.0, 0.0, index * 5.0 + 5.0, 10.0)}
        for index, value in enumerate(text)
    ]
    span = _AnalyzeSpan("text", (0.0, 0.0, 20.0, 10.0), metadata={"chars": chars})
    chars_to_content(span, detect_scripts=False)
    assert span.content == text
