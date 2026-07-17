# Copyright (c) Opendatalab. All rights reserved.
"""VLM后端两阶段解耦的自动化集成测试（Fake模型驱动，无需权重/GPU）。"""
import asyncio
import json
import os
from pathlib import Path

import pytest
from mineru_vl_utils.structs import ContentBlock

from mineru.backend.vlm import vlm_analyze
from mineru.backend.vlm.stages import (
    DEFAULT_LAYOUT_DOC_FILENAME,
    LAYOUT_DOC_VERSION,
    ContentRecognizer,
    LayoutDetector,
    PrecomputedLayoutDetector,
    block_from_jsonable,
    block_to_jsonable,
    blocks_from_jsonable,
    build_layout_doc,
    layout_doc_to_json,
    load_layout_doc,
)
from mineru.data.data_reader_writer import FileBasedDataWriter

SMALL_OCR_PDF = Path(__file__).parent.parent.parent / "demo" / "pdfs" / "small_ocr.pdf"
SMALL_OCR_PAGE_COUNT = 8


def _read_small_ocr_pdf() -> bytes:
    return SMALL_OCR_PDF.read_bytes()


def _make_blocks(page_idx: int) -> list[ContentBlock]:
    return [
        ContentBlock("title", [0.1, 0.05, 0.9, 0.1], angle=0),
        ContentBlock("text", [0.1, 0.15, 0.9, 0.4], angle=0),
    ]


class FakeLayoutDetector(LayoutDetector):
    name = "fake-layout"

    def __init__(self, blocks_factory=_make_blocks):
        self._blocks_factory = blocks_factory
        self.calls = []

    def batch_detect(self, images, start_page_idx: int = 0):
        self.calls.append((len(images), start_page_idx))
        return [
            self._blocks_factory(start_page_idx + offset)
            for offset in range(len(images))
        ]


class FakeContentRecognizer(ContentRecognizer):
    name = "fake-recognizer"

    def __init__(self):
        self.calls = []

    def batch_recognize(self, images, blocks_list, image_analysis: bool = True):
        self.calls.append(len(images))
        results = []
        for page_offset, page_blocks in enumerate(blocks_list):
            page_results = []
            for block_idx, block in enumerate(page_blocks):
                filled = ContentBlock(
                    block["type"],
                    list(block["bbox"]),
                    angle=block.get("angle"),
                    content=block.get("content") or f"fake {block['type']} {block_idx}",
                )
                for key, value in block.items():
                    if key not in ("type", "bbox", "angle", "content", "merge_prev"):
                        filled[key] = value
                page_results.append(filled)
            results.append(page_results)
        return results


class FakePredictor:
    """默认路径守卫：只实现batch_two_step_extract，证明无注入时仍走原调用。"""

    def __init__(self):
        self.two_step_calls = 0

    def batch_two_step_extract(self, images, image_analysis=True):
        self.two_step_calls += 1
        return [
            [ContentBlock("text", [0.1, 0.1, 0.9, 0.5], angle=0, content="two-step text")]
            for _ in images
        ]


# ---------------------------------------------------------------------------
# serde
# ---------------------------------------------------------------------------

def test_block_serde_roundtrip():
    block = ContentBlock("text", [0.1, 0.2, 0.5, 0.6], angle=90, content="hello", merge_prev=True)
    block["sub_type"] = "seal"
    data = block_to_jsonable(block)
    # 经过真实JSON编解码
    restored = block_from_jsonable(json.loads(json.dumps(data)))
    assert isinstance(restored, ContentBlock)
    assert dict(restored) == dict(block)


def test_block_from_jsonable_invalid_block_skipped():
    # bbox 超出0-1范围 → ContentBlock断言失败 → 返回None而不是抛异常
    assert block_from_jsonable({"type": "text", "bbox": [0, 0, 800, 600]}) is None
    assert block_from_jsonable({"type": "not_a_type", "bbox": [0.1, 0.1, 0.2, 0.2]}) is None
    blocks = blocks_from_jsonable([
        {"type": "text", "bbox": [0.1, 0.1, 0.9, 0.2]},
        {"type": "text", "bbox": [0, 0, 800, 600]},
    ])
    assert len(blocks) == 1


def test_layout_doc_build_and_load(tmp_path):
    blocks = _make_blocks(0)
    doc = build_layout_doc(
        [{"page_idx": 0, "page_size": [800, 1000], "blocks": blocks}],
        layout_backend="fake-layout",
        emits_formula_number=True,
    )
    assert doc["version"] == LAYOUT_DOC_VERSION
    assert doc["page_count"] == 1
    assert doc["emits_formula_number"] is True
    json_str = layout_doc_to_json(doc)
    # dict / JSON字符串 / 文件路径 三种来源
    assert load_layout_doc(doc) == doc
    assert load_layout_doc(json_str) == doc
    path = tmp_path / "layout.json"
    path.write_text(json_str, encoding="utf-8")
    assert load_layout_doc(str(path)) == doc


def test_load_layout_doc_rejects_bad_version():
    with pytest.raises(ValueError):
        load_layout_doc({"version": 999, "pages": []})


# ---------------------------------------------------------------------------
# PrecomputedLayoutDetector
# ---------------------------------------------------------------------------

def _build_doc_with_pages(page_indices):
    pages = [
        {
            "page_idx": idx,
            "page_size": [800, 1000],
            "blocks": [{"type": "text", "bbox": [0.1, 0.1, 0.9, 0.2], "content": f"p{idx}"}],
        }
        for idx in page_indices
    ]
    return build_layout_doc(pages, layout_backend="fake-layout")


def test_precomputed_detector_window_indexing():
    detector = PrecomputedLayoutDetector(_build_doc_with_pages(range(8)))
    # 模拟第二个窗口：页4-7
    blocks_list = detector.batch_detect([None] * 4, start_page_idx=4)
    assert len(blocks_list) == 4
    assert blocks_list[0][0]["content"] == "p4"
    assert blocks_list[3][0]["content"] == "p7"


def test_precomputed_detector_missing_page_raises():
    detector = PrecomputedLayoutDetector(_build_doc_with_pages([0, 1]))
    with pytest.raises(ValueError):
        detector.batch_detect([None] * 3, start_page_idx=0)


def test_precomputed_detector_returns_fresh_instances():
    detector = PrecomputedLayoutDetector(_build_doc_with_pages([0]))
    first = detector.batch_detect([None], start_page_idx=0)
    second = detector.batch_detect([None], start_page_idx=0)
    assert first[0][0] is not second[0][0]
    # 下游污染第一次结果不影响第二次
    first[0][0]["content"] = "polluted"
    assert second[0][0]["content"] == "p0"
    assert detector.batch_detect([None], start_page_idx=0)[0][0]["content"] == "p0"


# ---------------------------------------------------------------------------
# doc_analyze 解耦路径全链路（Fake模型，无需权重）
# ---------------------------------------------------------------------------

def test_doc_analyze_decoupled_full_chain(tmp_path):
    pdf_bytes = _read_small_ocr_pdf()
    detector = FakeLayoutDetector()
    recognizer = FakeContentRecognizer()
    image_writer = FileBasedDataWriter(str(tmp_path / "images"))
    layout_writer = FileBasedDataWriter(str(tmp_path))

    middle_json, results = vlm_analyze.doc_analyze(
        pdf_bytes,
        image_writer,
        layout_detector=detector,
        content_recognizer=recognizer,
        layout_writer=layout_writer,
    )

    assert detector.calls and recognizer.calls
    assert len(middle_json["pdf_info"]) == SMALL_OCR_PAGE_COUNT
    assert len(results) == SMALL_OCR_PAGE_COUNT
    # 识别内容进入了每页结果
    for page_results in results:
        contents = [block.get("content") for block in page_results]
        assert any(c and c.startswith("fake") for c in contents)
    # middle_json 每页有块产出
    for page_info in middle_json["pdf_info"]:
        assert page_info["page_idx"] >= 0
        assert page_info["para_blocks"] or page_info["preproc_blocks"]

    # layout.json 落盘且符合schema
    layout_path = tmp_path / DEFAULT_LAYOUT_DOC_FILENAME
    assert layout_path.exists()
    layout_doc = load_layout_doc(str(layout_path))
    assert layout_doc["page_count"] == SMALL_OCR_PAGE_COUNT
    assert layout_doc["layout_backend"] == "fake-layout"
    for page in layout_doc["pages"]:
        assert len(page["page_size"]) == 2
        for block in page["blocks"]:
            assert block["type"]
            assert all(0.0 <= v <= 1.0 for v in block["bbox"])
    # layout快照必须是识别前的（未带Fake识别内容）
    for page in layout_doc["pages"]:
        for block in page["blocks"]:
            assert not (block.get("content") or "").startswith("fake")


def test_doc_analyze_default_path_uses_two_step():
    pdf_bytes = _read_small_ocr_pdf()
    predictor = FakePredictor()
    middle_json, results = vlm_analyze.doc_analyze(pdf_bytes, None, predictor=predictor)
    assert predictor.two_step_calls > 0
    assert len(middle_json["pdf_info"]) == SMALL_OCR_PAGE_COUNT


def test_doc_analyze_injected_layout_skips_two_step():
    pdf_bytes = _read_small_ocr_pdf()

    class NoTwoStepPredictor(FakePredictor):
        def batch_extract_with_layout(self, images, blocks_list, image_analysis=True):
            return FakeContentRecognizer().batch_recognize(images, blocks_list, image_analysis)

    predictor = NoTwoStepPredictor()
    middle_json, results = vlm_analyze.doc_analyze(
        pdf_bytes,
        None,
        predictor=predictor,
        layout_detector=FakeLayoutDetector(),
    )
    # 注入layout后：两步合并调用不应被触发，识别走extract_with_layout
    assert predictor.two_step_calls == 0
    assert len(results) == SMALL_OCR_PAGE_COUNT


def test_doc_analyze_multi_window(monkeypatch, tmp_path):
    monkeypatch.setenv("MINERU_PROCESSING_WINDOW_SIZE", "3")  # 8页 → 3窗口
    pdf_bytes = _read_small_ocr_pdf()
    detector = FakeLayoutDetector()
    layout_writer = FileBasedDataWriter(str(tmp_path))
    middle_json, results = vlm_analyze.doc_analyze(
        pdf_bytes,
        None,
        layout_detector=detector,
        content_recognizer=FakeContentRecognizer(),
        layout_writer=layout_writer,
    )
    assert detector.calls == [(3, 0), (3, 3), (2, 6)]
    layout_doc = load_layout_doc(str(tmp_path / DEFAULT_LAYOUT_DOC_FILENAME))
    assert layout_doc["page_count"] == SMALL_OCR_PAGE_COUNT
    assert [page["page_idx"] for page in layout_doc["pages"]] == list(range(SMALL_OCR_PAGE_COUNT))


def test_aio_doc_analyze_decoupled():
    pdf_bytes = _read_small_ocr_pdf()
    middle_json, results = asyncio.run(
        vlm_analyze.aio_doc_analyze(
            pdf_bytes,
            None,
            layout_detector=FakeLayoutDetector(),
            content_recognizer=FakeContentRecognizer(),
        )
    )
    assert len(middle_json["pdf_info"]) == SMALL_OCR_PAGE_COUNT
    assert len(results) == SMALL_OCR_PAGE_COUNT


# ---------------------------------------------------------------------------
# formula_number 门控
# ---------------------------------------------------------------------------

def test_formula_number_gating():
    pdf_bytes = _read_small_ocr_pdf()

    def blocks_with_formula_number(page_idx):
        return [
            ContentBlock("equation", [0.1, 0.3, 0.7, 0.4], angle=0, content="E=mc^2"),
            ContentBlock("formula_number", [0.75, 0.3, 0.9, 0.4], angle=0, content="(1)"),
        ]

    class PassthroughRecognizer(ContentRecognizer):
        def batch_recognize(self, images, blocks_list, image_analysis=True):
            return [list(page_blocks) for page_blocks in blocks_list]

    detector = FakeLayoutDetector(blocks_factory=blocks_with_formula_number)
    detector.emits_formula_number = True
    middle_json, results = vlm_analyze.doc_analyze(
        pdf_bytes,
        None,
        layout_detector=detector,
        content_recognizer=PassthroughRecognizer(),
    )
    for page_results in results:
        types = [block.get("type") for block in page_results]
        # 编号块被合并进公式（或降级），不再以formula_number形态存在
        assert "formula_number" not in types
        equation_contents = [
            block.get("content") for block in page_results if block.get("type") == "equation"
        ]
        assert any(r"\tag{1}" in (c or "") for c in equation_contents)


# ---------------------------------------------------------------------------
# doc_layout_analyze（仅第一阶段）→ PrecomputedLayoutDetector（仅第二阶段）
# ---------------------------------------------------------------------------

def test_two_stage_split_run(tmp_path):
    pdf_bytes = _read_small_ocr_pdf()
    layout_writer = FileBasedDataWriter(str(tmp_path))

    # 阶段一：只跑layout，落盘layout.json
    layout_doc = vlm_analyze.doc_layout_analyze(
        pdf_bytes,
        FakeLayoutDetector(),
        layout_writer=layout_writer,
    )
    assert layout_doc["page_count"] == SMALL_OCR_PAGE_COUNT
    layout_path = tmp_path / DEFAULT_LAYOUT_DOC_FILENAME
    assert layout_path.exists()

    # 阶段二：从layout.json恢复layout，只跑识别
    middle_json, results = vlm_analyze.doc_analyze(
        pdf_bytes,
        None,
        layout_detector=PrecomputedLayoutDetector(str(layout_path)),
        content_recognizer=FakeContentRecognizer(),
    )
    assert len(middle_json["pdf_info"]) == SMALL_OCR_PAGE_COUNT
    for page_results in results:
        assert any((block.get("content") or "").startswith("fake") for block in page_results)
