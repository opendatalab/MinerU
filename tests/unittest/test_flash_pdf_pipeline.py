from __future__ import annotations

import ast
import inspect

import pytest

from mineru.backend.flash import pdf_extractor
from mineru.backend.flash.native_pdf import (
    auxiliary_text,
    formulas,
    geometry,
    graphics,
    index_blocks,
    line_layout,
    line_merging,
    models,
    native_text,
    pipeline,
    tables,
    text_blocks,
    titles,
    visual_annotations,
)
from mineru.utils.pdf_document import PDFImageInfo

from _flash_pdf_test_utils import (
    _prepared_text_page,
    _text_line,
)


def _image_info(
    fingerprint: str | None,
    bbox: tuple[float, float, float, float],
) -> PDFImageInfo:
    """构造跨页图片水印规则使用的轻量图片信息。"""

    return PDFImageInfo(bbox=bbox, fingerprint=fingerprint)


def test_prepare_page_materializes_table_against_original_semantic_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证表格检测避开预分类行，但物化阶段可看到 core 内误标页脚。"""

    body = _text_line("body", (10.0, 10.0, 90.0, 20.0), 0)
    footer = _text_line(
        "table tail",
        (10.0, 80.0, 40.0, 90.0),
        1,
        semantic_type="footer",
    )
    source = models._PageSource(
        page_size=(100.0, 100.0),
        lines=[body, footer],
        chars=[],
        drawing_lines=[],
    )
    observed: dict[str, list[int]] = {}

    def fake_detect(
        analysis_source: models._PageSource,
        *,
        excluded_bboxes: list[tuple[float, float, float, float]],
    ) -> list[models._TableCandidate]:
        """记录候选检测阶段可见的来源行。"""

        assert excluded_bboxes == []
        observed["detect"] = [line.source_index for line in analysis_source.lines]
        return []

    def fake_materialize(
        materialization_source: models._PageSource,
        candidates: list[models._TableCandidate],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], set[int]]:
        """记录表格物化阶段可见的来源行。"""

        assert candidates == []
        observed["materialize"] = [
            line.source_index for line in materialization_source.lines
        ]
        return [], [], set()

    monkeypatch.setattr(pipeline, "_detect_table_candidates", fake_detect)
    monkeypatch.setattr(pipeline, "_materialize_table_blocks", fake_materialize)

    pipeline._prepare_page_source(source)

    assert observed == {"detect": [0], "materialize": [0, 1]}


def test_prepare_page_uses_only_table_body_bbox_and_keeps_annotations_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Pipeline 仅用表体框作容器屏障，同时把预分类注释送入固定块。"""

    source = models._PageSource(
        page_size=(100.0, 100.0),
        lines=[],
        chars=[],
        drawing_lines=[],
    )
    table_block = {
        "type": "table",
        "bbox": (10.0, 30.0, 90.0, 70.0),
        "angle": 0,
        "content": "body",
    }
    caption_block = {
        "type": "caption",
        "bbox": (10.0, 10.0, 50.0, 20.0),
        "angle": 0,
        "content": "Table 1",
    }

    monkeypatch.setattr(
        pipeline,
        "_detect_table_candidates",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        pipeline,
        "_materialize_table_blocks",
        lambda *_args, **_kwargs: ([table_block], [caption_block], set()),
    )

    prepared = pipeline._prepare_page_source(source)

    assert prepared.table_bboxes == [table_block["bbox"]]
    assert [block["type"] for block in prepared.fixed_blocks] == [
        "caption",
        "table",
    ]


def test_repeated_large_image_requires_three_distinct_pages() -> None:
    """验证同页重复只计一次，面积恰好 8% 且跨三页时才命中水印。"""

    page_sizes = [(100.0, 100.0)] * 4
    page_image_infos = [
        [
            _image_info("watermark", (0.0, 0.0, 40.0, 20.0)),
            _image_info("watermark", (0.0, 0.0, 40.0, 20.0)),
            _image_info("two-pages", (0.0, 0.0, 50.0, 20.0)),
        ],
        [_image_info("watermark", (10.0, 10.0, 50.0, 30.0))],
        [_image_info("watermark", (20.0, 20.0, 60.0, 40.0))],
        [_image_info("two-pages", (0.0, 0.0, 50.0, 20.0))],
    ]

    fingerprints = pipeline._detect_repeated_raster_watermark_fingerprints(
        page_image_infos,
        page_sizes,
    )

    assert fingerprints == {"watermark"}


def test_repeated_watermark_filter_keeps_small_or_unfingerprinted_images() -> None:
    """验证已命中指纹也只删除大图，小图和指纹读取失败的图片继续进入现有流程。"""

    image_infos = [
        _image_info("watermark", (0.0, 0.0, 40.0, 20.0)),
        _image_info("watermark", (0.0, 0.0, 39.9, 20.0)),
        _image_info(None, (0.0, 0.0, 80.0, 80.0)),
        _image_info("ordinary", (0.0, 0.0, 80.0, 80.0)),
    ]

    filtered = pipeline._filter_repeated_raster_watermark_bboxes(
        image_infos,
        (100.0, 100.0),
        {"watermark"},
    )

    assert filtered == [
        (0.0, 0.0, 39.9, 20.0),
        (0.0, 0.0, 80.0, 80.0),
        (0.0, 0.0, 80.0, 80.0),
    ]


def test_flash_extractor_has_no_local_ocr_runtime_logic() -> None:
    """守卫 Flash extractor 不再引入或实现本地 OCR 运行时逻辑。"""

    source = "\n".join(
        inspect.getsource(module)
        for module in (
            pdf_extractor,
            auxiliary_text,
            formulas,
            geometry,
            graphics,
            index_blocks,
            line_layout,
            line_merging,
            models,
            native_text,
            pipeline,
            tables,
            text_blocks,
            titles,
            visual_annotations,
        )
    )
    forbidden_tokens = (
        "import numpy",
        "import cv2",
        "project_ocr_table_text",
        "_load_ocr_runtime",
        "_run_full_page_ocr",
        "_project_ocr_candidate",
        "_recognize_rotated_ocr_table",
        "AtomModelSingleton",
        "run_ocr_inference",
        "get_processing_window_size",
        "bgr_image",
        "pixel_quad",
        "ocr_model",
    )

    assert not [token for token in forbidden_tokens if token in source]


def test_flash_extractor_only_exports_supported_entrypoints() -> None:
    """验证旧模块只声明两个稳定公共入口，不再承载内部实现。"""

    assert pdf_extractor.__all__ == ["doc_analyze", "extract_pages_text"]
    assert not hasattr(pdf_extractor, "_analyze_native_document")


def test_native_pdf_domain_modules_do_not_import_each_other() -> None:
    """守卫领域处理器只依赖共享层，跨领域组合统一留在 pipeline。"""

    domain_modules = (
        auxiliary_text,
        formulas,
        graphics,
        index_blocks,
        tables,
        text_blocks,
        titles,
        visual_annotations,
    )
    domain_names = {module.__name__.rsplit(".", 1)[-1] for module in domain_modules}
    for module in domain_modules:
        tree = ast.parse(inspect.getsource(module))
        relative_imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is not None
        }
        assert not relative_imports & domain_names, module.__name__


@pytest.mark.parametrize(
    "block_type",
    [
        "doc_title",
        "paragraph_title",
        "header",
        "footer",
        "page_number",
        "caption",
        "footnote",
        "page_footnote",
        "aside_text",
        "index",
        "equation",
    ],
)
def test_output_normalization_preserves_new_flash_types(block_type: str) -> None:
    """验证 Flash 文本语义类型和公式类型不会在归一化阶段退回 text。"""

    block = pipeline._normalize_output_block(
        {"type": block_type, "bbox": (10.0, 20.0, 40.0, 50.0), "angle": 0, "content": "value"},
        (100.0, 100.0),
    )

    assert block is not None
    assert block["type"] == block_type


def test_output_normalization_keeps_empty_equation_but_drops_empty_text() -> None:
    """验证纯矢量公式可保留空 content，普通空文本仍不会进入 model_list。"""

    equation = pipeline._normalize_output_block(
        {"type": "equation", "bbox": (10.0, 20.0, 40.0, 50.0), "angle": 0, "content": ""},
        (100.0, 100.0),
    )
    text = pipeline._normalize_output_block(
        {"type": "text", "bbox": (10.0, 20.0, 40.0, 50.0), "angle": 0, "content": ""},
        (100.0, 100.0),
    )

    assert equation == {
        "type": "equation",
        "bbox": [0.1, 0.2, 0.4, 0.5],
        "angle": 0,
        "content": "",
    }
    assert text is None


def test_index_is_claimed_before_formula_anchor_growth() -> None:
    """验证目录右缘页码先形成完整 index，不再扩张为重叠公式。"""

    heading = _text_line(
        "contents",
        (40.0, 5.0, 60.0, 10.0),
        0,
        effective_height=5.0,
    )
    rows = []
    source_index = 1
    for row_index, top in enumerate((20.0, 30.0, 40.0, 50.0, 60.0, 70.0)):
        visual_row_id = 10 + row_index
        rows.extend(
            [
                _text_line(
                    f"entry-{row_index}",
                    (10.0, top, 80.0, top + 5.0),
                    source_index,
                    visual_row_id=visual_row_id,
                    run_index=0,
                    split_from_row=True,
                    effective_height=5.0,
                ),
                _text_line(
                    str(row_index + 1),
                    (90.0, top, 95.0, top + 5.0),
                    source_index + 1,
                    visual_row_id=visual_row_id,
                    run_index=1,
                    split_from_row=True,
                    effective_height=5.0,
                ),
            ]
        )
        source_index += 2
    page = _prepared_text_page(heading, *rows)

    blocks = pipeline._finalize_prepared_page(page, page_index=1)

    assert [block["type"] for block in blocks].count("index") == 1
    assert not [block for block in blocks if block["type"] == "equation"]
    assert next(block for block in blocks if block["type"] == "index")[
        "content"
    ].count("entry-") == 6
    assert next(block for block in blocks if block["content"] == "contents")[
        "type"
    ] == "paragraph_title"


def test_numbered_formula_rows_are_not_claimed_as_index() -> None:
    """验证连续编号公式先保留公式语义，不被无标题目录候选吞并。"""

    body_font = ("Body", 0)
    lines = [
        _text_line(
            "ordinary body row",
            (0.0, 10.0 * index, 100.0, 10.0 * index + 7.0),
            index,
            effective_height=7.0,
            font_signature=body_font,
            font_coverage=1.0,
        )
        for index in range(10)
    ]
    source_index = 10
    for row_index, top in enumerate(
        (102.0, 113.0, 124.0, 135.0, 146.0, 157.0)
    ):
        visual_row_id = 100 + row_index
        lines.extend(
            [
                _text_line(
                    f"x{row_index}=y{row_index}",
                    (20.0, top, 55.0, top + 7.0),
                    source_index,
                    visual_row_id=visual_row_id,
                    run_index=0,
                    split_from_row=True,
                    effective_height=7.0,
                    font_signature=("Math", 0),
                    font_coverage=0.6,
                ),
                _text_line(
                    f"({row_index + 1})",
                    (91.0, top, 100.0, top + 7.0),
                    source_index + 1,
                    visual_row_id=visual_row_id,
                    run_index=1,
                    split_from_row=True,
                    effective_height=7.0,
                ),
            ]
        )
        source_index += 2

    blocks = pipeline._finalize_prepared_page(
        _prepared_text_page(*lines, page_size=(100.0, 200.0)),
        page_index=1,
    )

    assert sum(block["type"] == "equation" for block in blocks) == 6
    assert not [block for block in blocks if block["type"] == "index"]
