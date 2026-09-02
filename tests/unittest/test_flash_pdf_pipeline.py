from __future__ import annotations

import ast
import inspect

import pytest

from mineru.model.flash import PdfModel
from mineru.model.flash.pdf import (
    auxiliary_text,
    char_geometry,
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
    text_styles,
    titles,
    visual_annotations,
)
from mineru.model.flash.pdf.document import PDFImageInfo

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
        **_kwargs: object,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], set[int]]:
        """记录表格物化阶段可见的来源行。"""

        assert candidates == []
        observed["materialize"] = [line.source_index for line in materialization_source.lines]
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


def test_prepare_page_resplits_repaired_cross_column_row_before_text_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证生产页面准备会按修复字符框重切跨栏粗行并分配唯一来源序号。"""

    positions = (0.0, 6.0, 12.0, 18.0, 62.0, 68.0, 74.0, 80.0)
    chars = [
        {
            "char": character,
            "bbox": (position, 10.0, position + 20.0, 20.0),
            "font": {
                "name": "ABCDEF+Body",
                "flags": 0,
                "weight": 400,
                "size": 10.0,
            },
            "char_idx": index,
        }
        for index, (character, position) in enumerate(
            zip("ABCDEFGH", positions, strict=True),
        )
    ]
    line = models._LineItem(
        text="ABCDEFGH",
        bbox=(0.0, 10.0, 100.0, 20.0),
        angle=0,
        source_index=5,
        chars=chars,  # type: ignore[arg-type]
        visual_row_id=3,
        effective_height=10.0,
        em_height=10.0,
    )
    plan = char_geometry.DocumentGeometryPlan(
        char_repairs={
            (0, index): char_geometry.CharLayoutGeometry(
                source_bbox=tuple(float(value) for value in char["bbox"]),  # type: ignore[arg-type]
                tight_bbox=(position, 10.0, position + 5.0, 20.0),
                origin=(position, 20.0),
                layout_bbox=(position, 10.0, position + 5.0, 20.0),
                ink_bbox=(position, 10.0, position + 5.0, 20.0),
                baseline=20.0,
                advance=6.0,
                em_height=10.0,
                x_state="abnormal",
                y_state="healthy",
                confidence=1.0,
            )
            for index, (char, position) in enumerate(
                zip(chars, positions, strict=True),
            )
        },
    )
    source = models._PageSource(
        page_size=(100.0, 100.0),
        lines=[line],
        chars=chars,  # type: ignore[arg-type]
        drawing_lines=[],
    )
    observed_next_indices: list[int] = []
    untouched_style = text_styles.PDFTextStyleLine(
        bbox=(0.0, 30.0, 20.0, 40.0),
        text="plain",
        style_ranges=(),
        source_index=4,
    )
    style_lines = [
        untouched_style,
        text_styles.PDFTextStyleLine(
            bbox=line.bbox,
            text="ABCDEFGH",
            style_ranges=(
                text_styles.PDFTextStyleRange(0, 2, ("bold",)),
                text_styles.PDFTextStyleRange(2, 6, ("bold", "underline")),
                text_styles.PDFTextStyleRange(6, 8, ("underline",)),
            ),
            source_index=5,
        ),
    ]
    link_lines = [
        text_styles.PDFTextLinkLine(
            bbox=line.bbox,
            text="ABCDEFGH",
            link_ranges=(
                text_styles.PDFTextLinkRange(
                    2,
                    6,
                    "https://example.test/split",
                ),
            ),
            source_index=5,
        ),
    ]

    def record_graphic_split_start(
        lines: list[models._LineItem],
        *_args: object,
        source_index_start: int,
    ) -> list[models._LineItem]:
        """记录跨栏重切后传给后续 graphic split 的下一来源序号。"""

        observed_next_indices.append(source_index_start)
        return lines

    monkeypatch.setattr(
        pipeline,
        "_split_parallel_graphic_rule_rows",
        record_graphic_split_start,
    )

    prepared = pipeline._prepare_page_source(
        source,
        geometry_plan=plan,
        page_index=0,
        style_lines=style_lines,
        link_lines=link_lines,
    )

    assert [member.text for member in prepared.remaining_lines] == ["ABCD", "EFGH"]
    assert [member.source_index for member in prepared.remaining_lines] == [5, 6]
    assert all(member.split_from_row for member in prepared.remaining_lines)
    assert observed_next_indices == [7]
    assert style_lines[0] is untouched_style
    assert style_lines[1:] == [
        text_styles.PDFTextStyleLine(
            bbox=(0.0, 10.0, 23.0, 20.0),
            text="ABCD",
            style_ranges=(
                text_styles.PDFTextStyleRange(0, 2, ("bold",)),
                text_styles.PDFTextStyleRange(2, 4, ("bold", "underline")),
            ),
            source_index=5,
        ),
        text_styles.PDFTextStyleLine(
            bbox=(62.0, 10.0, 85.0, 20.0),
            text="EFGH",
            style_ranges=(
                text_styles.PDFTextStyleRange(0, 2, ("bold", "underline")),
                text_styles.PDFTextStyleRange(2, 4, ("underline",)),
            ),
            source_index=6,
        ),
    ]
    assert link_lines == [
        text_styles.PDFTextLinkLine(
            bbox=(0.0, 10.0, 23.0, 20.0),
            text="ABCD",
            link_ranges=(
                text_styles.PDFTextLinkRange(
                    2,
                    4,
                    "https://example.test/split",
                ),
            ),
            source_index=5,
        ),
        text_styles.PDFTextLinkLine(
            bbox=(62.0, 10.0, 85.0, 20.0),
            text="EFGH",
            link_ranges=(
                text_styles.PDFTextLinkRange(
                    0,
                    2,
                    "https://example.test/split",
                ),
            ),
            source_index=6,
        ),
    ]
    split_blocks = [
        {
            "type": "text",
            "bbox": member.bbox,
            "content": member.text,
        }
        for member in prepared.remaining_lines
    ]
    assert {
        block_index: [line.source_index for line in lines]
        for block_index, lines in text_styles._assign_lines_to_blocks(
            split_blocks,
            style_lines[1:],
            source.page_size,
        ).items()
    } == {0: [5], 1: [6]}
    assert {
        block_index: [line.source_index for line in lines]
        for block_index, lines in text_styles._assign_lines_to_blocks(
            split_blocks,
            link_lines,
            source.page_size,
        ).items()
    } == {0: [5], 1: [6]}


def test_prepare_page_realigns_unsplit_repaired_text_evidence() -> None:
    """验证普通修复行未重切时，样式和链接 evidence 仍使用最终布局框。"""

    line = _text_line("linked", (0.0, 80.0, 200.0, 100.0), 5)
    source = models._PageSource(
        page_size=(200.0, 200.0),
        lines=[line],
        chars=[],
        drawing_lines=[],
    )
    repaired_bbox = (20.0, 80.0, 60.0, 100.0)
    plan = char_geometry.DocumentGeometryPlan(
        line_repairs={
            (0, 5): char_geometry.LineGeometryRepair(
                source_bbox=line.bbox,
                layout_bbox=repaired_bbox,
                ink_bbox=repaired_bbox,
                baseline=100.0,
                em_height=20.0,
                state="repair_x",
            )
        }
    )
    style_lines = [
        text_styles.PDFTextStyleLine(
            bbox=line.bbox,
            text=line.text,
            style_ranges=(text_styles.PDFTextStyleRange(0, len(line.text), ("bold",)),),
            source_index=line.source_index,
        )
    ]
    link_lines = [
        text_styles.PDFTextLinkLine(
            bbox=line.bbox,
            text=line.text,
            link_ranges=(text_styles.PDFTextLinkRange(0, len(line.text), "https://example.test/repaired"),),
            source_index=line.source_index,
        )
    ]

    prepared = pipeline._prepare_page_source(
        source,
        geometry_plan=plan,
        page_index=0,
        style_lines=style_lines,
        link_lines=link_lines,
    )

    assert [item.bbox for item in prepared.remaining_lines] == [repaired_bbox]
    assert [item.bbox for item in style_lines] == [repaired_bbox]
    assert [item.bbox for item in link_lines] == [repaired_bbox]
    block = {"type": "text", "bbox": repaired_bbox, "content": line.text}
    assert text_styles._assign_lines_to_blocks([block], style_lines, source.page_size)
    assert text_styles._assign_lines_to_blocks([block], link_lines, source.page_size)


def _repeated_separator_source(
    header_text: str,
    *,
    connected_grid: bool = False,
) -> models._PageSource:
    """构造重复页首横线，并可把该横线作为闭合表格的真实顶边。"""

    lines = [
        _text_line(header_text, (20.0, 15.0, 180.0, 25.0), 0),
        _text_line(f"{header_text} detail", (20.0, 30.0, 180.0, 40.0), 1),
    ]
    drawing_lines = [
        models._AxisLine(
            bbox=(10.0, 50.0, 190.0, 50.1),
            width=0.1,
            orientation="horizontal",
        )
    ]
    if connected_grid:
        lines.extend(
            [
                _text_line("left cell", (20.0, 65.0, 80.0, 75.0), 2),
                _text_line("right cell", (110.0, 65.0, 170.0, 75.0), 3),
            ]
        )
        drawing_lines.extend(
            [
                models._AxisLine((10.0, 90.0, 190.0, 90.1), 0.1, "horizontal"),
                models._AxisLine((10.0, 50.0, 10.1, 90.0), 0.1, "vertical"),
                models._AxisLine((100.0, 50.0, 100.1, 90.0), 0.1, "vertical"),
                models._AxisLine((189.9, 50.0, 190.0, 90.0), 0.1, "vertical"),
            ]
        )
    return models._PageSource(
        page_size=(200.0, 400.0),
        lines=lines,
        chars=[],
        drawing_lines=drawing_lines,
    )


def test_repeated_header_separator_requires_repeated_header_text() -> None:
    """验证只有横线重复、而上方普通文本不重复时不会移除表格规则。"""

    sources = [_repeated_separator_source(text) for text in ("alpha banner", "beta notice", "gamma heading")]

    assert pipeline._detect_repeated_header_separator_bboxes(sources) == [set(), set(), set()]


def test_repeated_header_separator_supports_alternating_headers() -> None:
    """验证奇偶页各自重复的刊头可以共同确认同一页眉分隔线。"""

    sources = [_repeated_separator_source("even journal" if page_index % 2 == 0 else "odd article") for page_index in range(4)]
    expected = {(10.0, 50.0, 190.0, 50.1)}

    assert pipeline._detect_repeated_header_separator_bboxes(sources) == [expected] * 4


def test_repeated_table_top_rule_is_not_header_separator() -> None:
    """验证重复表单的闭合网格顶边不会因上方重复标题而被删除。"""

    sources = [_repeated_separator_source("repeated form", connected_grid=True) for _page_index in range(3)]

    assert pipeline._detect_repeated_header_separator_bboxes(sources) == [set(), set(), set()]
    assert [candidate.bbox for candidate in tables._detect_table_candidates(sources[0])] == [(10.0, 50.0, 190.0, 90.1)]


def test_table_detection_excludes_confirmed_masthead_separator() -> None:
    """验证页首通栏分隔线不与下方真表格边界组成巨型候选。"""

    source = models._PageSource(
        page_size=(200.0, 300.0),
        lines=[
            _text_line("left masthead", (10.0, 10.0, 60.0, 20.0), 0),
            _text_line("center masthead", (70.0, 10.0, 130.0, 20.0), 1),
            _text_line("body", (10.0, 50.0, 190.0, 60.0), 2),
        ],
        chars=[],
        drawing_lines=[
            models._AxisLine(
                bbox=(10.0, 30.0, 190.0, 31.0),
                width=1.0,
                orientation="horizontal",
            ),
            models._AxisLine(
                bbox=(10.0, 100.0, 190.0, 101.0),
                width=1.0,
                orientation="horizontal",
            ),
        ],
    )

    retained = pipeline._table_detection_drawing_lines(
        source,
        {(10.0, 30.0, 190.0, 31.0)},
    )

    assert [line.bbox for line in retained] == [(10.0, 100.0, 190.0, 101.0)]


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
            PdfModel,
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
        "ref_text",
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


def test_output_normalization_applies_unicode_content_safety_net() -> None:
    """验证最终 model_list 归一化会兜底清理排版空格和安全零宽字符。"""

    block = pipeline._normalize_output_block(
        {
            "type": "text",
            "bbox": (10.0, 20.0, 40.0, 50.0),
            "angle": 0,
            "content": "alpha\u00a0beta\u200bgamma",
        },
        (100.0, 100.0),
    )

    assert block is not None
    assert block["content"] == "alpha betagamma"


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
    assert next(block for block in blocks if block["type"] == "index")["content"].count("entry-") == 6
    assert next(block for block in blocks if block["content"] == "contents")["type"] == "paragraph_title"


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
    for row_index, top in enumerate((102.0, 113.0, 124.0, 135.0, 146.0, 157.0)):
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
