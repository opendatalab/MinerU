from __future__ import annotations


import pytest


from mineru.backend.flash.native_pdf import (
    graphics,
    models,
    pipeline,
)
from mineru.utils.pdf_document import PDFPathInfo


from _flash_pdf_test_utils import (
    _text_line,
)


def _drawing_axis_line(
    orientation: str,
    bbox: tuple[float, float, float, float],
) -> models._AxisLine:
    """构造图形容器测试使用的 PDF 绘图线。"""

    return models._AxisLine(
        bbox=bbox,
        width=0.5,
        orientation=orientation,  # type: ignore[arg-type]
    )


def _path_info(
    bbox: tuple[float, float, float, float],
    source_index: int,
    *,
    segment_count: int = 5,
    fill_visible: bool = True,
    stroke_visible: bool = False,
    form_depth: int = 0,
) -> PDFPathInfo:
    """构造强图形核心测试使用的根层 PDF Path。"""

    return PDFPathInfo(
        bbox=bbox,
        segment_count=segment_count,
        fill_visible=fill_visible,
        stroke_visible=stroke_visible,
        form_depth=form_depth,
        source_index=source_index,
    )


def _graphic_source_fixture() -> models._PageSource:
    """构造双框图形、六个标签、拆分 caption 与邻近正文。"""

    lines = [
        _text_line("p = (x, y)", (378.0, 302.0, 421.0, 312.0), 0, visual_row_id=10),
        _text_line("pbar = (xbar, y)", (445.0, 302.0, 488.0, 312.0), 1, visual_row_id=10),
        _text_line("dp", (382.0, 282.0, 403.0, 292.0), 2, visual_row_id=11),
        _text_line("x", (468.0, 282.0, 475.0, 292.0), 3, visual_row_id=11),
        _text_line("Left camera", (360.0, 339.0, 411.0, 350.0), 4, visual_row_id=12),
        _text_line("Right camera", (458.0, 339.0, 509.0, 350.0), 5, visual_row_id=12),
        _text_line(
            "Figure 1: a deliberately long caption that must remain",
            (312.0, 359.0, 500.0, 371.0),
            6,
            visual_row_id=20,
            split_from_row=True,
        ),
        _text_line(
            "independent.",
            (510.0, 359.0, 563.0, 371.0),
            7,
            visual_row_id=20,
            split_from_row=True,
        ),
        _text_line("Nearby prose", (370.0, 225.0, 430.0, 237.0), 8, visual_row_id=9),
    ]
    drawing_lines = [
        _drawing_axis_line("horizontal", (348.0, 272.0, 430.0, 272.5)),
        _drawing_axis_line("horizontal", (348.0, 331.5, 430.0, 332.0)),
        _drawing_axis_line("vertical", (348.0, 272.0, 348.5, 332.0)),
        _drawing_axis_line("vertical", (429.5, 272.0, 430.0, 332.0)),
        _drawing_axis_line("horizontal", (436.0, 272.0, 526.0, 272.5)),
        _drawing_axis_line("horizontal", (436.0, 331.5, 526.0, 332.0)),
        _drawing_axis_line("vertical", (436.0, 272.0, 436.5, 332.0)),
        _drawing_axis_line("vertical", (525.5, 272.0, 526.0, 332.0)),
    ]
    return models._PageSource(
        page_size=(612.0, 792.0),
        lines=lines,
        chars=[],
        drawing_lines=drawing_lines,
    )


def test_double_box_graphic_claims_six_labels_but_not_caption_or_body() -> None:
    """验证双框图形整行聚合六个标签，拆分 caption 和邻近正文均不被部分认领。"""

    blocks, claimed = graphics._build_graphic_like_blocks(
        _graphic_source_fixture(),
        [],
        set(),
    )

    assert len(blocks) == 1
    assert claimed == set(range(6))
    assert blocks[0]["type"] == "image"
    for expected_text in ("dp", "x", "p = (x, y)", "pbar = (xbar, y)", "Left camera", "Right camera"):
        assert expected_text in blocks[0]["content"]
    assert blocks[0]["content"].count("Left camera") == 1
    assert blocks[0]["content"].count("Right camera") == 1
    assert "Figure 1" not in blocks[0]["content"]
    assert "Nearby prose" not in blocks[0]["content"]


def test_graphic_label_accepts_only_near_diagonal_corner_text() -> None:
    """验证无轴向重叠的短标签仅在严格角部距离内归入图形。"""

    core_bbox = (20.0, 20.0, 80.0, 80.0)
    near_corner = _text_line("unit", (5.0, 5.0, 15.0, 15.0), 0)
    distant_corner = _text_line("unit", (-5.0, -5.0, 5.0, 5.0), 1)
    long_title = _text_line(
        "deliberately wide title",
        (0.0, 5.0, 80.0, 15.0),
        2,
    )

    assert graphics._is_graphic_label_member(
        near_corner,
        core_bbox,
        10.0,
    )
    assert not graphics._is_graphic_label_member(
        distant_corner,
        core_bbox,
        10.0,
    )
    assert not graphics._is_graphic_label_member(
        long_title,
        core_bbox,
        10.0,
    )


def test_materialized_table_bbox_has_priority_over_graphic_candidate() -> None:
    """验证绘图组件与成功表格框重叠时跳过图形容器并保留全部文本身份。"""

    blocks, claimed = graphics._build_graphic_like_blocks(
        _graphic_source_fixture(),
        [(340.0, 265.0, 535.0, 355.0)],
        set(),
    )

    assert blocks == []
    assert claimed == set()


def test_complex_path_container_builds_graphic_without_drawing_lines() -> None:
    """验证大 Path 容器和内部二维复杂轮廓可直接形成图形核心。"""

    source = models._PageSource(
        page_size=(100.0, 100.0),
        lines=[
            _text_line("10", (15.0, 20.0, 25.0, 25.0), 0, visual_row_id=1),
            _text_line("label", (30.0, 55.0, 50.0, 60.0), 1, visual_row_id=2),
        ],
        chars=[],
        drawing_lines=[],
        path_infos=[
            _path_info((10.0, 10.0, 90.0, 70.0), 0),
            _path_info((20.0, 20.0, 75.0, 50.0), 1, segment_count=12),
            _path_info((25.0, 25.0, 35.0, 35.0), 2),
            _path_info((40.0, 25.0, 50.0, 40.0), 3),
            _path_info((55.0, 20.0, 65.0, 45.0), 4),
        ],
    )

    core_bboxes = graphics._detect_strong_graphic_bboxes(source)
    blocks, claimed = graphics._build_graphic_like_blocks(
        source,
        [],
        set(),
        core_bboxes,
    )

    assert core_bboxes == [(10.0, 10.0, 90.0, 70.0)]
    assert claimed == {0, 1}
    assert blocks[0]["bbox"] == (10.0, 10.0, 90.0, 70.0)
    assert blocks[0]["content"] == "10\nlabel"


def test_strong_graphic_core_binds_only_to_unique_containing_lane() -> None:
    """验证单栏强图形不会吸收邻栏文本，而真正跨栏核心仍保持跨栏。"""

    lanes = [
        models._TextLane(left=50.0, right=290.0),
        models._TextLane(left=305.0, right=545.0),
    ]

    assert graphics._strong_graphic_lane_index(
        (73.0, 516.0, 281.0, 655.0),
        lanes,
        10.0,
    ) == 0
    assert graphics._strong_graphic_lane_index(
        (73.0, 516.0, 520.0, 655.0),
        lanes,
        10.0,
    ) == -1


def test_axis_pair_requires_internal_two_dimensional_complex_path() -> None:
    """验证相交坐标轴须有二维复杂曲线支撑，规则矩形行带不会误报图形。"""

    source = models._PageSource(
        page_size=(100.0, 100.0),
        lines=[
            _text_line("0", (12.0, 65.0, 17.0, 70.0), 0),
            _text_line("x", (78.0, 72.0, 83.0, 77.0), 1),
        ],
        chars=[],
        drawing_lines=[],
        path_infos=[
            _path_info((20.0, 20.0, 21.0, 70.0), 0, segment_count=2, fill_visible=False, stroke_visible=True),
            _path_info((20.0, 69.0, 80.0, 70.0), 1, segment_count=2, fill_visible=False, stroke_visible=True),
            _path_info((25.0, 30.0, 75.0, 60.0), 2, segment_count=12, fill_visible=False, stroke_visible=True),
        ],
    )

    assert graphics._detect_strong_graphic_bboxes(source) == [
        (20.0, 20.0, 80.0, 70.0)
    ]

    source.path_infos[2] = _path_info(
        (25.0, 30.0, 75.0, 31.0),
        2,
        segment_count=12,
        fill_visible=False,
        stroke_visible=True,
    )
    assert graphics._detect_strong_graphic_bboxes(source) == []


def test_form_image_claims_internal_text_and_small_table_but_not_caption() -> None:
    """验证有效 Form 吞并图内文字与小表候选，外部 caption 仍保留。"""

    form_bbox = (10.0, 10.0, 90.0, 65.0)
    source = models._PageSource(
        page_size=(100.0, 100.0),
        lines=[
            _text_line("inside row one", (15.0, 20.0, 70.0, 30.0), 0, visual_row_id=1),
            _text_line("inside row two", (15.0, 35.0, 80.0, 45.0), 1, visual_row_id=2),
            _text_line("Figure 1: outside", (10.0, 70.0, 80.0, 80.0), 2, visual_row_id=3),
        ],
        chars=[],
        drawing_lines=[
            _drawing_axis_line("horizontal", (20.0, 15.0, 80.0, 15.5)),
            _drawing_axis_line("horizontal", (20.0, 55.0, 80.0, 55.5)),
            _drawing_axis_line("vertical", (20.0, 15.0, 20.5, 55.0)),
            _drawing_axis_line("vertical", (79.5, 15.0, 80.0, 55.0)),
        ],
        form_bboxes=[form_bbox],
    )

    selected = graphics._select_form_image_bboxes(source)
    blocks, claimed = graphics._build_form_image_blocks(source, selected, set())

    assert selected == [form_bbox]
    assert graphics._form_supersedes_nested_bbox(
        form_bbox,
        (20.0, 20.0, 40.0, 35.0),
    )
    assert not graphics._form_supersedes_nested_bbox(
        form_bbox,
        (15.0, 15.0, 85.0, 60.0),
    )
    assert claimed == {0, 1}
    assert blocks == [
        {
            "type": "image",
            "bbox": form_bbox,
            "angle": 0,
            "content": "inside row one\ninside row two",
        }
    ]


def test_form_image_bbox_tightens_to_supported_internal_evidence() -> None:
    """验证含充分嵌套 Path 的 Form 去除空白边缘，文本轻微越界仍纳入证据并裁到页面。"""

    source = models._PageSource(
        page_size=(120.0, 120.0),
        lines=[
            _text_line("top label", (20.0, 25.0, 70.0, 34.0), 0),
            _text_line("bottom label", (25.0, 68.0, 80.0, 81.0), 1),
        ],
        chars=[],
        drawing_lines=[
            _drawing_axis_line("horizontal", (20.0, 25.0, 85.0, 25.5)),
            _drawing_axis_line("horizontal", (20.0, 75.0, 85.0, 75.5)),
            _drawing_axis_line("vertical", (20.0, 25.0, 20.5, 75.0)),
            _drawing_axis_line("vertical", (84.5, 25.0, 85.0, 75.0)),
        ],
        form_bboxes=[(10.0, 10.0, 100.0, 80.0)],
        path_infos=[
            _path_info((20.0, 25.0, 80.0, 70.0), 0, form_depth=1),
            _path_info((25.0, 30.0, 85.0, 75.0), 1, form_depth=1),
        ],
    )

    assert graphics._select_form_image_bboxes(source) == [
        (20.0, 25.0, 85.0, 81.0)
    ]


def test_graphic_label_absorbs_short_axis_title_but_rejects_long_caption() -> None:
    """验证上下坐标轴标题可放宽到八倍行高，长图注仍被图形容器拒绝。"""

    core_bbox = (20.0, 30.0, 140.0, 80.0)
    top_axis_title = _text_line(
        "axis title",
        (20.0, 17.0, 70.0, 25.0),
        0,
        effective_height=8.0,
    )
    bottom_axis_title = _text_line(
        "axis title",
        (80.0, 85.0, 115.0, 91.0),
        1,
        effective_height=6.0,
    )
    long_caption = _text_line(
        "a deliberately long figure caption",
        (20.0, 85.0, 140.0, 93.0),
        2,
        effective_height=8.0,
    )

    assert graphics._is_graphic_label_member(
        top_axis_title,
        core_bbox,
        8.0,
        margin_scale=1.0,
    )
    assert graphics._is_graphic_label_member(
        bottom_axis_title,
        core_bbox,
        8.0,
        margin_scale=1.0,
    )
    assert not graphics._is_graphic_label_member(
        long_caption,
        core_bbox,
        8.0,
        margin_scale=1.0,
    )


def _inline_raster_sequence_source() -> models._PageSource:
    """构造四张点阵图、三个同行间隔符及其右侧三行正文。"""

    return models._PageSource(
        page_size=(100.0, 100.0),
        lines=[
            _text_line(
                "、",
                (20.0, 22.0, 22.0, 25.0),
                0,
                visual_row_id=7,
                run_index=0,
                split_from_row=True,
            ),
            _text_line(
                "、",
                (32.0, 22.0, 34.0, 25.0),
                1,
                visual_row_id=7,
                run_index=1,
                split_from_row=True,
            ),
            _text_line(
                "、",
                (48.0, 22.0, 50.0, 25.0),
                2,
                visual_row_id=7,
                run_index=2,
                split_from_row=True,
            ),
            _text_line(
                "尾随文字",
                (60.0, 22.0, 90.0, 25.0),
                3,
                visual_row_id=7,
                run_index=3,
                split_from_row=True,
            ),
            _text_line("正文第一行", (10.0, 27.0, 90.0, 30.0), 4, visual_row_id=8),
            _text_line("正文第二行", (10.0, 32.0, 50.0, 35.0), 5, visual_row_id=9),
        ],
        chars=[],
        drawing_lines=[],
        image_bboxes=[
            (10.0, 20.0, 20.0, 24.0),
            (22.0, 20.0, 32.0, 24.0),
            (34.0, 20.0, 48.0, 24.0),
            (50.0, 20.0, 60.0, 24.0),
        ],
    )


def test_raster_image_threshold_accepts_point_38_percent_only() -> None:
    """验证页面面积达到 0.38% 时准入，略低于阈值的孤立图片仍过滤。"""

    source = models._PageSource(
        page_size=(100.0, 100.0),
        lines=[],
        chars=[],
        drawing_lines=[],
        image_bboxes=[
            (0.0, 0.0, 9.5, 4.0),
            (20.0, 0.0, 29.49, 4.0),
        ],
    )

    blocks, claimed = graphics._build_raster_image_blocks(source, [], set())

    assert [block["bbox"] for block in blocks] == [(0.0, 0.0, 9.5, 4.0)]
    assert claimed == set()


def test_signature_images_bypass_raster_threshold_and_deduplicate_same_geometry() -> None:
    """验证小签名仍输出，且签名间及签名与点阵图的同框候选只保留一次。"""

    source = models._PageSource(
        page_size=(1000.0, 1000.0),
        lines=[],
        chars=[],
        drawing_lines=[],
        image_bboxes=[(100.0, 100.0, 200.0, 200.0)],
        signature_bboxes=[
            (10.0, 10.0, 30.0, 30.0),
            (10.2, 10.1, 30.1, 30.2),
            (100.0, 100.0, 200.0, 200.0),
        ],
    )

    blocks, claimed = graphics._build_raster_image_blocks(source, [], set())

    assert [block["bbox"] for block in blocks] == [
        (10.0, 10.0, 30.0, 30.0),
        (100.0, 100.0, 200.0, 200.0),
    ]
    assert claimed == set()


def test_inline_raster_images_and_gap_runs_form_one_composite_image() -> None:
    """验证四张已准入图片与三个同行间隔符合成一个复合 image。"""

    source = _inline_raster_sequence_source()

    blocks, claimed = graphics._build_raster_image_blocks(source, [], set())

    assert len(blocks) == 1
    assert blocks[0]["bbox"] == (10.0, 20.0, 60.0, 25.0)
    assert blocks[0]["content"].replace(" ", "") == "、、、"
    assert blocks[0]["_inline_visual_row_id"] == 7
    assert claimed == {0, 1, 2}
    assert 3 not in claimed


@pytest.mark.parametrize(
    "failure_mode",
    [
        "misaligned",
        "wide_gap",
        "missing_gap",
        "inconsistent_row",
        "too_few_images",
        "container_bridge",
    ],
)
def test_inline_raster_composite_requires_complete_spatial_sequence(
    failure_mode: str,
) -> None:
    """验证图片或间隔符结构不完整时保留独立图片而不生成复合容器。"""

    source = _inline_raster_sequence_source()
    container_blocks: list[dict[str, object]] = []
    if failure_mode == "misaligned":
        source.image_bboxes[1] = (22.0, 10.0, 32.0, 14.0)
    elif failure_mode == "wide_gap":
        source.image_bboxes[2] = (42.0, 20.0, 56.0, 24.0)
        source.image_bboxes[3] = (58.0, 20.0, 68.0, 24.0)
        source.lines[2].bbox = (56.0, 22.0, 58.0, 25.0)
    elif failure_mode == "missing_gap":
        source.lines = [line for line in source.lines if line.source_index != 1]
    elif failure_mode == "inconsistent_row":
        source.lines[1].visual_row_id = 8
    elif failure_mode == "too_few_images":
        source.image_bboxes = source.image_bboxes[:2]
    else:
        container_blocks = [
            {
                "type": "table",
                "bbox": (32.5, 20.0, 33.5, 25.0),
                "angle": 0,
                "content": "",
            }
        ]

    blocks, claimed = graphics._build_raster_image_blocks(
        source,
        container_blocks,
        set(),
    )

    assert len(blocks) == len(source.image_bboxes)
    assert not [block for block in blocks if "_inline_visual_row_id" in block]
    assert claimed == set()


def test_inline_composite_sorts_before_overlapping_multiline_text() -> None:
    """验证复合图片与含同行首行的多行正文绑定后始终先输出图片。"""

    blocks = [
        {
            "type": "image",
            "bbox": (10.0, 20.0, 60.0, 25.0),
            "angle": 0,
            "content": "、、、",
            "_inline_visual_row_id": 7,
        },
        {
            "type": "text",
            "bbox": (10.0, 22.0, 90.0, 35.0),
            "angle": 0,
            "content": "尾随文字正文第一行正文第二行",
            "_visual_row_ids": {7, 8, 9},
            "_local_line_bboxes": [
                (60.0, 22.0, 90.0, 25.0),
                (10.0, 27.0, 90.0, 30.0),
                (10.0, 32.0, 50.0, 35.0),
            ],
        },
        {
            "type": "paragraph_title",
            "bbox": (10.0, 40.0, 30.0, 44.0),
            "angle": 0,
            "content": "下一节",
        },
    ]

    sorted_blocks = pipeline._sort_blocks_with_visual_row_groups(
        blocks,
        (100.0, 100.0),
    )

    assert [block["type"] for block in sorted_blocks] == [
        "image",
        "text",
        "paragraph_title",
    ]


def test_raster_images_filter_small_objects_avoid_containers_and_claim_text_once() -> None:
    """验证点阵图过滤、容器优先、空 content 和重叠对象的唯一文本归属。"""

    source = models._PageSource(
        page_size=(100.0, 100.0),
        lines=[
            _text_line("inside", (25.0, 25.0, 35.0, 35.0), 0, visual_row_id=1),
            _text_line("outside caption", (10.0, 52.0, 50.0, 60.0), 1, visual_row_id=2),
            _text_line("covered container text", (20.0, 70.0, 40.0, 80.0), 2, visual_row_id=3),
        ],
        chars=[],
        drawing_lines=[],
        image_bboxes=[
            (10.0, 10.0, 50.0, 50.0),
            (20.0, 20.0, 40.0, 40.0),
            (60.0, 10.0, 90.0, 40.0),
            (0.0, 90.0, 9.0, 94.0),
            (10.0, 60.0, 50.0, 90.0),
        ],
    )

    blocks, claimed = graphics._build_raster_image_blocks(
        source,
        [{"type": "table", "bbox": (10.0, 60.0, 50.0, 90.0), "angle": 0, "content": "table"}],
        set(),
    )

    assert len(blocks) == 3
    assert all(block["type"] == "image" and block["angle"] == 0 for block in blocks)
    assert [block["bbox"] for block in blocks] == [
        (10.0, 10.0, 50.0, 50.0),
        (60.0, 10.0, 90.0, 40.0),
        (20.0, 20.0, 40.0, 40.0),
    ]
    assert [block["content"] for block in blocks] == ["", "", "inside"]
    assert claimed == {0}


def test_raster_image_content_is_removed_from_text_and_empty_image_page_is_kept() -> None:
    """验证图内文本只进入 image，caption 保持 text，纯图片页仍输出空 content。"""

    source = models._PageSource(
        page_size=(100.0, 100.0),
        lines=[
            _text_line("inside row one", (20.0, 20.0, 50.0, 30.0), 0, visual_row_id=1),
            _text_line("inside row two", (20.0, 35.0, 50.0, 45.0), 1, visual_row_id=2),
            _text_line("Figure 1: outside caption", (10.0, 70.0, 80.0, 80.0), 2, visual_row_id=3),
        ],
        chars=[],
        drawing_lines=[],
        image_bboxes=[(10.0, 10.0, 60.0, 60.0)],
    )

    blocks = pipeline._analyze_page_source(source)
    image_block = next(block for block in blocks if block["type"] == "image")
    text_block = next(block for block in blocks if block["type"] == "text")

    assert image_block["content"] == "inside row one\ninside row two"
    assert text_block["content"] == "Figure 1: outside caption"
    assert sum("inside row" in block["content"] for block in blocks) == 1

    empty_page_blocks = pipeline._analyze_page_source(
        models._PageSource(
            page_size=(100.0, 100.0),
            lines=[],
            chars=[],
            drawing_lines=[],
            image_bboxes=[(10.0, 10.0, 60.0, 60.0)],
        )
    )
    assert empty_page_blocks == [
        {
            "type": "image",
            "bbox": [0.1, 0.1, 0.6, 0.6],
            "angle": 0,
            "content": "",
        }
    ]
