from __future__ import annotations



from mineru.backend.flash.native_pdf import (
    graphics,
    models,
    pipeline,
)


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


def test_materialized_table_bbox_has_priority_over_graphic_candidate() -> None:
    """验证绘图组件与成功表格框重叠时跳过图形容器并保留全部文本身份。"""

    blocks, claimed = graphics._build_graphic_like_blocks(
        _graphic_source_fixture(),
        [(340.0, 265.0, 535.0, 355.0)],
        set(),
    )

    assert blocks == []
    assert claimed == set()


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
            (0.0, 90.0, 10.0, 94.0),
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


