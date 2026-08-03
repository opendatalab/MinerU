from __future__ import annotations

import pytest

from mineru.backend.flash.native_pdf import pipeline, visual_annotations


_PAGE_SIZE = (200.0, 200.0)


def _text_block(
    content: str,
    bbox: tuple[float, float, float, float],
    *,
    block_type: str = "text",
    row_id: int | None = None,
) -> dict[str, object]:
    """构造带稳定十点行高的独立文本测试块。"""

    return {
        "type": block_type,
        "bbox": bbox,
        "angle": 0,
        "content": content,
        "_line_heights": [10.0],
        "_single_run_row_id": row_id,
        "_lane_interval": (bbox[0], bbox[2]),
        "_lane_is_span": False,
    }


def _visual_block(
    bbox: tuple[float, float, float, float],
    *,
    block_type: str = "image",
) -> dict[str, object]:
    """构造不包含内部文本的视觉主体测试块。"""

    return {
        "type": block_type,
        "bbox": bbox,
        "angle": 0,
        "content": "",
    }


@pytest.mark.parametrize(
    "content",
    [
        "Fig. 2. Overview",
        "Figure A1: Overview",
        "Tab. 2-1 Results",
        "Table IV. Results",
        "Alg. 1 Procedure",
        "Algorithm 2: Procedure",
        "Listing 3 Source code",
        "Chart 4: Revenue",
        "Scheme Ⅴ. Architecture",
        "图２０：系统结构",
        "图表2-1 行业走势",
        "表格三 实验结果",
        "算法 A1：训练流程",
        "程序清单2 示例",
    ],
)
def test_strong_caption_markers_accept_numbered_titles(content: str) -> None:
    """验证中英文强标题标记兼容常用编号写法。"""

    assert visual_annotations._is_strong_caption_text(content)


@pytest.mark.parametrize(
    "content",
    [
        "Figure 2 shows the result.",
        "Table 1, 2 and 3 show the results.",
        "Figure without a number",
        "图2为系统结构。",
        "表1展示实验结果。",
        "图6和图7分别为输入与输出。",
        "图 8（a）、8（b）分别为两个模型的结果。",
        "普通正文中的 Figure 2",
    ],
)
def test_strong_caption_markers_reject_body_references(content: str) -> None:
    """验证正文引用、并列编号和无编号标签不会进入标题候选。"""

    assert not visual_annotations._is_strong_caption_text(content)


@pytest.mark.parametrize(
    "content",
    [
        "Source: survey",
        "Sources: survey",
        "Source(s): survey",
        "Data source: survey",
        "Note: adjusted values",
        "Note(s): adjusted values",
        "资料来源：公司公告",
        "数据来源：统计局",
        "来源：作者整理",
        "注：数值经过调整",
        "备注：数值经过调整",
    ],
)
def test_strong_footnote_markers_require_colon_and_body(content: str) -> None:
    """验证来源与注释强规则要求冒号和实际正文。"""

    assert visual_annotations._is_strong_footnote_text(content)


@pytest.mark.parametrize(
    "content",
    ["Note that the value changes.", "Source material is public.", "注释内容", "来源："],
)
def test_strong_footnote_markers_reject_narrative_text(content: str) -> None:
    """验证无冒号的叙述句和空标记不会进入脚注候选。"""

    assert not visual_annotations._is_strong_footnote_text(content)


@pytest.mark.parametrize(
    ("caption_bbox", "expected_order"),
    [
        ((40.0, 25.0, 80.0, 35.0), ("caption", "image")),
        ((40.0, 85.0, 80.0, 95.0), ("image", "caption")),
        ((5.0, 40.0, 35.0, 80.0), ("caption", "image")),
        ((85.0, 40.0, 115.0, 80.0), ("image", "caption")),
    ],
)
def test_caption_binds_on_all_four_sides(
    caption_bbox: tuple[float, float, float, float],
    expected_order: tuple[str, str],
) -> None:
    """验证上下左右紧邻标题均能绑定，并按相对方向展开。"""

    caption = _text_block("Figure 1: Overview", caption_bbox)
    image = _visual_block((40.0, 40.0, 80.0, 80.0))
    blocks = [caption, image]

    regions = visual_annotations._classify_and_bind_visual_annotations(
        blocks,
        _PAGE_SIZE,
    )

    assert caption["type"] == "caption"
    assert len(regions) == 1
    assert tuple(block["type"] for block in regions[0]) == expected_order


def test_rotated_caption_uses_parent_local_coordinates() -> None:
    """验证九十度页面块会先转入共同局部坐标再判断上方标题。"""

    caption = _text_block("Figure 1: Rotated", (165.0, 40.0, 175.0, 80.0))
    image = _visual_block((120.0, 40.0, 160.0, 80.0))
    caption["angle"] = image["angle"] = 90

    regions = visual_annotations._classify_and_bind_visual_annotations(
        [caption, image],
        _PAGE_SIZE,
    )

    assert caption["type"] == "caption"
    assert regions == [[caption, image]]


def test_caption_rejects_more_than_one_line_height_penetration() -> None:
    """验证边缘轻微浮入主体可接受，深入超过自身一行高则保持 text。"""

    image = _visual_block((40.0, 40.0, 80.0, 80.0))
    accepted = _text_block("Figure 1: Accepted", (40.0, 72.0, 80.0, 82.0))
    accepted_regions = visual_annotations._classify_and_bind_visual_annotations(
        [image, accepted],
        _PAGE_SIZE,
    )

    rejected = _text_block("Figure 2: Rejected", (40.0, 69.0, 80.0, 79.0))
    rejected_regions = visual_annotations._classify_and_bind_visual_annotations(
        [image.copy(), rejected],
        _PAGE_SIZE,
    )

    assert accepted["type"] == "caption"
    assert accepted_regions
    assert rejected["type"] == "text"
    assert rejected_regions == []


def test_footnote_accepts_six_line_gap_but_rejects_farther_text() -> None:
    """验证视觉块下方脚注的六行高距离上限为闭区间。"""

    image = _visual_block((20.0, 20.0, 100.0, 50.0))
    accepted = _text_block("Note: accepted", (20.0, 110.0, 100.0, 120.0))
    accepted_blocks = [image, accepted]
    accepted_regions = visual_annotations._classify_and_bind_visual_annotations(
        accepted_blocks,
        _PAGE_SIZE,
    )

    rejected = _text_block("Note: rejected", (20.0, 110.1, 100.0, 120.1))
    rejected_blocks = [image.copy(), rejected]
    rejected_regions = visual_annotations._classify_and_bind_visual_annotations(
        rejected_blocks,
        _PAGE_SIZE,
    )

    assert accepted["type"] == "footnote"
    assert [block["type"] for block in accepted_regions[0]] == ["image", "footnote"]
    assert rejected["type"] == "text"
    assert rejected_regions == []


def test_existing_visual_footnote_binds_without_content_marker() -> None:
    """验证已有横线小字 footnote 无需再次命中内容规则即可加入区域。"""

    image = _visual_block((20.0, 20.0, 100.0, 60.0))
    footnote = _text_block(
        "calculated by the authors",
        (20.0, 70.0, 100.0, 80.0),
        block_type="footnote",
    )

    regions = visual_annotations._classify_and_bind_visual_annotations(
        [image, footnote],
        _PAGE_SIZE,
    )

    assert [block["type"] for block in regions[0]] == ["image", "footnote"]


def test_multi_panel_images_only_group_when_union_improves_coverage() -> None:
    """验证跨面板标题绑定图片并集，而单图标题不会无故吞并相邻图片。"""

    caption = _text_block("Figure 1: Two panels", (10.0, 10.0, 90.0, 20.0))
    left = _visual_block((10.0, 25.0, 45.0, 60.0))
    right = _visual_block((50.0, 25.0, 90.0, 60.0))
    blocks = [caption, left, right]

    regions = visual_annotations._classify_and_bind_visual_annotations(
        blocks,
        _PAGE_SIZE,
    )

    assert len(regions) == 1
    assert regions[0] == [caption, left, right]

    left_caption = _text_block("Figure 2: Left", (10.0, 10.0, 45.0, 20.0))
    independent_blocks = [left_caption, left.copy(), right.copy()]
    independent_regions = visual_annotations._classify_and_bind_visual_annotations(
        independent_blocks,
        _PAGE_SIZE,
    )

    assert len(independent_regions) == 1
    assert independent_regions[0] == [left_caption, independent_blocks[1]]


def test_nearest_parent_wins_and_intervening_body_blocks_cross_binding() -> None:
    """验证竞争父块优先选择归一化边距更小者，正文阻挡远距绑定。"""

    upper = _visual_block((20.0, 10.0, 100.0, 40.0))
    caption = _text_block("Figure 1: Upper", (20.0, 45.0, 100.0, 55.0))
    lower = _visual_block((20.0, 80.0, 100.0, 110.0))
    blocks = [upper, caption, lower]
    regions = visual_annotations._classify_and_bind_visual_annotations(
        blocks,
        _PAGE_SIZE,
    )

    assert regions == [[upper, caption]]

    body = _text_block("ordinary paragraph", (20.0, 55.0, 100.0, 65.0))
    note = _text_block("Source: survey", (20.0, 75.0, 100.0, 85.0))
    blocked = [upper.copy(), body, note]
    blocked_regions = visual_annotations._classify_and_bind_visual_annotations(
        blocked,
        _PAGE_SIZE,
    )

    assert note["type"] == "text"
    assert blocked_regions == []


def test_visual_regions_prevent_cross_column_row_remerge() -> None:
    """验证同一拆分视觉行的左右标题在区域化后不会先于两个主体一起输出。"""

    left_caption = _text_block(
        "Chart 1: Left",
        (10.0, 10.0, 80.0, 20.0),
        row_id=7,
    )
    right_caption = _text_block(
        "Chart 2: Right",
        (110.0, 10.0, 180.0, 20.0),
        row_id=7,
    )
    left_image = _visual_block((10.0, 25.0, 80.0, 70.0))
    right_image = _visual_block((110.0, 25.0, 180.0, 70.0))
    blocks = [left_caption, right_caption, left_image, right_image]
    regions = visual_annotations._classify_and_bind_visual_annotations(
        blocks,
        _PAGE_SIZE,
    )

    ordered = pipeline._sort_blocks_with_visual_row_groups(
        blocks,
        _PAGE_SIZE,
        visual_annotation_regions=regions,
    )

    assert ordered == [left_caption, left_image, right_caption, right_image]


def test_non_text_marginals_never_enter_visual_reclassification() -> None:
    """验证 footer 与 page_footnote 即使内容像强规则也不会改为视觉脚注。"""

    image = _visual_block((20.0, 20.0, 100.0, 60.0))
    footer = _text_block(
        "Source: disclaimer",
        (20.0, 70.0, 100.0, 80.0),
        block_type="footer",
    )
    page_footnote = _text_block(
        "Note: author affiliation",
        (20.0, 85.0, 100.0, 95.0),
        block_type="page_footnote",
    )

    regions = visual_annotations._classify_and_bind_visual_annotations(
        [image, footer, page_footnote],
        _PAGE_SIZE,
    )

    assert regions == []
    assert footer["type"] == "footer"
    assert page_footnote["type"] == "page_footnote"
