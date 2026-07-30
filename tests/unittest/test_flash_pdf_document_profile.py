from __future__ import annotations

from mineru.backend.flash.native_pdf import titles

from _flash_pdf_test_utils import _prepared_text_page, _text_line


def test_document_body_profile_prefers_cross_page_body_and_regular_fonts() -> None:
    """验证全文画像优先跨页正文行高，并排除反复出现的明显粗体字体。"""

    body_font = ("BodyRegular", 0)
    mono_font = ("MonoRegular", 0)
    italic_font = ("BodyItalic", 1 << 6)
    bold_font = ("HeadingBold", 1)
    pages = []
    for page_index in range(4):
        lines = [
            _text_line(
                "body",
                (0.0, 10.0, 80.0, 20.0),
                page_index * 10,
                effective_height=10.0,
                font_signature=body_font,
                font_coverage=1.0,
                dominant_font_weight=400.0,
            )
        ]
        if page_index < 3:
            lines.extend(
                [
                    _text_line(
                        "mono",
                        (0.0, 30.0, 80.0, 38.0),
                        page_index * 10 + 1,
                        effective_height=8.0,
                        font_signature=mono_font,
                        font_coverage=1.0,
                        dominant_font_weight=400.0,
                    ),
                    _text_line(
                        "bold",
                        (0.0, 50.0, 80.0, 60.0),
                        page_index * 10 + 2,
                        effective_height=10.0,
                        font_signature=bold_font,
                        font_coverage=1.0,
                        dominant_font_weight=800.0,
                    ),
                    _text_line(
                        "italic",
                        (0.0, 70.0, 80.0, 78.0),
                        page_index * 10 + 3,
                        effective_height=8.0,
                        font_signature=italic_font,
                        font_coverage=1.0,
                        dominant_font_weight=400.0,
                    ),
                ]
            )
        pages.append(_prepared_text_page(*lines))

    profile = titles._infer_document_body_profile(pages)

    assert profile is not None
    assert profile.body_height == 10.0
    assert profile.body_weight == 400.0
    # 正文字体画像只统计正文高度带，较矮的等宽和斜体样本不再进入 regular_fonts。
    assert profile.regular_fonts == frozenset({body_font})


def test_sparse_repeated_font_is_not_document_regular_style() -> None:
    """验证只在多页短标题中反复出现的字体不会被并入全文常规字体。"""

    body_font = ("BodyRegular", 0)
    sparse_font = ("SparseHeading", 1)
    pages = [
        _prepared_text_page(
            _text_line(
                "body",
                (0.0, 10.0, 100.0, 20.0),
                page_index * 2,
                font_signature=body_font,
                font_coverage=1.0,
                dominant_font_weight=0.0,
            ),
            _text_line(
                "short",
                (0.0, 40.0, 50.0, 50.0),
                page_index * 2 + 1,
                font_signature=sparse_font,
                font_coverage=1.0,
                dominant_font_weight=0.0,
            ),
        )
        for page_index in range(5)
    ]

    profile = titles._infer_document_body_profile(pages)

    assert profile is not None
    assert body_font in profile.regular_fonts
    assert sparse_font not in profile.regular_fonts


def test_document_regular_font_suppresses_code_page_false_title() -> None:
    """验证代码页中的全文常规字体说明保持正文，粗体步骤标签仍识别为标题。"""

    mono_font = ("MonoRegular", 0)
    regular_font = ("BodyRegular", 0)
    bold_font = ("HeadingBold", 1)
    lines = [
        _text_line(
            "code one",
            (0.0, 5.0, 100.0, 13.0),
            0,
            effective_height=8.0,
            font_signature=mono_font,
            font_coverage=1.0,
            dominant_font_weight=400.0,
        ),
        _text_line(
            "code two",
            (0.0, 15.0, 100.0, 23.0),
            1,
            effective_height=8.0,
            font_signature=mono_font,
            font_coverage=1.0,
            dominant_font_weight=400.0,
        ),
        _text_line(
            "ordinary instruction",
            (0.0, 40.0, 45.0, 50.0),
            2,
            effective_height=10.0,
            font_signature=regular_font,
            font_coverage=1.0,
            dominant_font_weight=400.0,
        ),
        _text_line(
            "code three",
            (0.0, 62.0, 100.0, 70.0),
            3,
            effective_height=8.0,
            font_signature=mono_font,
            font_coverage=1.0,
            dominant_font_weight=400.0,
        ),
        _text_line(
            "code four",
            (0.0, 72.0, 100.0, 80.0),
            4,
            effective_height=8.0,
            font_signature=mono_font,
            font_coverage=1.0,
            dominant_font_weight=400.0,
        ),
        _text_line(
            "bold step",
            (0.0, 100.0, 35.0, 110.0),
            5,
            effective_height=10.0,
            font_signature=bold_font,
            font_coverage=1.0,
            dominant_font_weight=800.0,
        ),
        _text_line(
            "code five",
            (0.0, 122.0, 100.0, 130.0),
            6,
            effective_height=8.0,
            font_signature=mono_font,
            font_coverage=1.0,
            dominant_font_weight=400.0,
        ),
    ]
    profile = titles._DocumentBodyProfile(
        body_height=10.0,
        body_weight=400.0,
        regular_fonts=frozenset({mono_font, regular_font}),
    )

    titles._classify_page_titles(
        lines,
        (100.0, 150.0),
        page_index=1,
        container_bboxes=[],
        document_body_profile=profile,
    )

    assert lines[2].semantic_type is None
    assert lines[5].semantic_type == "paragraph_title"


def test_document_title_profile_promotes_table_adjacent_style_but_not_container_label() -> None:
    """验证跨页标题原型可越过邻表抑制，但容器内部同样式标签仍保持正文。"""

    heading_font = ("Heading", 0)
    heading = _text_line(
        "heading",
        (0.0, 10.0, 40.0, 24.0),
        0,
        effective_height=14.0,
        font_signature=heading_font,
        font_coverage=1.0,
        dominant_font_weight=600.0,
    )
    container_label = _text_line(
        "container label",
        (0.0, 40.0, 40.0, 54.0),
        1,
        effective_height=14.0,
        font_signature=heading_font,
        font_coverage=1.0,
        dominant_font_weight=600.0,
    )
    body_profile = titles._DocumentBodyProfile(
        body_height=10.0,
        body_weight=400.0,
        regular_fonts=frozenset({("Body", 0)}),
    )
    title_profile = titles._DocumentTitleProfile(
        (
            titles._TitleStylePrototype(
                font_family="heading",
                font_flags=0,
                height_ratio=1.4,
                weight=600.0,
                alignment="left",
                anchor_offset=0.0,
                support_count=4,
                support_pages=3,
            ),
        )
    )

    titles._classify_page_titles(
        [heading, container_label],
        (100.0, 100.0),
        page_index=1,
        container_bboxes=[(0.0, 28.0, 100.0, 90.0)],
        document_body_profile=body_profile,
        document_title_profile=title_profile,
    )

    assert heading.semantic_type == "paragraph_title"
    assert container_label.semantic_type is None


def test_document_title_profile_promotes_body_height_centered_section() -> None:
    """验证与正文同字体字号的居中章节行可凭重复标题原型和邻表关系晋升。"""

    regular_font = ("Body", 0)
    heading = _text_line(
        "section",
        (30.0, 24.0, 70.0, 34.0),
        0,
        effective_height=10.0,
        font_signature=regular_font,
        font_coverage=1.0,
        dominant_font_weight=400.0,
    )
    body = _text_line(
        "body",
        (0.0, 40.0, 100.0, 50.0),
        1,
        effective_height=10.0,
        font_signature=regular_font,
        font_coverage=1.0,
        dominant_font_weight=400.0,
    )
    body_profile = titles._DocumentBodyProfile(
        body_height=10.0,
        body_weight=400.0,
        regular_fonts=frozenset({regular_font}),
    )
    title_profile = titles._DocumentTitleProfile(
        (
            titles._TitleStylePrototype(
                font_family="body",
                font_flags=0,
                height_ratio=1.0,
                weight=400.0,
                alignment="center",
                anchor_offset=0.0,
                support_count=5,
                support_pages=3,
            ),
        )
    )

    titles._classify_page_titles(
        [heading, body],
        (100.0, 100.0),
        page_index=1,
        container_bboxes=[(0.0, 0.0, 100.0, 20.0)],
        document_body_profile=body_profile,
        document_title_profile=title_profile,
    )

    assert heading.semantic_type == "paragraph_title"
    assert body.semantic_type is None


def test_document_title_profile_infers_repeated_large_left_aligned_style() -> None:
    """验证两页重复的大字号左对齐样式可形成文档级标题原型。"""

    body_font = ("Body", 0)
    heading_font = ("RepeatedHeading", 0)
    pages = [
        _prepared_text_page(
            _text_line(
                "body cover",
                (0.0, 20.0, 100.0, 30.0),
                0,
                effective_height=10.0,
                font_signature=body_font,
                font_coverage=1.0,
            ),
            page_size=(100.0, 100.0),
        )
    ]
    for page_index in range(1, 3):
        pages.append(
            _prepared_text_page(
                _text_line(
                    f"heading {page_index}",
                    (0.0, 10.0, 45.0, 24.0),
                    page_index * 10,
                    effective_height=14.0,
                    font_signature=heading_font,
                    font_coverage=1.0,
                    dominant_font_weight=600.0,
                ),
                _text_line(
                    f"body {page_index}",
                    (0.0, 40.0, 100.0, 50.0),
                    page_index * 10 + 1,
                    effective_height=10.0,
                    font_signature=body_font,
                    font_coverage=1.0,
                    dominant_font_weight=400.0,
                ),
                page_size=(100.0, 100.0),
            )
        )
    body_profile = titles._DocumentBodyProfile(
        body_height=10.0,
        body_weight=400.0,
        regular_fonts=frozenset({body_font}),
    )

    profile = titles._infer_document_title_profile(pages, body_profile)

    assert profile is not None
    assert len(profile.prototypes) == 1
    assert profile.prototypes[0].font_family == "repeatedheading"
    assert profile.prototypes[0].support_pages == 2


def test_regular_pitch_style_change_does_not_become_title() -> None:
    """验证仅有字体变化、但没有额外段间净空的正文续行不会误判标题。"""

    body_font = ("Body", 0)
    alternate_font = ("Alternate", 0)
    lines = [
        _text_line(
            "body one",
            (0.0, 0.0, 100.0, 10.0),
            0,
            font_signature=body_font,
            font_coverage=1.0,
        ),
        _text_line(
            "continuation",
            (0.0, 16.0, 80.0, 26.0),
            1,
            font_signature=alternate_font,
            font_coverage=1.0,
        ),
        _text_line(
            "body two",
            (0.0, 32.0, 100.0, 42.0),
            2,
            font_signature=body_font,
            font_coverage=1.0,
        ),
    ]
    profile = titles._DocumentBodyProfile(
        body_height=10.0,
        body_weight=400.0,
        regular_fonts=frozenset({body_font}),
    )

    titles._classify_page_titles(
        lines,
        (100.0, 100.0),
        page_index=1,
        container_bboxes=[],
        document_body_profile=profile,
    )

    assert lines[1].semantic_type is None


def test_title_font_at_body_size_does_not_override_document_size_profile() -> None:
    """验证标题字体落入正文字号带时不会仅凭大留白继续误判为标题。"""

    heading_font = ("Heading", 0)
    lines = [
        _text_line(
            "body one",
            (0.0, 0.0, 100.0, 10.0),
            0,
            font_signature=("Body", 0),
            font_coverage=1.0,
        ),
        _text_line(
            "weak heading",
            (0.0, 35.0, 45.0, 46.0),
            1,
            effective_height=11.0,
            font_signature=heading_font,
            font_coverage=1.0,
            dominant_font_weight=400.0,
        ),
        _text_line(
            "body two",
            (0.0, 60.0, 100.0, 70.0),
            2,
            font_signature=("Body", 0),
            font_coverage=1.0,
        ),
    ]
    body_profile = titles._DocumentBodyProfile(
        body_height=10.0,
        body_weight=400.0,
        regular_fonts=frozenset({("Body", 0)}),
    )
    title_profile = titles._DocumentTitleProfile(
        (
            titles._TitleStylePrototype(
                font_family="heading",
                font_flags=0,
                height_ratio=1.4,
                weight=400.0,
                alignment="left",
                anchor_offset=0.0,
                support_count=4,
                support_pages=3,
            ),
        )
    )

    titles._classify_page_titles(
        lines,
        (100.0, 100.0),
        page_index=1,
        container_bboxes=[],
        document_body_profile=body_profile,
        document_title_profile=title_profile,
    )

    assert lines[1].semantic_type is None


def test_smaller_recurrent_regular_font_does_not_gain_title_style_signal() -> None:
    """验证较小的跨页常规代码字体即使留白较大，也不会仅凭字体切换升为标题。"""

    body_font = ("BodyRegular", 0)
    mono_font = ("MonoRegular", 0)
    lines = [
        _text_line(
            "body one",
            (0.0, 5.0, 100.0, 15.0),
            0,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
            dominant_font_weight=400.0,
        ),
        _text_line(
            "body two",
            (0.0, 17.0, 100.0, 27.0),
            1,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
            dominant_font_weight=400.0,
        ),
        _text_line(
            "small regular command",
            (2.0, 45.0, 98.0, 53.0),
            2,
            effective_height=8.0,
            font_signature=mono_font,
            font_coverage=1.0,
            dominant_font_weight=400.0,
        ),
        _text_line(
            "body three",
            (0.0, 70.0, 100.0, 80.0),
            3,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
            dominant_font_weight=400.0,
        ),
        _text_line(
            "body four",
            (0.0, 82.0, 100.0, 92.0),
            4,
            effective_height=10.0,
            font_signature=body_font,
            font_coverage=1.0,
            dominant_font_weight=400.0,
        ),
    ]
    profile = titles._DocumentBodyProfile(
        body_height=10.0,
        body_weight=400.0,
        regular_fonts=frozenset({body_font, mono_font}),
    )

    titles._classify_page_titles(
        lines,
        (100.0, 120.0),
        page_index=1,
        container_bboxes=[],
        document_body_profile=profile,
    )

    assert lines[2].semantic_type is None


def test_document_profile_finds_cover_title_without_promoting_bottom_metadata() -> None:
    """验证全文正文基准可识别纯封面标题，底部版本元数据仍保持普通文本。"""

    title_font = ("TitleBold", 1)
    metadata_font = ("MetadataBold", 1)
    lines = [
        _text_line(
            "cover title one",
            (15.0, 55.0, 85.0, 80.0),
            0,
            effective_height=25.0,
            font_signature=title_font,
            font_coverage=1.0,
            dominant_font_weight=800.0,
        ),
        _text_line(
            "cover title two",
            (25.0, 85.0, 75.0, 110.0),
            1,
            effective_height=25.0,
            font_signature=title_font,
            font_coverage=1.0,
            dominant_font_weight=800.0,
        ),
        _text_line(
            "version metadata",
            (35.0, 168.0, 65.0, 178.0),
            2,
            effective_height=10.0,
            font_signature=metadata_font,
            font_coverage=1.0,
            dominant_font_weight=800.0,
        ),
        _text_line(
            "date metadata",
            (30.0, 180.0, 70.0, 190.0),
            3,
            effective_height=10.0,
            font_signature=metadata_font,
            font_coverage=1.0,
            dominant_font_weight=800.0,
        ),
    ]
    profile = titles._DocumentBodyProfile(10.0, 400.0, frozenset())

    titles._classify_page_titles(
        lines,
        (100.0, 200.0),
        page_index=0,
        container_bboxes=[],
        document_body_profile=profile,
    )

    assert [line.semantic_type for line in lines] == [
        "doc_title",
        "doc_title",
        None,
        None,
    ]
