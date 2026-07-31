from __future__ import annotations

from pathlib import Path
from typing import Any


from mineru.backend.flash import pdf_extractor
from mineru.backend.flash.native_pdf import (
    formulas,
    geometry,
    graphics,
    line_merging,
    models,
    native_text,
    pipeline,
    tables,
)
from mineru.utils.pdf_document import get_lines_from_chars


def _native_model_list(pdf_name: str) -> list[list[dict[str, Any]]]:
    """运行仓库内数字 PDF 样例并返回 Flash 原生模型输出。"""

    pdf_path = Path(__file__).parents[2] / "demo" / "pdfs" / pdf_name
    with pdf_extractor.PDFDocument(str(pdf_path)) as pdf_doc:
        return pipeline._analyze_native_document(pdf_doc)


def _txt_model_list(pdf_name: str) -> list[list[dict[str, Any]]]:
    """显式使用 Flash TXT 模式解析仓库内回归 PDF，禁止经过 auto 分类。"""

    pdf_path = Path(__file__).parents[2] / "demo" / "pdfs" / pdf_name
    return pdf_extractor.doc_analyze(
        pdf_path.read_bytes(),
        parse_mode="txt",
    )


def _auto_model_list(pdf_name: str) -> list[list[dict[str, Any]]]:
    """通过稳定公共入口以 auto 模式解析仓库内最小回归 PDF。"""

    pdf_path = Path(__file__).parents[2] / "demo" / "pdfs" / pdf_name
    return pdf_extractor.doc_analyze(
        pdf_path.read_bytes(),
        parse_mode="auto",
    )


def _native_table_counts(pdf_name: str) -> list[int]:
    """返回仓库内数字 PDF 样例的逐页表格块数量。"""

    return [sum(block["type"] == "table" for block in page) for page in _native_model_list(pdf_name)]


def _native_page_source(pdf_name: str, page_idx: int) -> models._PageSource:
    """读取指定样例页并构造候选检测与认领测试使用的页面源。"""

    pdf_path = Path(__file__).parents[2] / "demo" / "pdfs" / pdf_name
    with pdf_extractor.PDFDocument(str(pdf_path)) as pdf_doc:
        page_size = pdf_doc.page_size(page_idx)
        chars = pdf_doc.get_page_chars(page_idx)
        lines = native_text._build_native_line_items(
            get_lines_from_chars(chars),
            page_size,
            page_rotation=pdf_doc.page_rotation(page_idx),
        )
        return models._PageSource(
            page_size=page_size,
            lines=lines,
            chars=chars,
            drawing_lines=native_text._get_pdf_drawing_lines(pdf_doc, page_idx),
            image_bboxes=pdf_doc.get_page_image_bboxes(page_idx),
            signature_bboxes=pdf_doc.get_page_signature_bboxes(page_idx),
            form_bboxes=pdf_doc.get_page_form_bboxes(page_idx),
        )


def _normalized_content_probe(text: str) -> str:
    """去除空白、标点与 dash 差异，生成原生行内容覆盖检查使用的探针。"""

    return "".join(char.casefold() for char in text if char.isalnum())


def _blocks_containing(
    blocks: list[dict[str, Any]],
    probe: str,
) -> list[dict[str, Any]]:
    """返回归一化内容中包含指定探针的模型块。"""

    normalized_probe = _normalized_content_probe(probe)
    return [
        block
        for block in blocks
        if normalized_probe
        in _normalized_content_probe(str(block.get("content", "")))
    ]


def test_demo1_keeps_five_real_tables_without_formula_false_positive() -> None:
    """验证 demo1 首页脚注、参考文献、公式与五个真实表格均保持正确。"""

    model_list = _native_model_list("demo1.pdf")

    assert [len(page) for page in model_list] == [16, 9, 12, 18, 10, 9, 11, 8, 10, 7, 10, 26, 9]
    assert [sum(block["type"] == "table" for block in page) for page in model_list] == [
        0,
        0,
        0,
        0,
        1,
        2,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
    ]
    assert sum(block["type"] == "doc_title" for page in model_list for block in page) == 1
    assert sum(block["type"] == "header" for page in model_list for block in page) == 12
    assert sum(block["type"] == "page_number" for page in model_list for block in page) == 12
    assert sum(block["type"] == "equation" for page in model_list for block in page) == 7
    assert [block["bbox"] for block in model_list[5] if block["type"] == "table"] == [
        [0.087, 0.125, 0.922, 0.363],
        [0.087, 0.669, 0.922, 0.893],
    ]
    assert [block["bbox"] for block in model_list[8] if block["type"] == "table"] == [
        [0.078, 0.125, 0.913, 0.416]
    ]
    page1_footnotes = [block for block in model_list[0] if block["type"] == "page_footnote"]
    assert len(page1_footnotes) == 1
    assert page1_footnotes[0]["content"].startswith("* Corresponding author.")
    copyright_block = next(
        block for block in model_list[0] if block["content"].startswith("0022-1694/$")
    )
    assert copyright_block["type"] == "text"
    assert "doi:10.1016/j.jhydrol.2005.01.006" in copyright_block["content"]
    assert next(block for block in model_list[0] if block["content"] == "Abstract")["type"] == "paragraph_title"
    assert next(block for block in model_list[6] if block["content"].startswith("4.2."))["type"] == "paragraph_title"


def test_demo1_rotated_table_claims_all_206_lines_without_residual_text() -> None:
    """验证 demo1 第五页旋转表完整认领 206 行且表框内没有残留文本。"""

    source = _native_page_source("demo1.pdf", 4)
    candidates = tables._detect_table_candidates(source)
    blocks, claimed = tables._materialize_table_blocks(source, candidates)
    rotated_indices = {line.source_index for line in source.lines if line.angle == 270}

    assert len(blocks) == 1
    assert len(rotated_indices) == 206
    assert claimed == rotated_indices
    assert not [
        line
        for line in source.lines
        if line.angle == 270
        and line.source_index not in claimed
        and any(
            candidate.angle == 270
            and geometry._point_in_bbox(
                (geometry._bbox_center_x(line.bbox), geometry._bbox_center_y(line.bbox)),
                candidate.bbox,
            )
            for candidate in candidates
        )
    ]


def test_demo2_rejects_figure_grid_and_keeps_two_real_tables() -> None:
    """验证 demo2 曲线图被拒绝且第四、五页真实表格保留。"""

    assert _native_table_counts("demo2.pdf") == [0, 0, 0, 1, 1, 0]


def test_demo2_page1_forms_sixteen_blocks_and_keeps_figure_caption_separate() -> None:
    """验证 demo2 首页正文自然聚合，六个 Figure 1 标签单块且 caption 独立。"""

    page = _native_model_list("demo2.pdf")[0]
    graphic_block = next(block for block in page if "Left camera" in block["content"])
    caption_block = next(block for block in page if "Figure 1:" in block["content"])

    assert len(page) == 16
    assert not [block for block in page if block["type"] == "table"]
    assert next(block for block in page if block["content"].startswith("Real-time Temporal"))["type"] == "doc_title"
    assert next(block for block in page if block["content"] == "I. INTRODUCTION")["type"] == "paragraph_title"
    for expected_text in ("dp", "¯x", "p =", "¯p =", "Left camera", "Right camera"):
        assert expected_text in graphic_block["content"]
    assert graphic_block is not caption_block
    assert "Figure 1:" not in graphic_block["content"]
    assert "Left camera" not in caption_block["content"]
    assert sum(block["content"].count("Left camera") for block in page) == 1
    assert graphic_block["bbox"] == [0.516, 0.311, 0.913, 0.442]
    copyright_block = next(
        block
        for block in page
        if block["content"].startswith("978-1-4673-5208-6")
    )
    assert copyright_block["type"] == "footer"


def test_demo2_pages2_to6_restore_paragraphs_formulas_and_reading_order() -> None:
    """验证 demo2 后续页达到目标块数，正文、公式、caption 与双栏顺序均稳定。"""

    model_list = _native_model_list("demo2.pdf")

    assert [len(page) for page in model_list] == [16, 16, 21, 13, 16, 16]
    assert [sum(block["type"] == "image" for block in page) for page in model_list] == [1, 0, 0, 5, 2, 0]
    assert [sum(block["type"] == "table" for block in page) for page in model_list] == [0, 0, 0, 1, 1, 0]
    assert sum(block["type"] == "equation" for page in model_list for block in page) == 9

    page2 = model_list[1]
    page2_contents = [block["content"] for block in page2]
    humans = next(content for content in page2_contents if content.startswith("Humans group shapes"))
    matching = next(content for content in page2_contents if content.startswith("To identify a match"))
    dissimilarity = next(content for content in page2_contents if content.startswith("where the pixel dissimilarity"))
    formula3 = next(content for content in page2_contents if content.endswith("(3)"))
    assert not [content for content in page2_contents if content.strip() == "by"]
    assert humans.endswith("is given by")
    assert "Sp denotes a set of matching candidates" in matching
    assert "green, and blue components given by" in dissimilarity
    assert "green, and blue" not in formula3

    page3 = model_list[2]
    page3_contents = [block["content"] for block in page3]
    assert next(
        block for block in page3 if block["content"] == "B. Temporal cost aggregation"
    )["type"] == "paragraph_title"
    assert page3_contents[12] == "D. Iterative Disparity Refinement"
    assert all(block["bbox"][2] <= 0.5 for block in page3[:12])
    assert "O(ω2) to O(ω)" in page3_contents[0]
    assert "disparity estimates Dip" in page3_contents[13]
    for formula_number in range(4, 10):
        marker = f"({formula_number})"
        assert sum(marker in content for content in page3_contents) == 1
    formula7 = next(content for content in page3_contents if "(7)" in content)
    formula4 = next(content for content in page3_contents if "(4)" in content)
    formula8 = next(content for content in page3_contents if "(8)" in content)
    assert formula4.splitlines()[-1] == "(4)"
    assert formula8.splitlines()[-1] == "(8)"
    assert "(4)" not in "\n".join(formula4.splitlines()[:-1])
    assert "(8)" not in "\n".join(formula8.splitlines()[:-1])
    assert "Fp =" in formula7
    assert "otherwise" in formula7
    assert not [content for content in page3_contents if content.strip() in {"2", "p", "otherwise", "(7)"}]
    assert "available at http://mc2.unl.edu/current-research" in page3_contents[-1]
    assert "/image-processing/. Figure 2" in page3_contents[-1]

    page4_contents = [block["content"] for block in model_list[3]]
    figure2 = next(content for content in page4_contents if content.startswith("Figure 2:"))
    figure3 = next(content for content in page4_contents if content.startswith("Figure 3:"))
    results = next(content for content in page4_contents if content.startswith("The results of temporal stereo"))
    improvements = next(content for content in page4_contents if content.startswith("Significant improvements"))
    assert figure2.endswith("(4th row).")
    assert figure3.endswith("without temporal aggregation.")
    assert results.endswith("methods that operate on pairs of images.")
    assert improvements.endswith("has the effect")

    page5 = model_list[4]
    page5_contents = [block["content"] for block in page5]
    optimal_feedback = next(content for content in page5_contents if content.startswith("The optimal value"))
    page5_references = [content for content in page5_contents if content.startswith("[")]
    references_title = next(block for block in page5 if block["content"] == "REFERENCES")
    assert "noise ranging between ±0 to ±40" in optimal_feedback
    assert optimal_feedback.endswith("temporal stereo matching is used.")
    assert references_title["type"] == "paragraph_title"
    assert [content.partition("]")[0] + "]" for content in page5_references] == [
        f"[{number}]" for number in range(1, 6)
    ]

    page6_contents = [block["content"] for block in model_list[5]]
    assert len(page6_contents) == 16
    assert [content.partition("]")[0] + "]" for content in page6_contents] == [
        f"[{number}]" for number in range(6, 22)
    ]


def test_demo2_container_claims_are_pairwise_disjoint() -> None:
    """验证表格、图形和公式阶段按 source_index 唯一认领，不重复消费文本身份。"""

    for page_idx in (1, 2, 3):
        source = _native_page_source("demo2.pdf", page_idx)
        table_candidates = tables._detect_table_candidates(source)
        table_blocks, table_claimed = tables._materialize_table_blocks(source, table_candidates)
        table_bboxes = [block["bbox"] for block in table_blocks]
        _graphic_blocks, graphic_claimed = graphics._build_graphic_like_blocks(
            source,
            table_bboxes,
            table_claimed,
        )
        remaining = line_merging._merge_same_baseline_text_lines(
            [
                line
                for line in source.lines
                if line.source_index not in table_claimed | graphic_claimed
            ],
            source.page_size,
            table_bboxes,
        )
        formula_input_indices = {line.source_index for line in remaining}
        _formula_blocks, formula_remaining = formulas._build_formula_like_blocks(
            remaining,
            table_bboxes,
            source.page_size,
        )
        formula_claimed = formula_input_indices - {line.source_index for line in formula_remaining}

        assert table_claimed.isdisjoint(graphic_claimed)
        assert table_claimed.isdisjoint(formula_claimed)
        assert graphic_claimed.isdisjoint(formula_claimed)
        combined = table_claimed | graphic_claimed | formula_claimed
        assert len(combined) == len(table_claimed) + len(graphic_claimed) + len(formula_claimed)


def test_demo2_page4_groups_five_graphics_and_keeps_table1() -> None:
    """验证 demo2 第四页五个图形区域分别聚合，Table 1 继续优先输出为 table。"""

    page = _native_model_list("demo2.pdf")[3]
    table_blocks = [block for block in page if block["type"] == "table"]
    graphic_markers = ("Frame 30", "Noise: ±0", "Noise: ±20", "Noise: ±40", "Noise ±")
    graphic_blocks = [
        next(block for block in page if block["type"] == "image" and marker in block["content"])
        for marker in graphic_markers
    ]

    assert len(table_blocks) == 1
    assert "Table I:" in table_blocks[0]["content"]
    assert len({id(block) for block in graphic_blocks}) == 5
    assert "Frame 90" in graphic_blocks[0]["content"]
    assert all("Figure" not in block["content"] for block in graphic_blocks)


def test_demo2_table_captions_and_numeric_footnotes_are_not_text_blocks() -> None:
    """验证 demo2 两张表的换行标题和数字脚注全部并入表格投影。"""

    model_list = _native_model_list("demo2.pdf")
    page4_table = next(block for block in model_list[3] if block["type"] == "table")
    page5_table = next(block for block in model_list[4] if block["type"] == "table")
    residual_text = "\n".join(
        block["content"] for page_idx in (3, 4) for block in model_list[page_idx] if block["type"] == "text"
    )

    assert "ral stereo matching." in page4_table["content"]
    assert "1 To enable propagation of disparity information" in page4_table["content"]
    assert "0.01, respectively." in page4_table["content"]
    assert "Noise: ±20" not in page4_table["content"]
    assert "1 Millions of Disparity Estimates per Second." in page5_table["content"]
    assert "2 Assumes 320 × 240 images with 32 disparity levels." in page5_table["content"]
    assert "the avgerage % of bad pixels." in page5_table["content"]
    assert "ral stereo matching." not in residual_text
    assert "Millions of Disparity Estimates per Second." not in residual_text
    assert page4_table["bbox"] == [0.51, 0.484, 0.915, 0.675]
    assert page5_table["bbox"] == [0.51, 0.218, 0.915, 0.438]


def test_demo3_keeps_tables_and_covers_every_native_source_line() -> None:
    """验证 demo3 容器、后续页段落边界及每条原生 source line 均保持稳定。"""

    pdf_path = Path(__file__).parents[2] / "demo" / "pdfs" / "demo3.pdf"
    with pdf_extractor.PDFDocument(str(pdf_path)) as pdf_doc:
        model_list = pipeline._analyze_native_document(pdf_doc)
        source_lines_by_page: list[list[models._LineItem]] = []
        for page_idx in range(pdf_doc.page_count):
            page_size = pdf_doc.page_size(page_idx)
            source_lines_by_page.append(
                native_text._build_native_line_items(
                    get_lines_from_chars(pdf_doc.get_page_chars(page_idx)),
                    page_size,
                    page_rotation=pdf_doc.page_rotation(page_idx),
                )
            )

    assert [sum(block["type"] == "table" for block in page) for page in model_list] == [
        2,
        0,
        0,
        0,
        1,
        2,
        2,
        2,
        0,
        0,
    ]
    assert [len(page) for page in model_list] == [
        23,
        15,
        13,
        21,
        19,
        16,
        16,
        15,
        17,
        18,
    ]
    assert sum(len(page) for page in model_list) == 173
    page7_tables = [block for block in model_list[6] if block["type"] == "table"]
    page7_table4 = next(
        block for block in page7_tables if "Number of parameters" in block["content"]
    )
    page7_table4_caption = next(
        block
        for block in model_list[6]
        if block["content"] == "Table 4: Model size comparison."
    )
    page7_inline_body = next(
        block
        for block in model_list[6]
        if block["content"].startswith("Row, Column, & Global Positional IDs.")
    )
    page9_conclusion = next(
        block
        for block in model_list[8]
        if block["content"].startswith("In this paper, we identified")
    )
    page10_first_reference = next(
        block
        for block in model_list[9]
        if block["content"].startswith("Xiang Deng, Huan Sun")
    )
    assert all(
        marker in page7_table4["content"]
        for marker in (
            "Model",
            "TAPASBASE",
            "TABLEFORMERBASE",
            "TAPASLARGE",
            "TABLEFORMERLARGE",
        )
    )
    assert page7_table4_caption["content"] not in page7_table4["content"]
    assert page7_table4["bbox"][3] < page7_table4_caption["bbox"][1]
    assert page7_inline_body["type"] == "text"
    assert "With TAPASBASE" in page7_inline_body["content"]
    assert "To tackle this" in page9_conclusion["content"]
    assert "Acknowledgments" not in page9_conclusion["content"]
    assert "Cong Yu. 2021. TURL:" in page10_first_reference["content"]
    assert "Jacob Devlin" not in page10_first_reference["content"]
    for page, source_lines in zip(model_list, source_lines_by_page, strict=True):
        output_probe = _normalized_content_probe("".join(str(block.get("content") or "") for block in page))
        missing_lines = [
            line.text
            for line in source_lines
            if (line_probe := _normalized_content_probe(line.text)) and line_probe not in output_probe
        ]
        assert not missing_lines


def test_demo3_auxiliary_text_types_match_real_page_geometry() -> None:
    """验证真实 PDF 的首页侧栏及第 1、5、6、9 页脚注命中，公式页不误报。"""

    model_list = _native_model_list("demo3.pdf")

    assert [sum(block["type"] == "aside_text" for block in page) for page in model_list] == [
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert [sum(block["type"] == "page_footnote" for block in page) for page in model_list] == [
        2,
        0,
        0,
        0,
        2,
        1,
        0,
        0,
        1,
        0,
    ]
    assert not [
        block
        for block in model_list[6]
        if block["type"] in {"aside_text", "page_footnote"}
    ]


def test_demo3_pages1_and2_fix_title_front_matter_and_embedding_formula() -> None:
    """验证首页标题稳定，第二页标题、公式和栏尾正文各自保持完整。"""

    page1, page2 = _native_model_list("demo3.pdf")[:2]
    title = next(block for block in page1 if block["content"].startswith("TABLEFORMER:"))
    front_matter_contents = {
        "Aditya Gupta† Rahul Goel†",
        "Jingfeng Yang∗ Luheng He†",
        "Shyam Upadhyay† Shachi Paul †",
        "?Georgia Institute of Technology",
        "†Google Assistant",
        "jingfengyangpku@gmail.com",
        "tableformer@google.com",
    }
    front_matter = [
        block for block in page1 if block["content"] in front_matter_contents
    ]
    released_code = next(
        block for block in page1 if "TABLEFORMER.md" in block["content"]
    )
    aside = next(block for block in page1 if block["type"] == "aside_text")
    introduction = next(
        block for block in page1 if block["content"].startswith("Recently, semi-structured")
    )
    nutshell = next(
        block for block in page1 if block["content"].startswith("In a nutshell")
    )
    figure_caption = next(
        block for block in page1 if block["content"].startswith("Figure 1:")
    )
    tables_body = next(
        block for block in page1 if block["content"].startswith("tables or rows")
    )

    assert title["type"] == "doc_title"
    assert len(front_matter) == len(front_matter_contents)
    assert all(block["type"] == "text" for block in front_matter)
    assert not [block for block in page1 if block["content"] == "∗"]
    assert next(block for block in page1 if block["content"] == "Abstract")["type"] == "paragraph_title"
    assert released_code["type"] == "page_footnote"
    assert aside["angle"] == 270
    assert aside["bbox"][2] <= 0.12
    assert introduction["content"].endswith("(Eisenschlos et al., 2021; Liu et al., 2021).")
    assert nutshell["type"] == "text"
    assert nutshell["content"].endswith("by serializing")
    assert figure_caption["type"] == "text"
    assert figure_caption["content"].endswith("both questions.")
    assert "tables or rows" not in figure_caption["content"]
    assert tables_body["type"] == "text"

    section_title = next(
        block for block in page2 if block["content"].startswith("2 Preliminaries:")
    )
    equations = [block for block in page2 if block["type"] == "equation"]
    assert section_title["type"] == "paragraph_title"
    assert section_title["content"] == "2 Preliminaries: TAPAS for Table Encoding"
    assert len(equations) == 1
    assert equations[0]["content"].splitlines() == [
        "token ids (W) = {wv1, wv2, · · · , wvn }",
        "positional ids (B) = {b1, b2, · · · , bn}",
        "segment ids (G) = {gseg1, gseg2, · · · , gsegn }",
        "column ids (C) = {ccol1, ccol2, · · · , ccoln}",
        "row ids (R) = {rrow1, rrow2, · · · , rrown }",
        "rank ids (Z) = {zrank1, zrank2, · · · , zrankn}",
    ]
    as_model_blocks = [
        block
        for block in page2
        if "As for the model" in block["content"]
        or "attends to all the tokens." in block["content"]
        or "Let the layer input" in block["content"]
    ]
    assert len(as_model_blocks) == 1
    assert as_model_blocks[0]["type"] == "text"
    assert as_model_blocks[0]["content"].startswith("As for the model")
    assert "attends to all the tokens." in as_model_blocks[0]["content"]
    assert "Let the layer input" in as_model_blocks[0]["content"]


def test_demo3_pages6_7_and10_fix_caption_inline_titles_and_reference_tail() -> None:
    """验证跨栏 caption、行内粗体正文与参考文献尾行均保持正确归属。"""

    model_list = _native_model_list("demo3.pdf")
    page6 = model_list[5]
    page7 = model_list[6]
    page10 = model_list[9]

    table2_caption = next(
        block for block in page6 if block["content"].startswith("Table 2:")
    )
    assert table2_caption["type"] == "text"
    assert "Median of 5 independent runs are reported." in table2_caption["content"]
    assert table2_caption["content"].endswith("not reported in the original paper.")
    assert sum(
        "not reported in the original paper." in block["content"]
        for block in page6
    ) == 1

    attention_bias = next(
        block for block in page7 if block["content"].startswith("Attention Bias Scaling.")
    )
    positional_ids = next(
        block
        for block in page7
        if block["content"].startswith("Row, Column, & Global Positional IDs.")
    )
    formula6 = next(
        block
        for block in page7
        if block["type"] == "equation" and "(6)" in block["content"]
    )
    assert attention_bias["type"] == "text"
    assert attention_bias["content"].endswith("attention score by:")
    assert positional_ids["type"] == "text"
    assert "With TAPASBASE" in positional_ids["content"]
    assert formula6["content"].splitlines()[-1] == "(6)"
    assert not [
        block
        for block in page7
        if block["type"] == "paragraph_title"
        and block["content"].startswith(
            ("Attention Bias Scaling.", "Row, Column, & Global Positional IDs.")
        )
    ]

    ying_reference = next(
        block for block in page10 if block["content"].startswith("Chengxuan Ying")
    )
    yu_reference = next(
        block for block in page10 if block["content"].startswith("Tao Yu")
    )
    assert ying_reference["type"] == yu_reference["type"] == "text"
    assert ying_reference["content"].endswith("arXiv:2106.05234.")
    assert "Tao Yu" not in ying_reference["content"]
    assert not [
        block for block in page10 if block["content"] == "arXiv:2106.05234."
    ]


def test_demo3_page3_form_image_formulas_titles_and_inline_body_are_whole() -> None:
    """验证第三页大 Form、caption、公式、标题及行内粗体都按整体输出。"""

    page = _native_model_list("demo3.pdf")[2]
    image_blocks = [block for block in page if block["type"] == "image"]
    assert len(image_blocks) == 1
    assert not [block for block in page if block["type"] == "table"]
    image_block = image_blocks[0]
    assert "Transformer (Self Attention)" in image_block["content"]
    assert "Screwed Up" in image_block["content"]
    assert "Figure 2:" not in image_block["content"]
    assert all(
        block is image_block
        or not geometry._point_in_bbox(
            (
                (block["bbox"][0] + block["bbox"][2]) / 2.0,
                (block["bbox"][1] + block["bbox"][3]) / 2.0,
            ),
            tuple(image_block["bbox"]),
        )
        for block in page
    )
    caption_blocks = [
        block
        for block in page
        if "Figure 2:" in block["content"]
        or "types of task independent biases" in block["content"]
    ]
    assert len(caption_blocks) == 1
    assert caption_blocks[0]["type"] == "text"
    assert "This example corresponds to table (a)" in caption_blocks[0]["content"]
    assert "associated text." in caption_blocks[0]["content"]

    formula1 = next(block for block in page if "(1)" in block["content"])
    section3 = next(block for block in page if block["content"].startswith("3 TABLEFORMER:"))
    inline_item = next(block for block in page if block["content"].startswith("2) Per cell positional ids."))
    inline_heading = next(
        block for block in page if block["content"].startswith("Positional Encoding in TABLEFORMER.")
    )
    assert formula1["type"] == "equation"
    assert "Q = HWQ" in formula1["content"] and "K = HWK" in formula1["content"]
    assert section3["type"] == "paragraph_title"
    assert section3["content"] == "3 TABLEFORMER: Robust Structural Table Encoding"
    assert inline_item["type"] == "text" and "To further remove any" in inline_item["content"]
    assert inline_heading["type"] == "text" and "Transformer model" in inline_heading["content"]


def test_demo3_pages4_and5_fix_lists_formula_titles_italics_and_footnotes() -> None:
    """验证第四、五页列表、公式、独立标题、行内标题、斜体续行及脚注边界。"""

    page4, page5 = _native_model_list("demo3.pdf")[3:5]
    left_bullets = [
        block
        for block in page4
        if block["type"] == "text"
        and block["content"].startswith("•")
        and block["bbox"][2] <= 0.5
    ]
    assert len(left_bullets) == 6
    assert all(len(block["content"].split()) > 8 for block in left_bullets)
    attention_biases = next(
        block for block in page4 if block["content"].startswith("Attention Biases in TABLEFORMER.")
    )
    assert attention_biases["type"] == "text"
    assert "13 bias types" in attention_biases["content"]

    formula3 = next(block for block in page4 if "(3)" in block["content"])
    formula4 = next(block for block in page4 if "(4)" in block["content"])
    assert formula3["type"] == formula4["type"] == "equation"
    assert formula3 is not formula4
    assert "A =" in formula3["content"] and "(4)" not in formula3["content"]
    assert "(3)" not in formula4["content"]

    relation_blocks = [
        block
        for block in page4
        if "Relation between TABLEFORMER and ETC." in block["content"]
        or "ETC (Ainslie et al., 2020)" in block["content"]
    ]
    assert len(relation_blocks) == 1
    assert relation_blocks[0]["type"] == "text"
    assert relation_blocks[0]["content"].startswith("Relation between TABLEFORMER and ETC.")
    assert "uses vectors to represent relative position labels" in relation_blocks[0]["content"]

    title4 = next(block for block in page4 if block["content"] == "4 Experimental Setup")
    title41 = next(block for block in page4 if block["content"] == "4.1 Datasets and Evaluation")
    assert title4["type"] == title41["type"] == "paragraph_title"
    for prefix, continuation in (
        ("Table Question Answering.", "conducted experiments"),
        ("Table-Text Entailment.", "TABFACT dataset"),
    ):
        inline_block = next(block for block in page4 if block["content"].startswith(prefix))
        assert inline_block["type"] == "text"
        assert continuation in inline_block["content"]

    assert next(block for block in page5 if block["content"] == "4.2 Baselines")["type"] == "paragraph_title"
    assert next(
        block for block in page5 if block["content"] == "4.3 Perturbing Tables as Augmented Data"
    )["type"] == "paragraph_title"
    italic_body = next(
        block for block in page5 if block["content"].startswith("Could we alleviate")
    )
    assert italic_body["type"] == "text"
    assert italic_body["content"].endswith("without making any")

    final_bullet = next(
        block for block in page5 if block["content"].startswith("• How does TABLEFORMER compare")
    )
    final_footnote = next(
        block for block in page5 if block["content"].startswith("3By perturbation")
    )
    assert final_bullet["type"] == "text"
    assert final_footnote["type"] == "page_footnote"
    assert "3By perturbation" not in final_bullet["content"]


def test_demo4_nct00083083_targeted_flash_regressions() -> None:
    """验证 demo4 的页眉脚注、续行、公式否决和参考文献分组。"""

    model_list = _native_model_list("demo4.pdf")

    page1 = model_list[0]
    assert any(
        block["type"] == "header"
        and "07388-8" in str(block["content"])
        for block in page1
    )
    assert next(
        block for block in page1 if block["content"] == "ORIGINAL ARTICLE"
    )["type"] == "text"
    assert _blocks_containing(page1, "contributed equally")[0]["type"] == "page_footnote"
    assert _blocks_containing(page1, "University of Edinburgh")[0]["type"] == "page_footnote"

    abstract_sections = [
        _blocks_containing(page1, opener)
        for opener in (
            "Purpose Prognostication",
            "Methods This study",
            "Results Conventional",
            "Conclusions Innate",
        )
    ]
    assert all(len(matches) == 1 for matches in abstract_sections)
    assert len({id(matches[0]) for matches in abstract_sections}) == 4
    for matches, tail in zip(
        abstract_sections,
        ("glucose metabolism", "chemoradiation", "0.77", "NSCLC patients"),
        strict=True,
    ):
        assert _normalized_content_probe(tail) in _normalized_content_probe(
            str(matches[0]["content"])
        )

    email = _blocks_containing(page1, "adriana.tavares@ed.ac.uk")[0]
    affiliation_blocks = [
        block
        for block in page1
        if block["type"] == "page_footnote"
        and block["bbox"][1] > email["bbox"][1]
        and str(block.get("content", "")).lstrip()[:1] in set("123456")
    ]
    assert len(affiliation_blocks) == 6
    assert {
        str(block["content"]).lstrip()[:1]
        for block in affiliation_blocks
    } == set("123456")
    assert not [
        block
        for block in page1
        if block["type"] == "page_footnote"
        and str(block.get("content", "")).strip() in set("123456")
    ]

    introduction = _blocks_containing(page1, "Cancer is one of the leading causes")
    assert len(introduction) == 1
    assert "throughoutthebody14" in _normalized_content_probe(
        str(introduction[0]["content"])
    )

    page4 = model_list[3]
    caption_tail = _blocks_containing(page4, "included in the ACRIN 6668 study")
    assert len(caption_tail) == 1
    assert _normalized_content_probe(str(caption_tail[0]["content"])).startswith(
        "inreducingglucoseuptake"
    )
    figure_caption = _blocks_containing(page4, "Fig. 1 Conventional")
    figure_body = _blocks_containing(page4, "in reducing glucose uptake")
    assert len(figure_caption) == 1
    assert len(figure_body) == 1
    assert figure_caption[0] is not figure_body[0]
    assert "inreducingglucoseuptake" not in _normalized_content_probe(
        str(figure_caption[0]["content"])
    )

    page5 = model_list[4]
    right_continuation = _blocks_containing(
        page5,
        "also observed when conducting Cox regression",
    )
    assert len(right_continuation) == 1
    assert "importantlythereisahomogenisation" in _normalized_content_probe(
        str(right_continuation[0]["content"])
    )

    page6 = model_list[5]
    left_continuation = _blocks_containing(page6, "This study aimed")
    assert len(left_continuation) == 1
    assert "primarytumoursite" in _normalized_content_probe(
        str(left_continuation[0]["content"])
    )

    page9 = model_list[8]
    assert not [block for block in page9 if block["type"] == "equation"]
    supplementary = _blocks_containing(page9, "Supplementary Information")
    assert len(supplementary) == 1
    assert "073888" in _normalized_content_probe(str(supplementary[0]["content"]))

    page11 = model_list[10]
    reference_blocks = [
        block
        for block in page11
        if block["type"] == "text"
        and any(
            str(block["content"]).lstrip().startswith(f"{number}.")
            for number in range(45, 54)
        )
    ]
    assert len(reference_blocks) == 9
    assert {
        int(str(block["content"]).lstrip().partition(".")[0])
        for block in reference_blocks
    } == set(range(45, 54))
    reference48 = next(
        block
        for block in reference_blocks
        if str(block["content"]).lstrip().startswith("48.")
    )
    assert "103389fimmu20231222129" in _normalized_content_probe(
        str(reference48["content"])
    )


def test_demo5_targeted_table_and_footer_regressions() -> None:
    """验证 demo5 代表页表格完整、框内无残文且页码外图片成为页脚。"""

    model_list = _native_model_list("demo5.pdf")

    sparse_table_counts = {
        4: 2,
        49: 1,
        55: 1,
        56: 2,
        59: 1,
        61: 1,
        83: 2,
        84: 3,
        85: 3,
        87: 1,
    }
    for page_number, expected_count in sparse_table_counts.items():
        page = model_list[page_number - 1]
        table_blocks = [block for block in page if block["type"] == "table"]
        assert len(table_blocks) == expected_count
        for table_block in table_blocks:
            table_bbox = table_block["bbox"]
            assert not [
                block
                for block in page
                if block["type"] == "text"
                and table_bbox[0]
                <= (block["bbox"][0] + block["bbox"][2]) / 2
                <= table_bbox[2]
                and table_bbox[1]
                <= (block["bbox"][1] + block["bbox"][3]) / 2
                <= table_bbox[3]
            ]

    for page_number in (9, 10, 22, 38, 70, 77):
        page = model_list[page_number - 1]
        table_blocks = [block for block in page if block["type"] == "table"]
        assert len(table_blocks) == 1
        table_bbox = table_blocks[0]["bbox"]
        residual_text = [
            block
            for block in page
            if block["type"] == "text"
            and table_bbox[0]
            <= (block["bbox"][0] + block["bbox"][2]) / 2
            <= table_bbox[2]
            and table_bbox[1]
            <= (block["bbox"][1] + block["bbox"][3]) / 2
            <= table_bbox[3]
        ]
        assert not residual_text

    repaired_table_bboxes = {
        23: [0.034, 0.151, 0.968, 0.696],
        24: [0.034, 0.151, 0.968, 0.696],
        36: [0.033, 0.151, 0.967, 0.525],
        71: [0.057, 0.151, 0.943, 0.685],
        75: [0.057, 0.151, 0.943, 0.684],
        81: [0.143, 0.086, 0.857, 0.785],
    }
    for page_number, expected_bbox in repaired_table_bboxes.items():
        page = model_list[page_number - 1]
        table_blocks = [block for block in page if block["type"] == "table"]
        assert [block["bbox"] for block in table_blocks] == [expected_bbox]
        assert not [block for block in page if block["type"] == "code"]
        assert not [
            block
            for block in page
            if block["type"] in {"text", "header"}
            and expected_bbox[0]
            <= (block["bbox"][0] + block["bbox"][2]) / 2
            <= expected_bbox[2]
            and expected_bbox[1]
            <= (block["bbox"][1] + block["bbox"][3]) / 2
            <= expected_bbox[3]
        ]

    page77_note = _blocks_containing(model_list[76], "注：（1）指气量")
    assert len(page77_note) == 1
    assert page77_note[0]["type"] == "text"

    page9 = model_list[8]
    page_number_block = next(
        block for block in page9 if block["type"] == "page_number"
    )
    empty_footer = next(
        block
        for block in page9
        if block["type"] == "footer" and not str(block["content"]).strip()
    )
    assert empty_footer["bbox"][1] > page_number_block["bbox"][3]


def test_demo6_default3_targeted_title_regressions() -> None:
    """验证 demo6 的五个正文负样本和九个真实章节标题。"""

    model_list = _native_model_list("demo6.pdf")
    blocks = [block for page in model_list for block in page]
    expected_titles = {
        "一、招标条件",
        "二、项目概况和招标范围",
        "三、投标人资格要求",
        "四、招标文件的获取",
        "五、投标文件的递交",
        "六、开标时间及地点",
        "七、其他",
        "八、监督部门",
        "九、联系方式",
    }
    assert {
        block["content"]
        for block in blocks
        if block["type"] == "paragraph_title"
    } == expected_titles

    for probe in (
        "招标编号",
        "(001)河南农业大学",
        "递交方式",
        "[2014]68号）。",
        "招 标 人",
    ):
        matches = _blocks_containing(blocks, probe)
        assert matches
        assert {block["type"] for block in matches} == {"text"}

    page7_images = [block for block in model_list[6] if block["type"] == "image"]
    assert [block["bbox"] for block in page7_images] == [
        [0.601, 0.106, 0.811, 0.255],
        [0.615, 0.244, 0.839, 0.403],
    ]
    assert [block["content"] for block in page7_images] == ["（签名）", "（盖章）"]
    assert sum("签名" in str(block["content"]) for block in model_list[6]) == 1
    assert sum("盖章" in str(block["content"]) for block in model_list[6]) == 1


def test_mixed_elements_pages_03_06_force_txt_regressions() -> None:
    """验证原文 3–6 页的标题、公式、图内坐标和边缘文本修复。"""

    model_list = _txt_model_list("mixed_elements_pages_03_06.pdf")

    assert len(model_list) == 4
    for probe in (
        "INTRODUCTION",
        "EXPERIMENTAL",
        "RESULTS AND DISCUSSION",
        "CONCLUSIONS",
        "REFERENCES",
    ):
        matches = _blocks_containing(
            [block for page in model_list for block in page],
            probe,
        )
        assert len(matches) == 1
        assert matches[0]["type"] == "paragraph_title"

    page4 = model_list[1]
    assert sum(block["type"] == "equation" for block in page4) == 1
    assert not [
        block
        for block in page4
        if block["type"] == "text"
        and 0.65 <= block["bbox"][1] <= 0.87
        and block["bbox"][2] - block["bbox"][0] < 0.05
    ]
    math_paragraph = _blocks_containing(
        page4,
        "This allowed us to estimate the low-temperature hop distance",
    )
    assert len(math_paragraph) == 1
    assert math_paragraph[0]["bbox"] == [0.081, 0.665, 0.475, 0.911]
    assert set(math_paragraph[0]) == {"type", "bbox", "angle", "content"}
    for probe in (
        "where T0",
        "Fermi-level density",
        "ized states N(EF)",
        "frequency factor",
        "Boltzmann constant",
    ):
        assert _blocks_containing(page4, probe) == math_paragraph
    assert [
        block
        for block in page4
        if block["type"] == "text"
        and block != math_paragraph[0]
        and block["bbox"][0] < 0.49
        and block["bbox"][1] < 0.92
        and block["bbox"][3] > 0.66
    ] == []
    page4_footer = [block for block in page4 if block["type"] == "footer"]
    assert len(page4_footer) == 1
    assert page4_footer[0]["bbox"] == [0.104, 0.93, 0.449, 0.941]
    assert math_paragraph[0]["bbox"][3] < page4_footer[0]["bbox"][1]
    assert [
        block["bbox"]
        for block in page4
        if block["type"] in {"text", "image"}
        and block["bbox"][0] >= 0.49
        and block["bbox"][1] >= 0.49
    ] == [
        [0.498, 0.509, 0.892, 0.634],
        [0.503, 0.66, 0.886, 0.871],
        [0.521, 0.889, 0.869, 0.911],
    ]

    page5 = model_list[2]
    axis_label = _blocks_containing(page5, "T–1/4, K–1/4")
    assert len(axis_label) == 1
    assert axis_label[0]["type"] == "image"
    assert _blocks_containing(page5, "MIKOLAICHUK et al.")[0]["type"] == "header"
    assert len([block for block in page5 if block["type"] == "footer"]) == 1
    lower_graphs = [
        block
        for block in page5
        if block["type"] == "image" and block["bbox"][1] >= 0.6
    ]
    assert len(lower_graphs) == 1
    assert lower_graphs[0]["bbox"] == [0.096, 0.634, 0.46, 0.862]
    assert "σT1/2, S K1/2/m" in str(lower_graphs[0]["content"])
    assert "T–1/4, K–1/4" in str(lower_graphs[0]["content"])
    assert "ular films." not in str(lower_graphs[0]["content"])
    figure4_caption = _blocks_containing(page5, "Fig. 4. Plots of σT1/2")
    assert len(figure4_caption) == 1
    assert figure4_caption[0]["type"] == "text"


def test_caibao_table_reclaims_repeated_dates_but_keeps_real_marginals() -> None:
    """验证财报两页表格回收底部日期行，免责声明与页码仍独立保留。"""

    model_list = _auto_model_list("caibao1_pages_10_11.pdf")

    assert len(model_list) == 2
    for page in model_list:
        tables_on_page = [block for block in page if block["type"] == "table"]
        footers = [block for block in page if block["type"] == "footer"]
        page_numbers = [block for block in page if block["type"] == "page_number"]
        assert len(tables_on_page) == len(footers) == len(page_numbers) == 1
        assert "2024 年 07 月" in str(tables_on_page[0]["content"])
        assert footers[0]["content"] == "免责声明和披露以及分析师声明是报告的一部分，请务必一起阅读。"
        assert "2024 年" not in str(footers[0]["content"])
        assert tables_on_page[0]["bbox"][3] < footers[0]["bbox"][1]
        assert tables_on_page[0]["bbox"][3] < page_numbers[0]["bbox"][1]


def test_iebm_left_indented_compact_formula_is_equation() -> None:
    """验证右栏左缩进紧凑公式通过公共 auto 入口输出为 equation。"""

    page = _auto_model_list("IEBM_A_2667169_O-5_page_1.pdf")[0]
    target = [
        block
        for block in page
        if block["bbox"] == [0.532, 0.292, 0.658, 0.312]
    ]

    assert len(target) == 1
    assert target[0]["type"] == "equation"
    assert target[0]["content"] == "F ¼ @ψ @Y, G ¼ @ψ @X."


def test_frozen_soil_reference_tails_remain_single_text_blocks() -> None:
    """验证三条悬挂缩进参考文献尾行不再误报 equation，且各自仅出现一次。"""

    page = _auto_model_list("frozen_soil_ensemble_page_9.pdf")[0]

    assert not [block for block in page if block["type"] == "equation"]
    for probe in (
        "分析［J］. 深空探测学报，2023",
        "形变监测［J］. 测绘通报，2025",
        "岩爆预测［J］. 高压物理学报，2025",
    ):
        matches = _blocks_containing(page, probe)
        assert len(matches) == 1
        assert matches[0]["type"] == "text"
        assert str(matches[0]["content"]).count(probe) == 1


def test_mixed_elements_pages_07_10_force_txt_regressions() -> None:
    """验证原文 7–10 页作者列、代码边界、尾词标题和参考文献修复。"""

    model_list = _txt_model_list("mixed_elements_pages_07_10.pdf")
    page7, page8, page9, page10 = model_list

    author_blocks = [
        block
        for block in page7
        if block["type"] == "text"
        and 0.15 <= block["bbox"][1]
        and block["bbox"][3] <= 0.28
    ]
    assert len(author_blocks) == 4
    assert {
        name
        for name in ("Xing Wang", "Yingzhou Zhang", "Lian Zhao", "Xinghao Chen")
        if any(name in str(block["content"]) for block in author_blocks)
    } == {"Xing Wang", "Yingzhou Zhang", "Lian Zhao", "Xinghao Chen"}
    for probe in ("general applicability.", "called static slicing."):
        matches = _blocks_containing(page7, probe)
        assert len(matches) == 1
        assert matches[0]["type"] == "text"

    code_blocks = [block for block in page8 if block["type"] == "code"]
    assert len(code_blocks) == 3
    assert not [block for block in page8 if block["type"] == "table"]
    figure_caption = _blocks_containing(page8, "Figure 1. Program containing dead code")
    assert len(figure_caption) == 1
    assert figure_caption[0]["type"] == "text"

    framework_title = _blocks_containing(
        page9,
        "IV. DEAD CODE DETECTION FRAMEWORK BASED ON LLVM INFRASTRUCTURE",
    )
    assert len(framework_title) == 1
    assert framework_title[0]["type"] == "paragraph_title"
    experiment_title = _blocks_containing(page9, "V. EXPERIMENT RESULTS")
    assert len(experiment_title) == 1
    assert experiment_title[0]["type"] == "paragraph_title"
    url_footer = _blocks_containing(page9, "http://klee.github.io")
    assert len(url_footer) == 1
    assert url_footer[0]["type"] == "footer"

    reference5 = _blocks_containing(page10, "[5] A. Srivastava")
    assert len(reference5) == 1
    assert reference5[0]["type"] == "text"
    assert "[6]" not in str(reference5[0]["content"])


def test_mixed_elements_pages_11_15_force_txt_regressions() -> None:
    """验证原文 11–15 页双行标题、正文小节、页码、图注与网址页脚。"""

    model_list = _txt_model_list("mixed_elements_pages_11_15.pdf")
    all_blocks = [block for page in model_list for block in page]
    document_titles = [block for block in all_blocks if block["type"] == "doc_title"]

    assert len(document_titles) == 1
    assert "Ecabet sodium prevents esophageal lesions" in document_titles[0]["content"]
    assert "reflux of gastric juice in rats" in document_titles[0]["content"]
    for probe in (
        "2. Effects on esophageal lesions",
        "3. Effect on digestion of mucus",
    ):
        matches = _blocks_containing(all_blocks, probe)
        assert len(matches) == 1
        assert matches[0]["type"] == "paragraph_title"

    assert {
        block["content"]
        for block in all_blocks
        if block["type"] == "page_number"
    } == {"91", "92", "93", "94"}
    assert not [
        block
        for block in all_blocks
        if block["type"] == "image"
        and str(block["content"]).strip() in {"91", "92"}
    ]

    figure2 = _blocks_containing(model_list[2], "Fig. 2. Pathological changes")
    figure6 = _blocks_containing(model_list[3], "Fig. 6. Effect of ecabet")
    assert len(figure2) == len(figure6) == 1
    assert figure2[0]["type"] == figure6[0]["type"] == "text"
    assert "Under anesthesia with ether" in figure2[0]["content"]
    assert "pepsin. Isolated and everted esophagus" in figure6[0]["content"]
    assert not [block for block in all_blocks if block["type"] == "footnote"]

    online_footer = _blocks_containing(model_list[4], "http://www.birkhauser.ch/IPh")
    assert len(online_footer) == 1
    assert online_footer[0]["type"] == "footer"


def test_mixed_elements_pages_39_40_force_txt_regressions() -> None:
    """验证原文 39–40 页三个公式和公式后同视觉行正文的唯一输出。"""

    model_list = _txt_model_list("mixed_elements_pages_39_40.pdf")
    page40 = model_list[1]
    equations = [block for block in page40 if block["type"] == "equation"]

    assert len(equations) == 3
    assert sum("(8)" in str(block["content"]) for block in equations) == 1
    assert sum("(9)" in str(block["content"]) for block in equations) == 1
    inequality = _blocks_containing(equations, "Mπ(Xn)")
    assert len(inequality) == 1
    assert inequality[0]["type"] == "equation"

    continuation = _blocks_containing(page40, "so that we have a near-unbiased estimator")
    assert len(continuation) == 1
    assert continuation[0]["type"] == "text"
    assert "Coupling this observation" in continuation[0]["content"]
    assert sum(
        "Coupling this observation" in str(block["content"])
        for block in page40
    ) == 1
    equation9_following = _blocks_containing(
        page40,
        "Clearly, the random variable",
    )
    assert len(equation9_following) == 1
    assert equation9_following[0]["type"] == "text"
    assert "is an unbiased estimator" in equation9_following[0]["content"]
    assert "McAllester and Schapire" in equation9_following[0]["content"]
