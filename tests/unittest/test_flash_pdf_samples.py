from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterator
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
from bs4 import BeautifulSoup
from pypdf import PdfReader

from mineru.backend.postprocess.page_blocks import process_page_blocks
from mineru.backend.postprocess.pages import model_json_to_pages
from mineru.model.flash import PdfModel
from mineru.model.flash.pdf import (
    formulas,
    geometry,
    graphics,
    line_merging,
    models,
    native_text,
    tables,
)
from mineru.render import render_markdown
from mineru.types import MiddleJson, ModelJson
from mineru.model.flash.pdf.document import PDFDocument, get_lines_from_chars

from _span_test_utils import inline_text, inline_urls, visible_content


_PROJECT_ROOT = Path(__file__).parents[2]
_DEMO_PDF_DIR = _PROJECT_ROOT / "demo" / "pdfs"
_FIXTURE_PDF_DIR = Path(__file__).resolve().parent / "pdfs"
_FLASH_SYNTHETIC_PDF_NAME = "flash_table_annotations_synthetic.pdf"
_CJK_SYNTHETIC_PDF_NAME = "native_cjk_layout_synthetic.pdf"
_SAFE_WATERMARK_TEXT = "MINERU TEST WATERMARK"


def _pdf_cache_key(pdf_path: Path) -> tuple[str, int, int]:
    """以规范路径、文件大小和纳秒 mtime 构造单次 pytest 进程内缓存键。"""

    resolved = pdf_path.resolve()
    stat = resolved.stat()
    return str(resolved), stat.st_size, stat.st_mtime_ns


@lru_cache(maxsize=None)
def _cached_model_list(
    pdf_path: str,
    _size: int,
    _mtime_ns: int,
) -> list[list[dict[str, Any]]]:
    """每份不可变真实 PDF 只运行一次 Flash 预测，并缓存原始模型输出。"""

    with PDFDocument(Path(pdf_path).read_bytes()) as pdf_doc:
        return PdfModel().predict(pdf_doc)


def _cached_model_list_copy(pdf_path: Path) -> list[list[dict[str, Any]]]:
    """返回缓存模型输出的深拷贝，隔离不同测试对可变 block 的修改。"""

    return deepcopy(_cached_model_list(*_pdf_cache_key(pdf_path)))


@pytest.fixture(scope="module", autouse=True)
def _clear_real_pdf_model_cache_after_module() -> Iterator[None]:
    """模块结束后释放真实 PDF 模型缓存，避免影响后续 unittest 内存。"""

    yield
    _cached_model_list.cache_clear()


def _model_json(
    pages: list[list[dict[str, Any]]],
    *,
    page_index_map: list[int] | None = None,
) -> ModelJson:
    """为 Flash 样例后处理构造最小严格 ModelJson。"""
    return ModelJson(
        pages=pages,
        page_index_map=page_index_map or [],
        file_suffix="pdf",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )


# demo4 第 9 至 11 页中实际跨物理行显示的 URL，用于确保纯正则拼接覆盖全部形态。
_DEMO4_WRAPPED_URLS = (
    "https://doi.org/10.1007/s00259-025-07388-8",
    "https://doi.org/10.37921/690910twdfoo",
    "https://www.cancerimagingarchive.net/collection/acrin-nsclc-fdg-pet/",
    "http://creativecommons.org/licenses/by/4.0/",
    "https://doi.org/10.3322/caac.21660",
    "https://doi.org/10.1016/j.mayocp.2019.01.013",
    "https://doi.org/10.1016/j.jiph.2012.09.003",
    "https://doi.org/10.1016/s0140-6736(02)08352-6",
    "https://doi.org/10.1016/s0140-6736(02)08388-5",
    "https://doi.org/10.1378/chest.123.1_suppl.137s",
    "https://doi.org/10.1016/j.ejrad.2009.01.036",
    "https://doi.org/10.2214/ajr.16.16532",
    "https://doi.org/10.1200/jco.2012.47.5947",
    "https://doi.org/10.1007/s00259-022-05832-7",
    "https://doi.org/10.1016/j.radonc.2022.04.003",
    "https://doi.org/10.1016/j.crad.2015.03.010",
    "https://doi.org/10.3390/ijms22084120",
    "https://doi.org/10.1016/j.bone.2015.05.046",
    "https://doi.org/10.1016/j.cell.2007.05.047",
    "https://doi.org/10.1016/j.cell.2010.06.003",
    "https://doi.org/10.1038/s41577-019-0178-8",
    "https://doi.org/10.3389/fmed.2021.740615",
    "https://doi.org/10.1038/s41467-020-16878-2",
    "https://doi.org/10.1186/s12957-018-1439-x",
    "https://doi.org/10.1097/mnm.0000000000000483",
    "https://doi.org/10.1080/00031305.1992.10475879",
    "https://doi.org/10.3389/fmed.2025.1597844",
    "https://doi.org/10.1109/tpami.2024.3400281",
    "https://doi.org/10.3389/fonc.2022.922465",
    "https://doi.org/10.1002/acn3.121",
    "https://doi.org/10.36660/abc.20210463",
    "https://doi.org/10.18632/oncotarget.11816",
    "https://doi.org/10.1007/s00259-023-06192-6",
    "https://doi.org/10.2215/cjn.04151206",
    "https://doi.org/10.1038/s41593-018-0213-2",
    "https://doi.org/10.1038/s41698-022-00286-4",
    "https://doi.org/10.1016/j.bone.2022.116540",
    "https://doi.org/10.3389/fimmu.2023.1222129",
    "https://doi.org/10.5483/bmbrep.2008.41.7.495",
    "https://doi.org/10.1016/j.xjon.2022.09.001",
)


def _native_model_list(
    pdf_name: str,
    *,
    pdf_dir: Path = _DEMO_PDF_DIR,
) -> list[list[dict[str, Any]]]:
    """运行仓库内数字 PDF 样例并返回 Flash 原生模型输出。"""

    return _cached_model_list_copy(pdf_dir / pdf_name)


def _txt_model_list(
    pdf_name: str,
    *,
    pdf_dir: Path = _DEMO_PDF_DIR,
) -> list[list[dict[str, Any]]]:
    """显式使用 Flash TXT 模式解析仓库内回归 PDF，禁止经过 auto 分类。"""

    return _cached_model_list_copy(pdf_dir / pdf_name)


def _auto_model_list(pdf_name: str) -> list[list[dict[str, Any]]]:
    """通过稳定公共入口以 auto 模式解析仓库内最小回归 PDF。"""

    pdf_path = Path(__file__).parents[2] / "demo" / "pdfs" / pdf_name
    with PDFDocument(pdf_path.read_bytes()) as pdf_doc:
        assert pdf_doc.classify() == "txt"
    return _cached_model_list_copy(pdf_path)


def _native_table_counts(pdf_name: str) -> list[int]:
    """返回仓库内数字 PDF 样例的逐页表格块数量。"""

    return [sum(block["type"] == "table" for block in page) for page in _native_model_list(pdf_name)]


def test_real_pdf_model_cache_reuses_parse_and_isolates_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证相同文件只预测一次，且调用方修改深拷贝不会污染缓存。"""

    pdf_path = tmp_path / "cache-fixture.pdf"
    pdf_path.write_bytes(b"cache fixture")
    predict_calls: list[object] = []

    class FakePDFDocument:
        """提供缓存测试所需的最小 PDFDocument 上下文。"""

        def __init__(self, _payload: bytes) -> None:
            pass

        def __enter__(self) -> FakePDFDocument:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakePdfModel:
        """记录预测次数并返回可变模型块。"""

        def predict(self, document: object) -> list[list[dict[str, Any]]]:
            predict_calls.append(document)
            return [[{"type": "text", "content": "stable"}]]

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "PDFDocument", FakePDFDocument)
    monkeypatch.setattr(module, "PdfModel", FakePdfModel)

    first = _cached_model_list_copy(pdf_path)
    second = _cached_model_list_copy(pdf_path)
    first[0][0]["content"] = "changed"
    third = _cached_model_list_copy(pdf_path)

    assert len(predict_calls) == 1
    assert first is not second
    assert second[0][0]["content"] == "stable"
    assert third[0][0]["content"] == "stable"


def _native_page_source(pdf_name: str, page_idx: int) -> models._PageSource:
    """读取指定样例页并构造候选检测与认领测试使用的页面源。"""

    pdf_path = Path(__file__).parents[2] / "demo" / "pdfs" / pdf_name
    with PDFDocument(str(pdf_path)) as pdf_doc:
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


def _grouped_visual_blocks(
    page_blocks: list[dict[str, Any]],
    block_type: str,
) -> list[dict[str, Any]]:
    """通过新的单页后处理入口返回指定类型的视觉容器，且不修改原始样例结果。"""
    return [block for block in process_page_blocks(deepcopy(page_blocks), use_bbox=True) if block.get("type") == block_type]


def _visible_content(value: Any) -> str:
    """返回 model block 的专用字符串或 InlineSpan 可见文本。"""

    content = value.get("content") if isinstance(value, dict) else value
    if isinstance(content, str) and content.lstrip().lower().startswith("<table"):
        soup = BeautifulSoup(content, "html.parser")
        content = " ".join(cell.get_text() for cell in soup.find_all(["td", "th"]))
    return visible_content(content)


def _html_table_rows(content: str) -> list[tuple[str, ...]]:
    """读取原生结构恢复输出的 HTML 表格行与单元格纯文本。"""

    soup = BeautifulSoup(content, "html.parser")
    return [tuple(cell.get_text() for cell in row.find_all(["td", "th"], recursive=False)) for row in soup.find_all("tr")]


def _normalized_content_probe(text: str) -> str:
    """还原 LaTeX tag 并去除排版差异，生成原生行内容覆盖检查使用的探针。"""

    text = _visible_content(text)
    text = re.sub(r"\\tag\{([^{}]+)\}", r"\1", text)
    return "".join(char.casefold() for char in text if char.isalnum())


def _unsafe_flash_content_characters(text: str) -> list[str]:
    """返回 Flash content 中不应跨接口保留的排版空白与控制字符。"""

    return [
        char
        for char in text
        if (
            (unicodedata.category(char).startswith("Z") and char != " ")
            or (unicodedata.category(char) == "Cc" and char != "\n")
            or char in {"\u00ad", "\u200b", "\u2060", "\ufeff"}
        )
    ]


def _blocks_containing(
    blocks: list[dict[str, Any]],
    probe: str,
) -> list[dict[str, Any]]:
    """返回归一化内容中包含指定探针的模型块。"""

    normalized_probe = _normalized_content_probe(probe)
    return [block for block in blocks if normalized_probe in _normalized_content_probe(_visible_content(block))]


def test_explicit_pdf_fixtures_keep_expected_txt_block_inventory() -> None:
    """验证显式登记的 demo 与合成 PDF 逐文件保持预期页数和块库存。"""

    expected_inventory = {
        "IEBM_A_2667169_O-5.pdf": (1, Counter({"equation": 16, "header": 2, "text": 12})),
        "caibao1.pdf": (
            22,
            Counter(
                {
                    "caption": 15,
                    "footer": 22,
                    "footnote": 17,
                    "header": 43,
                    "image": 14,
                    "page_number": 22,
                    "paragraph_title": 26,
                    "table": 20,
                    "text": 70,
                }
            ),
        ),
        "demo1.pdf": (
            13,
            Counter(
                {
                    "caption": 10,
                    "doc_title": 1,
                    "equation": 7,
                    "footer": 1,
                    "footnote": 5,
                    "header": 12,
                    "image": 8,
                    "page_footnote": 1,
                    "page_number": 12,
                    "paragraph_title": 18,
                    "table": 5,
                    "text": 85,
                }
            ),
        ),
        "demo2.pdf": (
            6,
            Counter(
                {
                    "caption": 7,
                    "doc_title": 1,
                    "equation": 9,
                    "footer": 1,
                    "footnote": 2,
                    "image": 8,
                    "paragraph_title": 10,
                    "table": 2,
                    "text": 62,
                }
            ),
        ),
        "demo3.pdf": (
            10,
            Counter(
                {
                    "aside_text": 1,
                    "caption": 8,
                    "doc_title": 1,
                    "equation": 8,
                    "image": 1,
                    "page_footnote": 6,
                    "paragraph_title": 21,
                    "table": 9,
                    "text": 118,
                }
            ),
        ),
        "demo4.pdf": (
            11,
            Counter(
                {
                    "caption": 10,
                    "doc_title": 1,
                    "footer": 11,
                    "header": 11,
                    "image": 5,
                    "page_footnote": 9,
                    "page_number": 10,
                    "paragraph_title": 12,
                    "table": 1,
                    "text": 112,
                }
            ),
        ),
        "demo6.pdf": (7, Counter({"image": 2, "paragraph_title": 9, "text": 24})),
        "mixed_elements_pages_03_06.pdf": (
            4,
            Counter(
                {
                    "caption": 4,
                    "doc_title": 1,
                    "equation": 1,
                    "footer": 4,
                    "header": 5,
                    "image": 7,
                    "page_number": 3,
                    "paragraph_title": 5,
                    "table": 1,
                    "text": 32,
                }
            ),
        ),
        "mixed_elements_pages_07_10.pdf": (
            4,
            Counter(
                {
                    "caption": 6,
                    "code": 3,
                    "doc_title": 1,
                    "footer": 3,
                    "image": 2,
                    "page_number": 4,
                    "paragraph_title": 16,
                    "table": 2,
                    "text": 57,
                }
            ),
        ),
        "mixed_elements_pages_11_15.pdf": (
            5,
            Counter(
                {
                    "caption": 7,
                    "doc_title": 1,
                    "footer": 1,
                    "header": 8,
                    "image": 8,
                    "page_footnote": 1,
                    "page_number": 4,
                    "paragraph_title": 13,
                    "table": 1,
                    "text": 50,
                }
            ),
        ),
        "mixed_elements_pages_39_40.pdf": (
            2,
            Counter({"equation": 3, "page_number": 2, "paragraph_title": 2, "text": 15}),
        ),
        "small_ocr.pdf": (8, Counter({"header": 49, "image": 447})),
        "中文论文.pdf": (
            10,
            Counter(
                {
                    "caption": 18,
                    "doc_title": 2,
                    "equation": 6,
                    "footnote": 1,
                    "header": 24,
                    "image": 8,
                    "page_footnote": 4,
                    "page_number": 9,
                    "paragraph_title": 19,
                    "table": 3,
                    "text": 106,
                }
            ),
        ),
        "中文论文3.pdf": (
            4,
            Counter(
                {
                    "caption": 6,
                    "doc_title": 1,
                    "header": 12,
                    "image": 3,
                    "page_footnote": 4,
                    "page_number": 3,
                    "paragraph_title": 6,
                    "text": 56,
                }
            ),
        ),
        "中文论文4.pdf": (
            5,
            Counter(
                {
                    "caption": 10,
                    "doc_title": 2,
                    "header": 19,
                    "image": 5,
                    "page_footnote": 1,
                    "paragraph_title": 13,
                    "table": 5,
                    "text": 46,
                }
            ),
        ),
        _FLASH_SYNTHETIC_PDF_NAME: (
            8,
            Counter(
                {
                    "caption": 6,
                    "footer": 1,
                    "footnote": 3,
                    "page_number": 8,
                    "table": 6,
                    # 第 5 页三条等节奏句号行按当前正文段界规则拆开；视觉 gold 已完成人工批准。
                    "text": 18,
                }
            ),
        ),
        _CJK_SYNTHETIC_PDF_NAME: (
            4,
            Counter(
                {
                    "caption": 2,
                    "image": 2,
                    "index": 1,
                    "page_number": 4,
                    "paragraph_title": 2,
                    "text": 12,
                }
            ),
        ),
    }
    pdf_paths = {pdf_name: _DEMO_PDF_DIR / pdf_name for pdf_name in expected_inventory}
    pdf_paths[_FLASH_SYNTHETIC_PDF_NAME] = _FIXTURE_PDF_DIR / _FLASH_SYNTHETIC_PDF_NAME
    pdf_paths[_CJK_SYNTHETIC_PDF_NAME] = _FIXTURE_PDF_DIR / _CJK_SYNTHETIC_PDF_NAME
    actual_inventory: dict[str, tuple[int, Counter[str]]] = {}
    unsafe_content: list[tuple[str, int, int, list[str]]] = []
    for pdf_name, pdf_path in pdf_paths.items():
        model_list = _txt_model_list(pdf_name, pdf_dir=pdf_path.parent)
        actual_inventory[pdf_name] = (
            len(model_list),
            Counter(block["type"] for page in model_list for block in page),
        )
        unsafe_content.extend(
            (pdf_name, page_index, block_index, unsafe_chars)
            for page_index, page in enumerate(model_list)
            for block_index, block in enumerate(page)
            if (unsafe_chars := _unsafe_flash_content_characters(str(block.get("content", ""))))
        )

    assert actual_inventory == expected_inventory
    assert unsafe_content == []


def test_demo1_keeps_five_real_tables_without_formula_false_positive() -> None:
    """验证 demo1 首页脚注尾段页脚、参考文献、公式与五个真实表格均正确。"""

    model_list = _native_model_list("demo1.pdf")

    assert [len(page) for page in model_list] == [16, 9, 12, 18, 12, 13, 11, 10, 12, 7, 10, 26, 9]
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
    assert sum(block["type"] == "caption" for page in model_list for block in page) == 10
    assert sum(block["type"] == "footnote" for page in model_list for block in page) == 5
    middle_pages = model_json_to_pages(_model_json(model_list))
    page3_continuation = next(
        block for block in middle_pages[2].blocks if block.type == "text" and inline_text(block.content).startswith("were to")
    )
    assert page3_continuation.continues_prev is True
    assert [block["bbox"] for block in model_list[5] if block["type"] == "table"] == [
        [0.087, 0.153, 0.922, 0.333],
        [0.087, 0.697, 0.922, 0.876],
    ]
    assert [block["bbox"] for block in model_list[8] if block["type"] == "table"] == [[0.078, 0.167, 0.913, 0.333]]
    page5_visual_blocks = [block for block in model_list[4] if block["type"] in {"caption", "table", "footnote"}]
    assert [block["type"] for block in page5_visual_blocks] == [
        "caption",
        "table",
        "footnote",
    ]
    assert [block["bbox"] for block in page5_visual_blocks] == [
        [0.076, 0.727, 0.109, 0.892],
        [0.117, 0.125, 0.431, 0.891],
        [0.438, 0.434, 0.454, 0.892],
    ]
    assert [block["angle"] for block in page5_visual_blocks] == [270, 270, 270]
    assert _visible_content(page5_visual_blocks[2]) == (
        "For *rainfall distribution, U, uniform; W, winter dominated; S, summer dominated. BFI, baseflow index."
    )
    assert [[child["type"] for child in group["content"]] for group in _grouped_visual_blocks(model_list[5], "table")] == [
        ["table_caption", "table_body", "table_footnote"],
        ["table_caption", "table_body", "table_footnote"],
    ]
    page1_footnotes = [block for block in model_list[0] if block["type"] == "page_footnote"]
    assert len(page1_footnotes) == 1
    assert _visible_content(page1_footnotes[0]).startswith("* Corresponding author.")
    copyright_block = next(block for block in model_list[0] if _visible_content(block).startswith("0022-1694/$"))
    assert copyright_block["type"] == "footer"
    assert copyright_block["bbox"] == [0.077, 0.869, 0.514, 0.895]
    assert "doi:10.1016/j.jhydrol.2005.01.006" in _visible_content(copyright_block)
    assert model_list[0].index(page1_footnotes[0]) < model_list[0].index(copyright_block)
    assert next(block for block in model_list[0] if _visible_content(block) == "Abstract")["type"] == "paragraph_title"
    assert next(block for block in model_list[6] if _visible_content(block).startswith("4.2."))["type"] == "paragraph_title"


def test_demo1_rotated_table_claims_all_206_lines_without_residual_text() -> None:
    """验证 demo1 第五页旋转表完整认领 206 行且表框内没有残留文本。"""

    source = _native_page_source("demo1.pdf", 4)
    candidates = tables._detect_table_candidates(source)
    blocks, annotation_blocks, claimed = tables._materialize_table_blocks(
        source,
        candidates,
    )
    rotated_indices = {line.source_index for line in source.lines if line.angle == 270}

    assert len(blocks) == 1
    assert {block["type"] for block in annotation_blocks} == {
        "caption",
        "footnote",
    }
    assert next(block for block in annotation_blocks if block["type"] == "caption")["bbox"] == next(
        annotation.bbox for annotation in candidates[0].annotations if annotation.kind == "caption"
    )
    assert blocks[0]["bbox"] == candidates[0].core_bbox
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
    """验证 demo2 首页正文、Abstract 粗体和 Figure 1 图文归属保持正确。"""

    page = _native_model_list("demo2.pdf")[0]
    graphic_block = next(block for block in page if "Left camera" in _visible_content(block))
    caption_block = next(block for block in page if "Figure 1:" in _visible_content(block))
    abstract_block = next(block for block in page if _visible_content(block).startswith("Abstract—Stereo"))

    assert len(page) == 16
    assert not [block for block in page if block["type"] == "table"]
    assert next(block for block in page if _visible_content(block).startswith("Real-time Temporal"))["type"] == "doc_title"
    assert next(block for block in page if _visible_content(block) == "I. INTRODUCTION")["type"] == "paragraph_title"
    for expected_text in ("dp", "¯x", "p =", "¯p =", "Left camera", "Right camera"):
        assert expected_text in _visible_content(graphic_block)
    assert graphic_block is not caption_block
    assert caption_block["type"] == "caption"
    assert "Figure 1:" not in _visible_content(graphic_block)
    assert "Left camera" not in _visible_content(caption_block)
    assert sum(_visible_content(block).count("Left camera") for block in page) == 1
    assert graphic_block["bbox"] == [0.516, 0.311, 0.913, 0.442]
    copyright_block = next(block for block in page if _visible_content(block).startswith("978-1-4673-5208-6"))
    assert copyright_block["type"] == "footer"
    assert _visible_content(abstract_block).startswith("Abstract—Stereo")
    assert _visible_content(abstract_block).endswith("Middlebury stereo performance benchmark.")
    assert abstract_block["content"][0].get("styles") == ["bold"]
    assert sum(span.get("styles") == ["bold"] for span in abstract_block["content"]) == 1
    assert "de<text" not in _visible_content(abstract_block)

    page_markdown = render_markdown(
        MiddleJson(
            pages=model_json_to_pages(_model_json([page], page_index_map=[0])),
            is_full_document=False,
            file_suffix="pdf",
            effort="flash",
            parse_mode="txt",
            mineru_version="test",
        )
    )
    abstract_markdown = next(
        paragraph for paragraph in page_markdown.split("\n\n") if paragraph.startswith("**Abstract—Stereo")
    )
    assert abstract_markdown.endswith("Middlebury stereo performance benchmark.**")
    assert abstract_markdown.count("**") == 2


def test_demo2_pages2_to6_restore_paragraphs_formulas_and_reading_order() -> None:
    """验证 demo2 后续页达到目标块数，正文、公式、caption 与双栏顺序均稳定。"""

    model_list = _native_model_list("demo2.pdf")

    assert [len(page) for page in model_list] == [16, 16, 21, 15, 18, 16]
    assert [sum(block["type"] == "image" for block in page) for page in model_list] == [1, 0, 0, 5, 2, 0]
    assert [sum(block["type"] == "table" for block in page) for page in model_list] == [0, 0, 0, 1, 1, 0]
    assert sum(block["type"] == "equation" for page in model_list for block in page) == 9
    assert sum(block["type"] == "caption" for page in model_list for block in page) == 7
    assert sum(block["type"] == "footnote" for page in model_list for block in page) == 2

    page2 = model_list[1]
    page2_contents = [_visible_content(block) for block in page2]
    humans = next(content for content in page2_contents if content.startswith("Humans group shapes"))
    matching = next(content for content in page2_contents if content.startswith("To identify a match"))
    dissimilarity = next(content for content in page2_contents if content.startswith("where the pixel dissimilarity"))
    formula3 = next(content for content in page2_contents if content.endswith(r"\tag{3}"))
    assert not [content for content in page2_contents if content.strip() == "by"]
    assert humans.endswith("is given by")
    assert "Sp denotes a set of matching candidates" in matching
    assert "green, and blue components given by" in dissimilarity
    assert "green, and blue" not in formula3

    page3 = model_list[2]
    page3_contents = [_visible_content(block) for block in page3]
    assert (
        next(block for block in page3 if _visible_content(block) == "B. Temporal cost aggregation")["type"] == "paragraph_title"
    )
    assert page3_contents[12] == "D. Iterative Disparity Refinement"
    assert all(block["bbox"][2] <= 0.5 for block in page3[:12])
    assert "O(ω2) to O(ω)" in page3_contents[0]
    assert "disparity estimates Dip" in page3_contents[13]
    for formula_number in range(4, 10):
        marker = rf"\tag{{{formula_number}}}"
        assert sum(marker in content for content in page3_contents) == 1
    formula7 = next(content for content in page3_contents if r"\tag{7}" in content)
    formula4 = next(content for content in page3_contents if r"\tag{4}" in content)
    formula8 = next(content for content in page3_contents if r"\tag{8}" in content)
    assert formula4.endswith(r"\tag{4}")
    assert formula8.endswith(r"\tag{8}")
    assert "\n(4)" not in formula4
    assert "\n(8)" not in formula8
    assert "Fp =" in formula7
    assert "otherwise" in formula7
    assert not [content for content in page3_contents if content.strip() in {"2", "p", "otherwise", "(7)"}]
    assert "available at http://mc2.unl.edu/current-research/image-processing/. Figure 2" in page3_contents[-1]
    assert "current-research /image-processing" not in page3_contents[-1]

    page4_contents = [_visible_content(block) for block in model_list[3]]
    figure2 = next(content for content in page4_contents if content.startswith("Figure 2:"))
    figure3 = next(content for content in page4_contents if content.startswith("Figure 3:"))
    results = next(content for content in page4_contents if content.startswith("The results of temporal stereo"))
    improvements = next(content for content in page4_contents if content.startswith("Significant improvements"))
    assert figure2.endswith("(4th row).")
    assert figure3.endswith("without temporal aggregation.")
    assert results.endswith("methods that operate on pairs of images.")
    assert improvements.endswith("has the effect")

    page5 = model_list[4]
    page5_contents = [_visible_content(block) for block in page5]
    optimal_feedback = next(content for content in page5_contents if content.startswith("The optimal value"))
    page5_references = [content for content in page5_contents if content.startswith("[")]
    references_title = next(block for block in page5 if _visible_content(block) == "REFERENCES")
    assert "noise ranging between ±0 to ±40" in optimal_feedback
    assert optimal_feedback.endswith("temporal stereo matching is used.")
    assert references_title["type"] == "paragraph_title"
    assert [content.partition("]")[0] + "]" for content in page5_references] == [f"[{number}]" for number in range(1, 6)]

    page6_contents = [_visible_content(block) for block in model_list[5]]
    assert len(page6_contents) == 16
    assert [content.partition("]")[0] + "]" for content in page6_contents] == [f"[{number}]" for number in range(6, 22)]


def test_demo2_container_claims_are_pairwise_disjoint() -> None:
    """验证表格、图形和公式阶段按 source_index 唯一认领，不重复消费文本身份。"""

    for page_idx in (1, 2, 3):
        source = _native_page_source("demo2.pdf", page_idx)
        table_candidates = tables._detect_table_candidates(source)
        table_blocks, _table_annotations, table_claimed = tables._materialize_table_blocks(source, table_candidates)
        table_bboxes = [block["bbox"] for block in table_blocks]
        _graphic_blocks, graphic_claimed = graphics._build_graphic_like_blocks(
            source,
            table_bboxes,
            table_claimed,
        )
        remaining = line_merging._merge_same_baseline_text_lines(
            [line for line in source.lines if line.source_index not in table_claimed | graphic_claimed],
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
    """验证 demo2 第四页正文尾行不被图形吸收，五个图形和 Table 1 均保持完整。"""

    page = _native_model_list("demo2.pdf")[3]
    table_blocks = [block for block in page if block["type"] == "table"]
    graphic_markers = ("Frame 30", "Noise: ±0", "Noise: ±20", "Noise: ±40", "Noise ±")
    graphic_blocks = [
        next(block for block in page if block["type"] == "image" and marker in _visible_content(block))
        for marker in graphic_markers
    ]

    assert len(table_blocks) == 1
    table_caption = next(block for block in page if _visible_content(block).startswith("Table I:"))
    assert table_caption["type"] == "caption"
    assert "Table I:" not in table_blocks[0]["content"]
    assert "Symbol" in table_blocks[0]["content"]
    assert len({id(block) for block in graphic_blocks}) == 5
    assert "Frame 90" in _visible_content(graphic_blocks[0])
    assert all("Figure" not in _visible_content(block) for block in graphic_blocks)
    body_tail_block = next(block for block in page if _visible_content(block).startswith("of the synthetic stereo scene"))
    assert body_tail_block["type"] == "text"
    assert _visible_content(body_tail_block).endswith("discontinuity map.")
    assert "discontinuity map." not in graphic_blocks[0]["content"]


def test_demo2_table_captions_and_numeric_footnotes_are_independent_blocks() -> None:
    """验证 demo2 两张表的换行标题和数字脚注独立输出并绑定为三段子块。"""

    model_list = _native_model_list("demo2.pdf")
    page4_table = next(block for block in model_list[3] if block["type"] == "table")
    page5_table = next(block for block in model_list[4] if block["type"] == "table")
    page4_caption = _blocks_containing(model_list[3], "Table I: Parameters")[0]
    page5_caption = _blocks_containing(model_list[4], "Table II: A comparison")[0]
    page4_footnote = _blocks_containing(
        model_list[3],
        "1 To enable propagation of disparity information",
    )[0]
    page5_footnote = _blocks_containing(
        model_list[4],
        "1 Millions of Disparity Estimates per Second",
    )[0]
    residual_text = "\n".join(
        _visible_content(block) for page_idx in (3, 4) for block in model_list[page_idx] if block["type"] == "text"
    )

    assert page4_caption["type"] == page5_caption["type"] == "caption"
    assert page4_footnote["type"] == page5_footnote["type"] == "footnote"
    assert "ral stereo matching." in _visible_content(page4_caption)
    assert "0.01, respectively." in _visible_content(page4_footnote)
    assert "Noise: ±20" not in page4_table["content"]
    assert "2 Assumes 320 × 240 images with 32 disparity levels." in _visible_content(page5_footnote)
    assert "the avgerage % of bad pixels." in _visible_content(page5_footnote)
    assert "Table I:" not in page4_table["content"]
    assert "To enable propagation" not in page4_table["content"]
    assert "Table II:" not in page5_table["content"]
    assert "Millions of Disparity" not in page5_table["content"]
    expected_table2_rows = [
        ("Method", "GPU", "MDE/s1", "FPS2", "Error3"),
        ("Our Method", "GeForce GTX 680", "215.7", "90", "6.20"),
        ("CostFilter [10]", "GeForce GTX 480", "57.9", "24", "5.55"),
        ("FastBilateral [7]", "Tesla C2070", "50.6", "21", "7.31"),
        ("RealtimeBFV [8]", "GeForce 8800 GTX", "114.3", "46", "7.65"),
        ("RealtimeBP [21]", "GeForce 7900 GTX", "20.9", "8", "7.69"),
        ("ESAW [6]", "GeForce 8800 GTX", "194.8", "79", "8.21"),
        ("RealTimeGPU [5]", "Radeon XL1800", "52.8", "21", "9.82"),
        ("DCBGrid [19]", "Quadro FX 5800", "25.1", "10", "10.90"),
    ]
    assert _html_table_rows(page5_table["content"]) == expected_table2_rows
    assert "6.20CostFilter [10]" not in page5_table["content"]
    assert "ral stereo matching." not in residual_text
    assert "Millions of Disparity Estimates per Second." not in residual_text
    assert [page4_caption["bbox"], page4_table["bbox"], page4_footnote["bbox"]] == [
        [0.51, 0.484, 0.915, 0.514],
        [0.539, 0.53, 0.891, 0.638],
        [0.549, 0.639, 0.891, 0.675],
    ]
    assert [page5_caption["bbox"], page5_table["bbox"], page5_footnote["bbox"]] == [
        [0.51, 0.218, 0.915, 0.263],
        [0.52, 0.278, 0.91, 0.39],
        [0.53, 0.391, 0.91, 0.438],
    ]
    for page_index in (3, 4):
        assert [child["type"] for child in _grouped_visual_blocks(model_list[page_index], "table")[0]["content"]] == [
            "table_caption",
            "table_body",
            "table_footnote",
        ]


def test_demo3_keeps_tables_and_covers_every_native_source_line() -> None:
    """验证 demo3 容器、后续页段落边界及每条原生 source line 均保持稳定。"""

    pdf_path = Path(__file__).parents[2] / "demo" / "pdfs" / "demo3.pdf"
    model_list = _native_model_list("demo3.pdf")
    with PDFDocument(str(pdf_path)) as pdf_doc:
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
    assert sum(block["type"] == "caption" for page in model_list for block in page) == 8
    page7_tables = [block for block in model_list[6] if block["type"] == "table"]
    page7_table4 = next(block for block in page7_tables if "Number of parameters" in block["content"])
    page7_table4_caption = next(
        block for block in model_list[6] if _visible_content(block) == "Table 4: Model size comparison."
    )
    page7_inline_body = next(
        block for block in model_list[6] if _visible_content(block).startswith("Row, Column, & Global Positional IDs.")
    )
    page9_conclusion = next(
        block for block in model_list[8] if _visible_content(block).startswith("In this paper, we identified")
    )
    page10_first_reference = next(
        block for block in model_list[9] if _visible_content(block).startswith("Xiang Deng, Huan Sun")
    )
    table4_text = _visible_content(page7_table4)
    assert all(
        marker in table4_text
        for marker in (
            "Model",
            "TAPASBASE",
            "TABLEFORMERBASE",
            "TAPASLARGE",
            "TABLEFORMERLARGE",
        )
    )
    assert _visible_content(page7_table4_caption) not in page7_table4["content"]
    assert page7_table4_caption["type"] == "caption"
    assert page7_table4["bbox"][3] < page7_table4_caption["bbox"][1]
    assert page7_inline_body["type"] == "text"
    assert "With TAPASBASE" in _visible_content(page7_inline_body)
    assert "To tackle this" in _visible_content(page9_conclusion)
    assert "Acknowledgments" not in _visible_content(page9_conclusion)
    assert "Cong Yu. 2021. TURL:" in _visible_content(page10_first_reference)
    assert "Jacob Devlin" not in _visible_content(page10_first_reference)
    for page, source_lines in zip(model_list, source_lines_by_page, strict=True):
        output_probe = _normalized_content_probe("".join(_visible_content(block) for block in page))
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
    assert not [block for block in model_list[6] if block["type"] in {"aside_text", "page_footnote"}]
    assert not [block for page in model_list for block in page if block["type"] == "footer"]


def test_demo3_pages1_and2_fix_title_front_matter_and_embedding_formula() -> None:
    """验证首页标题稳定，第二页标题、公式和栏尾正文各自保持完整。"""

    page1, page2 = _native_model_list("demo3.pdf")[:2]
    title = next(block for block in page1 if _visible_content(block).startswith("TABLEFORMER:"))
    front_matter_contents = {
        "Aditya Gupta† Rahul Goel†",
        "Jingfeng Yang∗ Luheng He†",
        "Shyam Upadhyay† Shachi Paul †",
        "?Georgia Institute of Technology",
        "†Google Assistant",
        "jingfengyangpku@gmail.com",
        "tableformer@google.com",
    }
    front_matter = [block for block in page1 if _visible_content(block) in front_matter_contents]
    released_code = next(block for block in page1 if "TABLEFORMER.md" in _visible_content(block))
    aside = next(block for block in page1 if block["type"] == "aside_text")
    introduction = next(block for block in page1 if _visible_content(block).startswith("Recently, semi-structured"))
    nutshell = next(block for block in page1 if _visible_content(block).startswith("In a nutshell"))
    figure_caption = next(block for block in page1 if _visible_content(block).startswith("Figure 1:"))
    tables_body = next(block for block in page1 if _visible_content(block).startswith("tables or rows"))

    assert title["type"] == "doc_title"
    assert len(front_matter) == len(front_matter_contents)
    assert all(block["type"] == "text" for block in front_matter)
    assert not [block for block in page1 if _visible_content(block) == "∗"]
    assert next(block for block in page1 if _visible_content(block) == "Abstract")["type"] == "paragraph_title"
    assert released_code["type"] == "page_footnote"
    github_target = "https://github.com/google-research/tapas/blob/master/TABLEFORMER.md"
    assert inline_urls(released_code["content"]) == [github_target]
    assert inline_text(released_code["content"]) == f"1Code has been released at {github_target}"
    assert aside["angle"] == 270
    assert aside["bbox"][2] <= 0.12
    assert _visible_content(introduction).endswith("(Eisenschlos et al., 2021; Liu et al., 2021).")
    assert nutshell["type"] == "text"
    assert _visible_content(nutshell).endswith("by serializing")
    assert figure_caption["type"] == "text"
    assert _visible_content(figure_caption).endswith("both questions.")
    assert "tables or rows" not in _visible_content(figure_caption)
    assert tables_body["type"] == "text"

    section_title = next(block for block in page2 if _visible_content(block).startswith("2 Preliminaries:"))
    equations = [block for block in page2 if block["type"] == "equation"]
    assert section_title["type"] == "paragraph_title"
    assert _visible_content(section_title) == "2 Preliminaries: TAPAS for Table Encoding"
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
        if "As for the model" in _visible_content(block)
        or "attends to all the tokens." in _visible_content(block)
        or "Let the layer input" in _visible_content(block)
    ]
    assert len(as_model_blocks) == 1
    assert as_model_blocks[0]["type"] == "text"
    assert _visible_content(as_model_blocks[0]).startswith("As for the model")
    assert "attends to all the tokens." in _visible_content(as_model_blocks[0])
    assert "Let the layer input" in _visible_content(as_model_blocks[0])


def test_demo3_pages6_7_and10_fix_caption_inline_titles_and_reference_tail() -> None:
    """验证跨栏 caption、行内粗体正文与参考文献尾行均保持正确归属。"""

    model_list = _native_model_list("demo3.pdf")
    page6 = model_list[5]
    page7 = model_list[6]
    page9 = model_list[8]
    page10 = model_list[9]

    table2_caption = next(block for block in page6 if _visible_content(block).startswith("Table 2:"))
    assert table2_caption["type"] == "caption"
    assert "Median of 5 independent runs are reported." in _visible_content(table2_caption)
    assert _visible_content(table2_caption).endswith("not reported in the original paper.")
    assert sum("not reported in the original paper." in _visible_content(block) for block in page6) == 1

    attention_bias = next(block for block in page7 if _visible_content(block).startswith("Attention Bias Scaling."))
    positional_ids = next(
        block for block in page7 if _visible_content(block).startswith("Row, Column, & Global Positional IDs.")
    )
    formula6 = next(block for block in page7 if block["type"] == "equation" and r"\tag{6}" in block["content"])
    assert attention_bias["type"] == "text"
    assert attention_bias["content"][0].get("styles") == ["bold"]
    assert attention_bias["content"][0].get("content") == "Attention Bias Scaling."
    assert _visible_content(attention_bias).endswith("attention score by:")
    assert positional_ids["type"] == "text"
    assert "With TAPASBASE" in _visible_content(positional_ids)
    assert formula6["content"].endswith(r"\tag{6}")
    assert not [
        block
        for block in page7
        if block["type"] == "paragraph_title"
        and _visible_content(block).startswith(("Attention Bias Scaling.", "Row, Column, & Global Positional IDs."))
    ]
    page7_markdown = render_markdown(
        MiddleJson(
            pages=model_json_to_pages(_model_json([page7], page_index_map=[6])),
            is_full_document=False,
            file_suffix="pdf",
            effort="flash",
            parse_mode="txt",
            mineru_version="test",
        )
    )
    assert "**Attention Bias Scaling.** Unlike" in page7_markdown

    etc_reference = next(block for block in page9 if "2020.emnlp-main.19" in str(block.get("content", "")))
    etc_target = "https://doi.org/10.18653/v1/2020.emnlp-main.19"
    assert inline_urls(etc_reference["content"]).count(etc_target) == 1
    etc_link = next(span for span in etc_reference["content"] if span.get("type") == "hyperlink")
    assert inline_text(etc_link["content"]) == "ETC: Encoding long and structured inputs in transformers"

    dehyphenated_references = (
        (
            "https://doi.org/10.18653/v1/N19-1423",
            "BERT: Pre-training of deep bidirectional transformers for language understanding",
        ),
        (
            "https://doi.org/10.18653/v1/2020.findings-emnlp.27",
            "Understanding tables with intermediate pre-training",
        ),
        (
            "https://doi.org/10.18653/v1/P17-1167",
            "Search-based neural structured learning for sequential question answering",
        ),
        (
            "https://doi.org/10.18653/v1/D19-1603",
            "Answering conversational questions on structured data without logical forms",
        ),
        (
            "https://doi.org/10.3115/v1/P15-1142",
            "Compositional semantic parsing on semi-structured tables",
        ),
    )
    for target, label in dehyphenated_references:
        reference = next(block for block in page10 if target in str(block.get("content", "")))
        assert inline_urls(reference["content"]).count(target) == 1
        link = next(span for span in reference["content"] if span.get("type") == "hyperlink" and span.get("url") == target)
        assert inline_text(link["content"]) == label

    ying_reference = next(block for block in page10 if _visible_content(block).startswith("Chengxuan Ying"))
    yu_reference = next(block for block in page10 if _visible_content(block).startswith("Tao Yu"))
    assert ying_reference["type"] == yu_reference["type"] == "text"
    assert _visible_content(ying_reference).endswith("arXiv:2106.05234.")
    assert "Tao Yu" not in _visible_content(ying_reference)
    assert not [block for block in page10 if _visible_content(block) == "arXiv:2106.05234."]


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
        if "Figure 2:" in _visible_content(block) or "types of task independent biases" in _visible_content(block)
    ]
    assert len(caption_blocks) == 1
    assert caption_blocks[0]["type"] == "caption"
    assert "This example corresponds to table (a)" in _visible_content(caption_blocks[0])
    assert "associated text." in _visible_content(caption_blocks[0])

    formula1 = next(block for block in page if r"\tag{1}" in block["content"])
    section3 = next(block for block in page if _visible_content(block).startswith("3 TABLEFORMER:"))
    inline_item = next(block for block in page if _visible_content(block).startswith("2) Per cell positional ids."))
    inline_heading = next(block for block in page if _visible_content(block).startswith("Positional Encoding in TABLEFORMER."))
    assert formula1["type"] == "equation"
    assert "Q = HWQ" in formula1["content"] and "K = HWK" in formula1["content"]
    assert section3["type"] == "paragraph_title"
    assert _visible_content(section3) == "3 TABLEFORMER: Robust Structural Table Encoding"
    assert inline_item["type"] == "text" and "To further remove any" in _visible_content(inline_item)
    assert inline_heading["type"] == "text" and "Transformer model" in _visible_content(inline_heading)


def test_demo3_pages4_and5_fix_lists_formula_titles_italics_and_footnotes() -> None:
    """验证第四、五页列表、公式、独立标题、行内标题、斜体续行及脚注边界。"""

    page4, page5 = _native_model_list("demo3.pdf")[3:5]
    left_bullets = [
        block
        for block in page4
        if block["type"] == "text" and _visible_content(block).startswith("•") and block["bbox"][2] <= 0.5
    ]
    assert len(left_bullets) == 6
    assert all(len(_visible_content(block).split()) > 8 for block in left_bullets)
    attention_biases = next(block for block in page4 if _visible_content(block).startswith("Attention Biases in TABLEFORMER."))
    assert attention_biases["type"] == "text"
    assert "13 bias types" in _visible_content(attention_biases)

    formula3 = next(block for block in page4 if r"\tag{3}" in block["content"])
    formula4 = next(block for block in page4 if r"\tag{4}" in block["content"])
    assert formula3["type"] == formula4["type"] == "equation"
    assert formula3 is not formula4
    assert "A =" in formula3["content"] and r"\tag{4}" not in formula3["content"]
    assert r"\tag{3}" not in formula4["content"]

    relation_blocks = [
        block
        for block in page4
        if "Relation between TABLEFORMER and ETC." in _visible_content(block)
        or "ETC (Ainslie et al., 2020)" in _visible_content(block)
    ]
    assert len(relation_blocks) == 1
    assert relation_blocks[0]["type"] == "text"
    assert _visible_content(relation_blocks[0]).startswith("Relation between TABLEFORMER and ETC.")
    assert "uses vectors to represent relative position labels" in _visible_content(relation_blocks[0])

    title4 = next(block for block in page4 if _visible_content(block) == "4 Experimental Setup")
    title41 = next(block for block in page4 if _visible_content(block) == "4.1 Datasets and Evaluation")
    assert title4["type"] == title41["type"] == "paragraph_title"
    for prefix, continuation in (
        ("Table Question Answering.", "conducted experiments"),
        ("Table-Text Entailment.", "TABFACT dataset"),
    ):
        inline_block = next(block for block in page4 if _visible_content(block).startswith(prefix))
        assert inline_block["type"] == "text"
        assert continuation in _visible_content(inline_block)

    assert next(block for block in page5 if _visible_content(block) == "4.2 Baselines")["type"] == "paragraph_title"
    assert (
        next(block for block in page5 if _visible_content(block) == "4.3 Perturbing Tables as Augmented Data")["type"]
        == "paragraph_title"
    )
    italic_body = next(block for block in page5 if _visible_content(block).startswith("Could we alleviate"))
    assert italic_body["type"] == "text"
    assert _visible_content(italic_body).endswith("without making any")

    final_bullet = next(block for block in page5 if _visible_content(block).startswith("• How does TABLEFORMER compare"))
    final_footnote = next(block for block in page5 if _visible_content(block).startswith("3By perturbation"))
    assert final_bullet["type"] == "text"
    assert final_footnote["type"] == "page_footnote"
    assert "3By perturbation" not in _visible_content(final_bullet)


def test_demo4_nct00083083_targeted_flash_regressions() -> None:
    """验证 demo4 的跨栏图注、页眉脚注、续行、公式否决和参考文献分组。"""

    model_list = _native_model_list("demo4.pdf")
    all_content = "\n".join(_visible_content(block) for page in model_list for block in page)
    assert len(_DEMO4_WRAPPED_URLS) == 40
    assert not [url for url in _DEMO4_WRAPPED_URLS if url not in all_content]

    page1 = model_list[0]
    assert any(block["type"] == "header" and "07388-8" in _visible_content(block) for block in page1)
    assert next(block for block in page1 if _visible_content(block) == "ORIGINAL ARTICLE")["type"] == "text"
    author_line = _blocks_containing(page1, "Rucha Ronghe1")[0]
    assert _visible_content(author_line) == (
        "Rucha Ronghe1 · Teresa Crespo Gonzalez2 · Catriona Wimberley3,4,5 · Karla Suchacki3,6 · Adriana A. S. Tavares3,4"
    )
    assert _blocks_containing(page1, "contributed equally")[0]["type"] == "page_footnote"
    assert _blocks_containing(page1, "University of Edinburgh")[0]["type"] == "page_footnote"
    assert [block["bbox"] for block in page1 if block["type"] == "footer"] == [[0.086, 0.934, 0.157, 0.957]]

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
        assert _normalized_content_probe(tail) in _normalized_content_probe(_visible_content(matches[0]))

    email = _blocks_containing(page1, "adriana.tavares@ed.ac.uk")[0]
    affiliation_blocks = [
        block
        for block in page1
        if block["type"] == "page_footnote"
        and block["bbox"][1] > email["bbox"][1]
        and _visible_content(block).lstrip()[:1] in set("123456")
    ]
    assert len(affiliation_blocks) == 6
    assert {_visible_content(block).lstrip()[:1] for block in affiliation_blocks} == set("123456")
    assert not [
        block for block in page1 if block["type"] == "page_footnote" and _visible_content(block).strip() in set("123456")
    ]

    introduction = _blocks_containing(page1, "Cancer is one of the leading causes")
    assert len(introduction) == 1
    assert "throughoutthebody14" in _normalized_content_probe(_visible_content(introduction[0]))

    cross_lane_visual_regions = {
        4: [
            [0.086, 0.073, 0.914, 0.282],
            [0.086, 0.292, 0.486, 0.358],
            [0.514, 0.293, 0.915, 0.359],
        ],
        5: [
            [0.086, 0.073, 0.914, 0.556],
            [0.086, 0.566, 0.486, 0.683],
            [0.514, 0.566, 0.915, 0.67],
        ],
        6: [
            [0.086, 0.073, 0.914, 0.344],
            [0.086, 0.356, 0.486, 0.421],
            [0.514, 0.355, 0.914, 0.408],
        ],
        8: [
            [0.086, 0.073, 0.914, 0.568],
            [0.086, 0.579, 0.486, 0.695],
            [0.514, 0.579, 0.914, 0.682],
        ],
    }
    for page_number, expected_bboxes in cross_lane_visual_regions.items():
        page = model_list[page_number - 1]
        members = []
        for bbox in expected_bboxes:
            matches = [block for block in page if block["bbox"] == bbox]
            assert len(matches) == 1
            members.append(matches[0])
        assert [block["type"] for block in members] == [
            "image",
            "caption",
            "caption",
        ]
        first_index = page.index(members[0])
        assert page[first_index : first_index + 3] == members

    page4 = model_list[3]
    caption_tail = _blocks_containing(page4, "included in the ACRIN 6668 study")
    assert len(caption_tail) == 1
    assert _normalized_content_probe(_visible_content(caption_tail[0])).startswith("inreducingglucoseuptake")
    figure_caption = _blocks_containing(page4, "Fig. 1 Conventional")
    figure_body = _blocks_containing(page4, "in reducing glucose uptake")
    assert len(figure_caption) == 1
    assert len(figure_body) == 1
    assert figure_caption[0] is not figure_body[0]
    assert figure_caption[0]["type"] == "caption"
    assert "inreducingglucoseuptake" not in _normalized_content_probe(_visible_content(figure_caption[0]))
    side_caption = _blocks_containing(page4, "Fig. 2 [18F]FDG uptake")
    side_image = next(block for block in page4 if block["type"] == "image" and block["bbox"] == [0.3, 0.608, 0.914, 0.897])
    assert len(side_caption) == 1
    assert side_caption[0]["type"] == "caption"
    assert side_caption[0]["bbox"][2] < side_image["bbox"][0]
    assert page4.index(side_caption[0]) < page4.index(side_image)
    assert sum(block["type"] == "caption" for page in model_list for block in page) == 10
    page2_table_caption = _blocks_containing(
        model_list[1],
        "Table 1 Patient demographics",
    )[0]
    page2_table = next(block for block in model_list[1] if block["type"] == "table")
    assert [page2_table_caption["type"], page2_table["type"]] == [
        "caption",
        "table",
    ]
    assert [page2_table_caption["bbox"], page2_table["bbox"]] == [
        [0.514, 0.588, 0.914, 0.616],
        [0.514, 0.615, 0.914, 0.894],
    ]
    assert "Patient demographics" not in str(page2_table["content"])
    assert [child["type"] for child in _grouped_visual_blocks(model_list[1], "table")[0]["content"]] == [
        "table_caption",
        "table_body",
    ]

    page5 = model_list[4]
    right_continuation = _blocks_containing(
        page5,
        "also observed when conducting Cox regression",
    )
    assert len(right_continuation) == 1
    assert "importantlythereisahomogenisation" in _normalized_content_probe(str(right_continuation[0]["content"]))

    page6 = model_list[5]
    left_continuation = _blocks_containing(page6, "This study aimed")
    assert len(left_continuation) == 1
    assert "primarytumoursite" in _normalized_content_probe(str(left_continuation[0]["content"]))

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
        and any(_visible_content(block).lstrip().startswith(f"{number}.") for number in range(45, 54))
    ]
    assert len(reference_blocks) == 9
    assert {int(_visible_content(block).lstrip().partition(".")[0]) for block in reference_blocks} == set(range(45, 54))
    reference48 = next(block for block in reference_blocks if _visible_content(block).lstrip().startswith("48."))
    assert "103389fimmu20231222129" in _normalized_content_probe(str(reference48["content"]))
    assert "\u200b" not in str(reference48["content"])
    assert "fimmu.2023.1222129" in str(reference48["content"])


def test_synthetic_flash_table_annotation_regressions() -> None:
    """验证合成 Flash 夹具覆盖表格、表注、跨页边界和空图片页脚。"""

    model_list = _native_model_list(
        _FLASH_SYNTHETIC_PDF_NAME,
        pdf_dir=_FIXTURE_PDF_DIR,
    )

    assert len(model_list) == 8

    page2 = model_list[1]
    page2_tables = [block for block in page2 if block["type"] == "table"]
    assert [block["bbox"] for block in page2_tables] == [
        [0.121, 0.168, 0.877, 0.302],
        [0.121, 0.37, 0.877, 0.504],
    ]
    assert [_html_table_rows(block["content"])[0] for block in page2_tables] == [
        ("ITEM", "VALUE", "STATE"),
        ("METRIC", "LOW", "HIGH"),
    ]
    assert [[child["type"] for child in group["content"]] for group in _grouped_visual_blocks(page2, "table")] == [
        ["table_caption", "table_body"],
        ["table_caption", "table_body"],
    ]
    for table_block in page2_tables:
        table_bbox = table_block["bbox"]
        assert not [
            block
            for block in page2
            if block["type"] == "text"
            and table_bbox[0] <= (block["bbox"][0] + block["bbox"][2]) / 2 <= table_bbox[2]
            and table_bbox[1] <= (block["bbox"][1] + block["bbox"][3]) / 2 <= table_bbox[3]
        ]

    page3 = model_list[2]
    page3_footnotes = [block for block in page3 if block["type"] == "footnote"]
    assert len(page3_footnotes) == 1
    assert page3_footnotes[0]["bbox"] == [0.128, 0.356, 0.505, 0.395]
    assert _visible_content(page3_footnotes[0]).endswith("Its second indented continuation completes the annotation.")
    assert [child["type"] for child in _grouped_visual_blocks(page3, "table")[0]["content"]] == [
        "table_caption",
        "table_body",
        "table_footnote",
    ]

    page4 = model_list[3]
    assert _blocks_containing(page4, "This sentence is inside the border")[0]["type"] == "table"
    assert not [block for block in page4 if block["type"] == "footnote"]

    page5 = model_list[4]
    assert _blocks_containing(page5, "ITEM_001 and VALUE_002")[0]["type"] == "text"
    assert not [block for block in page5 if block["type"] in {"caption", "code", "footnote", "table"}]

    page6 = model_list[5]
    token_table = _blocks_containing(page6, "ROW_001")
    assert len(token_table) == 1
    assert token_table[0]["type"] == "table"
    assert not [block for block in page6 if block["type"] == "code"]
    assert [child["type"] for child in _grouped_visual_blocks(page6, "table")[0]["content"]] == [
        "table_caption",
        "table_body",
        "table_footnote",
    ]

    page7 = model_list[6]
    page8 = model_list[7]
    assert _blocks_containing(page7, "annotation ends on page seven")[0]["type"] == "footnote"
    assert _blocks_containing(page8, "must not merge into the previous page footnote")[0]["type"] == "text"
    assert not [block for block in page8 if block["type"] == "footnote"]

    page_number_block = next(block for block in page8 if block["type"] == "page_number")
    empty_footer = next(block for block in page8 if block["type"] == "footer")
    assert empty_footer["content"] == []
    assert empty_footer["bbox"] == [0.399, 0.968, 0.601, 0.999]
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
    assert {_visible_content(block) for block in blocks if block["type"] == "paragraph_title"} == expected_titles

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
    assert not [block for page in model_list for block in page if block["type"] in {"caption", "footnote"}]


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
        if block["type"] == "text" and 0.65 <= block["bbox"][1] <= 0.87 and block["bbox"][2] - block["bbox"][0] < 0.05
    ]
    math_paragraph = _blocks_containing(
        page4,
        "This allowed us to estimate the low-temperature hop distance",
    )
    assert len(math_paragraph) == 1
    assert math_paragraph[0]["bbox"] == [0.08, 0.666, 0.476, 0.912]
    assert set(math_paragraph[0]) == {"type", "bbox", "angle", "content", "lines"}
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
    assert page4_footer[0]["bbox"] == [0.103, 0.929, 0.45, 0.94]
    assert math_paragraph[0]["bbox"][3] < page4_footer[0]["bbox"][1]
    assert [
        block["bbox"]
        for block in page4
        if block["type"] in {"text", "image", "caption"} and block["bbox"][0] >= 0.49 and block["bbox"][1] >= 0.49
    ] == [
        [0.496, 0.508, 0.893, 0.636],
        [0.503, 0.66, 0.886, 0.871],
        [0.52, 0.887, 0.87, 0.912],
    ]

    page5 = model_list[2]
    axis_label = _blocks_containing(page5, "T–1/4, K–1/4")
    assert len(axis_label) == 1
    assert axis_label[0]["type"] == "image"
    assert _blocks_containing(page5, "MIKOLAICHUK et al.")[0]["type"] == "header"
    assert len([block for block in page5 if block["type"] == "footer"]) == 1
    lower_graphs = [block for block in page5 if block["type"] == "image" and block["bbox"][1] >= 0.6]
    assert len(lower_graphs) == 1
    assert lower_graphs[0]["bbox"] == [0.096, 0.634, 0.46, 0.862]
    assert "σT1/2, S K1/2/m" in str(lower_graphs[0]["content"])
    assert "T–1/4, K–1/4" in str(lower_graphs[0]["content"])
    assert "ular films." not in str(lower_graphs[0]["content"])
    figure4_caption = _blocks_containing(page5, "Fig. 4. Plots of σT1/2")
    assert len(figure4_caption) == 1
    assert figure4_caption[0]["type"] == "caption"
    assert sum(block["type"] == "caption" for page in model_list for block in page) == 4

    figure1_caption = _blocks_containing(page4, "Fig. 1. Electron micrographs")
    panel_images = [block for block in page4 if block["type"] == "image" and block["bbox"][3] < 0.44]
    assert len(figure1_caption) == 1
    assert len(panel_images) == 4
    assert max(page4.index(block) for block in panel_images) < page4.index(figure1_caption[0])


def test_caibao_table_reclaims_repeated_dates_but_keeps_real_marginals() -> None:
    """验证完整财报第 9–16 页回收表格日期行，免责声明与页码仍独立保留。"""

    model_list = _auto_model_list("caibao1.pdf")

    assert len(model_list) == 22
    for page_number in range(9, 17):
        page = model_list[page_number - 1]
        tables_on_page = [block for block in page if block["type"] == "table"]
        footers = [block for block in page if block["type"] == "footer"]
        page_numbers = [block for block in page if block["type"] == "page_number"]
        assert len(tables_on_page) == len(footers) == len(page_numbers) == 1
        assert "2024年07月" in _normalized_content_probe(tables_on_page[0]["content"])
        assert _visible_content(footers[0]) == "免责声明和披露以及分析师声明是报告的一部分，请务必一起阅读。"
        assert "2024 年" not in _visible_content(footers[0])
        assert tables_on_page[0]["bbox"][3] < footers[0]["bbox"][1]
        assert tables_on_page[0]["bbox"][3] < page_numbers[0]["bbox"][1]
        assert not [block for block in page if block["type"] == "footnote"]

    page18 = model_list[17]
    assert len([block for block in page18 if block["type"] == "footer"]) == 1
    assert not [block for block in page18 if block["type"] == "footnote"]


def test_caibao_page2_parallel_chart_captions_stay_separate() -> None:
    """验证第二页左右图表的同行图注按各自横线和图形范围独立输出。"""

    model_list = _txt_model_list("caibao1.pdf")
    page = model_list[1]
    first_caption = [block for block in page if block["type"] == "caption" and _visible_content(block).startswith("图表1")]
    second_caption = [block for block in page if block["type"] == "caption" and _visible_content(block).startswith("图表2")]

    assert len(page) == 20
    assert len(first_caption) == len(second_caption) == 1
    assert first_caption[0]["type"] == second_caption[0]["type"] == "caption"
    assert first_caption[0]["bbox"] == [0.073, 0.235, 0.485, 0.245]
    assert second_caption[0]["bbox"] == [0.513, 0.235, 0.843, 0.245]
    assert "行业周涨幅" not in _visible_content(first_caption[0])
    assert "汽车指数上周下跌" not in _visible_content(second_caption[0])
    assert [block["bbox"] for block in page if block["type"] == "image" and block["bbox"][1] < 0.5] == [
        [0.082, 0.267, 0.48, 0.447],
        [0.522, 0.254, 0.92, 0.46],
    ]
    first_index = page.index(first_caption[0])
    second_index = page.index(second_caption[0])
    assert [block["type"] for block in page[first_index : first_index + 3]] == [
        "caption",
        "image",
        "footnote",
    ]
    assert [block["type"] for block in page[second_index : second_index + 3]] == [
        "caption",
        "image",
        "footnote",
    ]
    assert first_index + 3 == second_index

    page3_captions = [
        block
        for block in model_list[2]
        if block["type"] == "caption" and _visible_content(block).startswith(("图表4", "图表5"))
    ]
    assert [block["bbox"] for block in page3_captions] == [
        [0.073, 0.181, 0.298, 0.191],
        [0.514, 0.181, 0.738, 0.191],
    ]
    page3 = model_list[2]
    for caption in page3_captions:
        index = page3.index(caption)
        assert [block["type"] for block in page3[index : index + 3]] == [
            "caption",
            "image",
            "footnote",
        ]

    expected_new_footnotes = {1: 2, 6: 1, 7: 2, 8: 2, 17: 1, 19: 1}
    assert {
        page_number: sum(block["type"] == "footnote" for block in model_list[page_number - 1])
        for page_number in expected_new_footnotes
    } == expected_new_footnotes
    page7_table_note = _blocks_containing(
        model_list[6],
        "注：截止 2024 年 7 月 21 日",
    )
    assert len(page7_table_note) == 1
    assert page7_table_note[0]["type"] == "footnote"
    assert page7_table_note[0]["bbox"] == [0.073, 0.483, 0.219, 0.504]
    assert "资料来源：Wind、华泰研究" in _visible_content(page7_table_note[0])
    assert all(
        _visible_content(block).startswith("资料来源：")
        for page_number in expected_new_footnotes
        for block in model_list[page_number - 1]
        if block["type"] == "footnote" and block is not page7_table_note[0]
    )
    assert sum(block["type"] == "caption" for page_blocks in model_list for block in page_blocks) == 15
    assert sum(block["type"] == "footnote" for page_blocks in model_list for block in page_blocks) == 17

    page7_table_groups = _grouped_visual_blocks(model_list[6], "table")
    assert [[child["type"] for child in group["content"]] for group in page7_table_groups] == [
        ["table_caption", "table_body", "table_footnote"],
        ["table_caption", "table_body", "table_footnote"],
    ]
    assert [tuple(round(value * 1000) for value in child["bbox"]) for child in page7_table_groups[0]["content"]] == [
        (76, 83, 364, 94),
        (72, 94, 930, 480),
        (73, 483, 219, 504),
    ]

    page2_image_groups = _grouped_visual_blocks(page, "image")
    for caption_prefix in ("图表1", "图表2"):
        image_group = next(
            group
            for group in page2_image_groups
            if any(_visible_content(child).startswith(caption_prefix) for child in group["content"])
        )
        assert [child["type"] for child in image_group["content"]] == [
            "image_caption",
            "image_body",
            "image_footnote",
        ]


def test_iebm_left_indented_compact_formula_is_equation() -> None:
    """验证右栏左缩进紧凑公式通过公共 auto 入口输出为 equation。"""

    page = _auto_model_list("IEBM_A_2667169_O-5.pdf")[0]
    target = [block for block in page if block["bbox"] == [0.531, 0.291, 0.658, 0.311]]

    assert len(target) == 1
    assert target[0]["type"] == "equation"
    assert target[0]["content"] == "F ¼ @ψ @Y, G ¼ @ψ @X."
    assert not [block for block in page if block["type"] in {"caption", "footnote"}]


def test_synthetic_cjk_captions_and_urls_are_independent_annotations() -> None:
    """验证合成 CJK 图注稳定、跨行 URL 连续且多条独立 URL 不相连。"""

    model_list = _txt_model_list(
        _CJK_SYNTHETIC_PDF_NAME,
        pdf_dir=_FIXTURE_PDF_DIR,
    )
    page4 = model_list[3]
    captions = [block for block in page4 if block["type"] == "caption"]

    assert [inline_text(block["content"]) for block in captions] == [
        "图 1 合成示意图 A",
        "图 2 合成示意图 B",
    ]
    assert not [block for page in model_list for block in page if block["type"] == "footnote"]

    wrapped_url = "https://example.com/downloads/mineru/fixture/"
    wrapped_block = next(block.get("content") for block in page4 if wrapped_url in str(block.get("content", "")))
    assert inline_text(wrapped_block) == wrapped_url
    assert inline_urls(wrapped_block) == [wrapped_url]

    reference_urls = [
        "https://example.com/ref/a",
        "https://example.com/ref/b",
        "https://example.com/ref/c",
    ]
    reference_block = next(block.get("content") for block in page4 if reference_urls[0] in str(block.get("content", "")))
    reference_visible_text = inline_text(reference_block)
    assert reference_visible_text == " ".join(reference_urls)
    assert not any(first + second in reference_visible_text for first, second in zip(reference_urls, reference_urls[1:]))
    assert inline_urls(reference_block) == reference_urls


def test_synthetic_cjk_fixture_contains_only_safe_watermarks_and_links() -> None:
    """验证合成 CJK PDF 仅含安全旋转水印和预期 URI 注解。"""

    pdf_path = _FIXTURE_PDF_DIR / _CJK_SYNTHETIC_PDF_NAME
    model_list = _txt_model_list(
        _CJK_SYNTHETIC_PDF_NAME,
        pdf_dir=_FIXTURE_PDF_DIR,
    )
    with PDFDocument(pdf_path.read_bytes()) as document:
        rotated_lines = [
            "".join(str(span.get("text", "")) for span in line.get("spans", []))
            for page_index in range(len(document))
            for line in document.get_page_lines(page_index)
            if float(line.get("rotation", 0.0))
        ]

    expected_watermark_row = " ".join([_SAFE_WATERMARK_TEXT] * 4)
    assert len(rotated_lines) == 16
    assert {text.replace("\r", "").replace("\n", "").strip() for text in rotated_lines} == {expected_watermark_row}
    assert not [block for page in model_list for block in page if _SAFE_WATERMARK_TEXT in str(block.get("content", ""))]

    reader = PdfReader(pdf_path)
    link_targets = [str(annotation.get_object()["/A"]["/URI"]) for annotation in reader.pages[3]["/Annots"]]
    assert link_targets == [
        "https://example.com/downloads/mineru/fixture/",
        "https://example.com/downloads/mineru/fixture/",
        "https://example.com/ref/a",
        "https://example.com/ref/b",
        "https://example.com/ref/c",
    ]


def test_synthetic_pdf_fixtures_have_neutral_metadata_and_catalogs() -> None:
    """验证两份合成 PDF 均无加密、附件、脚本、隐藏元数据或无关批注。"""

    expected_metadata = {
        "/Author": "MinerU Test Suite",
        "/CreationDate": "D:20000101000000+00'00'",
        "/Creator": "MinerU Test Suite",
        "/Keywords": "MinerU synthetic test fixture",
        "/ModDate": "D:20000101000000+00'00'",
        "/Producer": "MinerU Test Suite",
        "/Subject": "Synthetic PDF regression fixture",
        "/Title": "MinerU Test Suite",
        "/Trapped": "/False",
    }
    expected_annotation_counts = {
        _FLASH_SYNTHETIC_PDF_NAME: [0] * 8,
        _CJK_SYNTHETIC_PDF_NAME: [0, 0, 0, 5],
    }
    for pdf_name, annotation_counts in expected_annotation_counts.items():
        reader = PdfReader(_FIXTURE_PDF_DIR / pdf_name)

        assert not reader.is_encrypted
        assert dict(reader.metadata or {}) == expected_metadata
        assert set(reader.trailer["/Root"]) == {"/PageMode", "/Pages", "/Type"}
        assert [len(page.get("/Annots", [])) for page in reader.pages] == annotation_counts


def test_frozen_soil_page3_formula3_remains_one_equation() -> None:
    """验证中文论文第三页公式编号完整转为 tag 并归入单个公式块。"""

    page = _txt_model_list("中文论文.pdf")[2]
    equations = [block for block in page if block["type"] == "equation"]
    formula3 = [block for block in equations if r"\tag{3}" in str(block.get("content", ""))]

    assert len(page) == 30
    assert len(equations) == 4
    assert len(formula3) == 1
    assert formula3[0]["bbox"] == [0.651, 0.717, 0.893, 0.746]
    assert all(probe in str(formula3[0]["content"]) for probe in ("at = 1", "2 ln", "1 - εt", "εt", r"\tag{3}"))
    assert not [
        block
        for block in page
        if block["type"] == "text" and any(probe in str(block.get("content", "")) for probe in ("at = 1", "2 ln", "1 - εt"))
    ]
    for marker in (r"\tag{1}", r"\tag{2}", r"\tag{3}", r"\tag{4}"):
        assert sum(marker in str(block.get("content", "")) for block in equations) == 1


def test_frozen_soil_reference_tails_remain_single_text_blocks() -> None:
    """验证中文论文双语图注独立标记，正文图引用和参考文献保持 text。"""

    model_list = _auto_model_list("中文论文.pdf")
    middle_pages = model_json_to_pages(_model_json(model_list))
    page2_continuation = next(
        block for block in middle_pages[1].blocks if block.type == "text" and inline_text(block.content).startswith("预测。")
    )
    assert page2_continuation.continues_prev is True
    page1_single_line_tail = next(
        block for block in middle_pages[0].blocks if block.type == "text" and inline_text(block.content) == "准的预测方法。"
    )
    assert page1_single_line_tail.continues_prev is True
    page = model_list[8]

    captions = [block for page_blocks in model_list for block in page_blocks if block["type"] == "caption"]
    assert len(captions) == 18
    page2 = model_list[1]
    image = next(block for block in page2 if block["type"] == "image")
    chinese_caption = _blocks_containing(page2, "图1 冻土抗剪强度试验流程示意图")
    english_caption = _blocks_containing(page2, "Fig. 1 Schematic diagram")
    assert len(chinese_caption) == len(english_caption) == 1
    assert chinese_caption[0]["type"] == english_caption[0]["type"] == "caption"
    assert page2.index(image) < page2.index(chinese_caption[0]) < page2.index(english_caption[0])

    page3 = model_list[2]
    page3_table_group = _grouped_visual_blocks(page3, "table")[0]
    assert [child["type"] for child in page3_table_group["content"]] == [
        "table_caption",
        "table_caption",
        "table_body",
        "table_footnote",
    ]
    assert [tuple(round(value * 1000) for value in child["bbox"]) for child in page3_table_group["content"]] == [
        (182, 688, 406, 700),
        (114, 706, 474, 715),
        (107, 721, 483, 838),
        (105, 837, 484, 926),
    ]
    assert inline_text(page3_table_group["content"][0]["content"]).startswith("表1")
    assert inline_text(page3_table_group["content"][1]["content"]).startswith("Table 1")

    page5_table_groups = _grouped_visual_blocks(model_list[4], "table")
    assert [[child["type"] for child in group["content"]] for group in page5_table_groups] == [
        ["table_caption", "table_caption", "table_body"],
        ["table_caption", "table_caption", "table_body"],
    ]
    assert [
        [tuple(round(value * 1000) for value in child["bbox"]) for child in group["content"]] for group in page5_table_groups
        ] == [
            [
                (215, 124, 374, 137),
                (169, 143, 421, 154),
                (107, 158, 483, 416),
            ],
            [
                (208, 628, 381, 640),
                (127, 646, 462, 656),
                (107, 662, 483, 826),
            ],
    ]

    narrative_reference = _blocks_containing(
        model_list[7],
        "图 8（a）、8（b）分别为",
    )
    assert len(narrative_reference) == 1
    assert narrative_reference[0]["type"] == "text"

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
    assert len(model_list) == 4
    page7, page8, page9, page10 = model_list

    author_blocks = [
        block for block in page7 if block["type"] == "text" and 0.15 <= block["bbox"][1] and block["bbox"][3] <= 0.28
    ]
    assert len(author_blocks) == 4
    assert {
        name
        for name in ("Xing Wang", "Yingzhou Zhang", "Lian Zhao", "Xinghao Chen")
        if any(name in _visible_content(block) for block in author_blocks)
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
    assert figure_caption[0]["type"] == "caption"
    algorithm1 = _blocks_containing(page8, "Algorithm 1 Detection of Irrelevant Code")
    algorithm2 = _blocks_containing(page8, "Algorithm 2 Detection of Unreachable Code")
    assert len(algorithm1) == len(algorithm2) == 1
    assert algorithm1[0]["type"] == algorithm2[0]["type"] == "caption"
    assert page8.index(algorithm1[0]) + 1 == page8.index(code_blocks[1])
    assert page8.index(algorithm2[0]) + 1 == page8.index(code_blocks[2])
    assert sum(block["type"] == "caption" for page in model_list for block in page) == 6

    page10_table_groups = _grouped_visual_blocks(page10, "table")
    assert [[child["type"] for child in group["content"]] for group in page10_table_groups] == [
        ["table_caption", "table_body"],
        ["table_caption", "table_body"],
    ]
    assert [
        [tuple(round(value * 1000) for value in child["bbox"]) for child in group["content"]] for group in page10_table_groups
    ] == [
        [(151, 333, 432, 344), (112, 352, 470, 496)],
        [(193, 607, 389, 618), (121, 626, 461, 710)],
    ]
    assert all("TABLE" not in str(group["content"][-1]["content"]) for group in page10_table_groups)

    algorithm_groups = [
        group
        for group in _grouped_visual_blocks(page8, "code")
        if any(_visible_content(child).startswith("Algorithm") for child in group["content"])
    ]
    assert len(algorithm_groups) == 2
    assert all([child["type"] for child in group["content"]] == ["code_caption", "code_body"] for group in algorithm_groups)

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
    assert "Ecabet sodium prevents esophageal lesions" in _visible_content(document_titles[0])
    assert "reflux of gastric juice in rats" in _visible_content(document_titles[0])
    for probe in (
        "2. Effects on esophageal lesions",
        "3. Effect on digestion of mucus",
    ):
        matches = _blocks_containing(all_blocks, probe)
        assert len(matches) == 1
        assert matches[0]["type"] == "paragraph_title"

    assert {_visible_content(block) for block in all_blocks if block["type"] == "page_number"} == {"91", "92", "93", "94"}
    assert not [block for block in all_blocks if block["type"] == "image" and str(block["content"]).strip() in {"91", "92"}]

    figure2 = _blocks_containing(model_list[2], "Fig. 2. Pathological changes")
    figure6 = _blocks_containing(model_list[3], "Fig. 6. Effect of ecabet")
    assert len(figure2) == len(figure6) == 1
    assert figure2[0]["type"] == figure6[0]["type"] == "caption"
    assert "Under anesthesia with ether" in _visible_content(figure2[0])
    assert "pepsin. Isolated and everted esophagus" in _visible_content(figure6[0])
    assert not [block for block in all_blocks if block["type"] == "footnote"]
    assert sum(block["type"] == "caption" for block in all_blocks) == 7
    page1_corresponding_author = _blocks_containing(
        model_list[0],
        "Corresponding author",
    )
    page1_right_column_tail = _blocks_containing(
        model_list[0],
        "The reflux of gastric juice was induced",
    )
    assert len(page1_corresponding_author) == len(page1_right_column_tail) == 1
    assert page1_corresponding_author[0]["type"] == "page_footnote"
    assert page1_right_column_tail[0]["type"] == "text"
    assert not [block for block in model_list[0] if block["type"] == "footer"]

    table_group = _grouped_visual_blocks(model_list[3], "table")[0]
    assert [child["type"] for child in table_group["content"]] == [
        "table_caption",
        "table_body",
    ]
    assert [tuple(round(value * 1000) for value in child["bbox"]) for child in table_group["content"]] == [
        (77, 92, 799, 104),
        (79, 109, 922, 247),
    ]
    assert "Table 1." not in str(table_group["content"][1]["content"])

    online_footer = _blocks_containing(model_list[4], "http://www.birkhauser.ch/IPh")
    assert len(online_footer) == 1
    assert online_footer[0]["type"] == "footer"


def test_mixed_elements_pages_39_40_force_txt_regressions() -> None:
    """验证原文 39–40 页三个公式和公式后同视觉行正文的唯一输出。"""

    model_list = _txt_model_list("mixed_elements_pages_39_40.pdf")
    page40 = model_list[1]
    equations = [block for block in page40 if block["type"] == "equation"]

    assert len(equations) == 3
    assert sum(r"\tag{8}" in str(block["content"]) for block in equations) == 1
    assert sum(r"\tag{9}" in str(block["content"]) for block in equations) == 1
    inequality = _blocks_containing(equations, "Mπ(Xn)")
    assert len(inequality) == 1
    assert inequality[0]["type"] == "equation"

    continuation = _blocks_containing(page40, "so that we have a near-unbiased estimator")
    assert len(continuation) == 1
    assert continuation[0]["type"] == "text"
    assert "Coupling this observation" in _visible_content(continuation[0])
    assert not [block for page in model_list for block in page if block["type"] in {"caption", "footnote"}]
    assert sum("Coupling this observation" in _visible_content(block) for block in page40) == 1
    equation9_following = _blocks_containing(
        page40,
        "Clearly, the random variable",
    )
    assert len(equation9_following) == 1
    assert equation9_following[0]["type"] == "text"
    assert "is an unbiased estimator" in _visible_content(equation9_following[0])
    assert "McAllester and Schapire" in _visible_content(equation9_following[0])
