"""用真实 PDF 验证整行仿斜体回填，并锁定既有旋转水印的准入行为。"""

from __future__ import annotations

import math
from copy import deepcopy
from io import BytesIO
from typing import Literal
from unittest.mock import MagicMock

import pytest
from docvortex.content.spans import inline_span_plain_text
from docvortex.document.pdf import Char, PDFDocument, PDFPageTextGeometry
from docvortex.postprocess.document import model_json_to_middle_json
from docvortex.schema import DocumentMetadata, ModelJson, Producer
from reportlab.pdfgen.canvas import Canvas

from mineru.backend.analysis.pdf.constants import VLM_TXT_DET_TYPE
from mineru.backend.analysis.pdf.normalization import _normalize_pdf_model_list
from mineru.backend.analysis.pdf.snapshots import create_page_snapshot_cache
from mineru.backend.analysis.pdf.text import content, native
from mineru.parser.base import ParseResult

_URL = "https://www.georisques.gouv.fr/risques/installations/donnees?page=1"
_PREFIX = "PREFIX: "
_HEADING = "Texts available on the GEORISQUES website:"
_FOOTER = "The establishment is subject to prefectoral order 2008-226-18."


def _pdf(
    text: str = _URL,
    *,
    angle: float = 7,
    rotate: bool = False,
    prefix: str = "",
    italic_font: bool = False,
    brochure: bool = False,
) -> bytes:
    """生成剪切与旋转可独立控制的文字，三行模式复现 issue #5731 的页面结构。"""
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(595, 842))
    canvas.setFont("Helvetica", 11)
    if brochure:
        canvas.drawString(50, 760, _HEADING)
    canvas.saveState()
    canvas.translate(50, 740 if brochure else 400)
    if prefix:
        canvas.drawString(0, 0, prefix)
        canvas.translate(canvas.stringWidth(prefix, "Helvetica", 11), 0)
    if italic_font:
        canvas.setFont("Helvetica-Oblique", 11)
    elif rotate:
        canvas.rotate(angle)
    else:
        canvas.skew(0, angle)
    canvas.drawString(0, 0, text)
    canvas.restoreState()
    if brochure:
        canvas.drawString(50, 720, _FOOTER)
    canvas.save()
    return stream.getvalue()


def _geometry(data: bytes) -> PDFPageTextGeometry:
    """通过公开接口物化真实字符，返回后即可关闭 PDFium 文档。"""
    with PDFDocument(data) as document:
        return document.get_page_chars_with_geometry(0)


def _text(chars: list[Char]) -> str:
    """保留原始内部空格，只忽略 PDFium 的行分隔符。"""
    return "".join(str(char["char"]) for char in chars).replace("\r", "").replace("\n", "")


@pytest.mark.parametrize("angle", [-19, -7, 7, 19])
@pytest.mark.parametrize("text", [_URL, "Synthetic italic words with spaces"])
@pytest.mark.parametrize("prefix", ["", _PREFIX])
def test_sheared_pdf_keeps_native_characters(angle: float, text: str, prefix: str) -> None:
    """正反剪切的独立行与混合行均完整回填，字符对象、几何和顺序不变。"""
    geometry = _geometry(_pdf(text, angle=angle, prefix=prefix))
    before = deepcopy([{**char, "bbox": tuple(char["bbox"])} for char in geometry.chars])
    before_origins = dict(geometry.origins)
    kept = native._get_chars_for_span_fill(geometry.chars, origins=geometry.origins)
    assert _text(kept) == prefix + text
    assert all(left is right for left, right in zip(kept, geometry.chars, strict=True))
    assert [{**char, "bbox": tuple(char["bbox"])} for char in geometry.chars] == before
    assert geometry.origins == before_origins


def test_true_italic_font_and_dict_input_keep_existing_behavior() -> None:
    """真正斜体字体仍走普通角度路径，旧字典输入也可补回整行仿斜体。"""
    for italic_font in (False, True):
        geometry = _geometry(_pdf(italic_font=italic_font))
        assert _text(native._get_chars_for_span_fill({"chars": geometry.chars})) == _URL


@pytest.mark.parametrize("angle", [-19, -7, 7, 19, 45])
@pytest.mark.parametrize("prefix", ["", _PREFIX])
def test_rotated_pdf_matches_recorded_filter_baseline(angle: float, prefix: str) -> None:
    """锁定修复前真实 PDF 的结果，包括旧规则已放行的混合小角度旋转文字。"""
    geometry = _geometry(_pdf(angle=angle, rotate=True, prefix=prefix))
    kept = native._get_chars_for_span_fill(geometry.chars, origins=geometry.origins)
    # 修改前记录：独立斜行全部过滤；有前缀的粗行仅放行小于 30 度的局部旋转。
    expected = prefix + (_URL if prefix and abs(angle) < 30 else "")
    assert _text(kept) == expected
    expected_chars = geometry.chars if expected.endswith(_URL) else geometry.chars[: len(prefix)]
    assert [char["char_idx"] for char in kept] == [char["char_idx"] for char in expected_chars]


@pytest.mark.parametrize(
    "case",
    [
        "missing_origin",
        "partially_missing_origin",
        "nonfinite_origin",
        "missing_angle",
        "nonfinite_angle",
        "inconsistent_angle",
        "repeated_origin",
        "two_origins_repeated",
        "tiny_advances",
        "two_origins",
        "different_objects",
        "interrupted_object",
        "missing_object",
        "superscript",
        "sloped_origins",
        "only_spaces",
        "only_generated",
        "only_two_real_chars",
        "invisible",
        "excess_shear",
    ],
)
def test_insufficient_baseline_evidence_preserves_old_rejection(case: str) -> None:
    """从真实剪切字符逐项移除证据，不能用默认方向、重复点或生成字符放行。"""
    chars = _geometry(_pdf("ABCD")).chars
    if case == "two_origins":
        chars = chars[:2]
    for index, char in enumerate(chars):
        if case == "missing_origin":
            char.pop("origin", None)
        elif case == "partially_missing_origin" and index == 1:
            char.pop("origin", None)
        elif case == "nonfinite_origin":
            char["origin"] = (float("nan"), 0)
        elif case == "missing_angle":
            char.pop("writing_angle", None)
        elif case == "nonfinite_angle":
            char["writing_angle"] = float("inf")
        elif case == "inconsistent_angle" and index == 1:
            char["writing_angle"] = math.radians(7)
        elif case == "repeated_origin":
            char["origin"] = (50, 442)
        elif case == "two_origins_repeated":
            char["origin"] = (50 + (index % 2) * 10, 442)
        elif case == "tiny_advances":
            char["origin"] = (50 + index * 0.0002, 442)
        elif case == "different_objects":
            char["text_object_id"] = index
        elif case == "interrupted_object":
            char["text_object_id"] = 1 if index == 1 else 0
        elif case == "missing_object":
            char.pop("text_object_id", None)
        elif case == "superscript" and index == 1:
            char["origin"] = (char["origin"][0], char["origin"][1] - 3)
        elif case == "sloped_origins":
            char["origin"] = (50 + index * 10, 442 + index)
        elif case == "only_spaces":
            char["char"] = " "
        elif case == "only_generated" or (case == "only_two_real_chars" and index >= 2):
            char["is_generated"] = True
        elif case == "invisible":
            char["text_is_visible"] = False
        elif case == "excess_shear":
            char["rotation"] = math.radians(31)
    assert native._get_chars_for_span_fill(chars) == []


@pytest.mark.parametrize("separator", ["\r", "\n", "\r\n"])
def test_line_breaks_cannot_join_short_runs_into_baseline_evidence(separator: str) -> None:
    """相同文本对象的两段短文字不能跨换行借用原点证据，换行本身不补回。"""
    chars = _geometry(_pdf("ABCDE")).chars
    chars[2]["char"] = separator
    assert native._get_chars_for_span_fill(chars) == []


def test_invisible_characters_are_not_restored_with_visible_run() -> None:
    """有效基线仅补回可见成员，不把同对象中的明确不可见字符顺带放行。"""
    chars = _geometry(_pdf("ABCDE")).chars
    chars[2]["text_is_visible"] = False
    chars[2].pop("origin", None)
    assert _text(native._get_chars_for_span_fill(chars)) == "ABDE"


def test_origin_side_map_has_priority_and_missing_entries_use_embedded_origin() -> None:
    """side map 覆盖字符内原点，只有缺失条目才允许回退，坏数据不能用另一份补证。"""
    geometry = _geometry(_pdf("ABCD"))
    assert _text(native._get_chars_for_span_fill(geometry.chars, origins={0: geometry.origins[0]})) == "ABCD"
    invalid = {0: (float("nan"), 0)}
    assert native._get_chars_for_span_fill(geometry.chars, origins=invalid) == []
    for char in geometry.chars:
        char.pop("origin", None)
    assert _text(native._get_chars_for_span_fill(geometry.chars, origins=geometry.origins)) == "ABCD"


@pytest.mark.parametrize("axis", [0, 90, 180, 270])
@pytest.mark.parametrize("deviation", [-0.101, -0.099, 0, 0.099, 0.101])
def test_baseline_direction_uses_circular_tolerance(axis: int, deviation: float) -> None:
    """四个标准方向及跨零角度使用同一严格容差，真实超限倾斜不能补回。"""
    chars = _geometry(_pdf("ABCD")).chars
    angle = math.radians(axis + deviation)
    for index, char in enumerate(chars):
        char["writing_angle"] = angle - math.tau
        char["origin"] = (50 + index * 10 * math.cos(angle), 442 + index * 10 * math.sin(angle))
    kept = native._get_chars_for_span_fill(chars)
    assert _text(kept) == ("ABCD" if abs(deviation) < 0.1 else "")


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("effort", ["medium", "high", "xhigh"])
def test_sheared_url_window_uses_native_text_without_post_ocr(
    owned: bool, effort: Literal["medium", "high", "xhigh"], monkeypatch: pytest.MonkeyPatch
) -> None:
    """固定检测框重放三个 TXT 档位，验证快照及普通路径直至 Markdown 都不调用 OCR。"""
    boxes = [(48, 71, 440, 87), (48, 91, 440, 107), (48, 111, 440, 127)]
    layout = [[{"type": "text", "bbox": [x0 / 595, y0 / 842, x1 / 595, y1 / 842]} for x0, y0, x1, y1 in boxes]]
    detection = [[{"label": "ocr_text", "bbox": [value * 2 for value in box], "score": 1.0, "text": ""} for box in boxes]]
    context = MagicMock()
    context.ocr_model.ocr.side_effect = AssertionError("synthetic italic unexpectedly reached post-OCR")
    cropper = MagicMock(wraps=native.get_crop_img)
    monkeypatch.setattr(native, "get_crop_img", cropper)
    with PDFDocument(_pdf(brochure=True)) as document:
        page = document[0]
        if owned:
            owner = page.get_text_snapshot()
            assert owner is not None
            assert not owner.supports_span_matching()
        rendered = document.render_page(0, scale=2)
        try:
            pages = content._fill_window_block_content_and_lines(
                [{"img_pil": rendered.pil_image, "scale": 2}],
                [page],
                layout,
                [[]],
                detection,
                "txt",
                effort,
                VLM_TXT_DET_TYPE,
                context,
                page_snapshots=create_page_snapshot_cache([page], enabled=owned),
            )
            _normalize_pdf_model_list(pages)
        finally:
            rendered.pil_image.close()
    cropper.assert_not_called()
    context.ocr_model.ocr.assert_not_called()
    model = ModelJson(
        metadata=DocumentMetadata(file_suffix="pdf", producer=Producer(name="mineru", version="test")),
        pages=pages,
        page_index_map=[],
    )
    assert [inline_span_plain_text(block["content"]) for block in pages[0]] == [_HEADING, _URL, _FOOTER]
    markdown = ParseResult(middle_json=model_json_to_middle_json(model)).markdown()
    assert markdown.count(_URL) == 1
    assert _HEADING in markdown and _FOOTER in markdown
