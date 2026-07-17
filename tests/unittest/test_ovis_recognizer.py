# Copyright (c) Opendatalab. All rights reserved.
"""OvisOCR2适配器测试：后处理纯函数 + mock模型调用的全链路，均无需网络/GPU。"""
from pathlib import Path

from mineru_vl_utils.structs import ContentBlock
from PIL import Image

from mineru.backend.vlm.ovis_ocr_recognizer import (
    OvisOcrContentRecognizer,
    _extract_table_html,
    _flatten_markdown,
    _strip_code_fences,
    _strip_math_wrappers,
    postprocess_block_content,
)

SMALL_OCR_PDF = Path(__file__).parent.parent.parent / "demo" / "pdfs" / "small_ocr.pdf"


# ---------------------------------------------------------------------------
# 后处理纯函数
# ---------------------------------------------------------------------------

def test_strip_math_wrappers():
    assert _strip_math_wrappers("$$E=mc^2$$") == "E=mc^2"
    assert _strip_math_wrappers("\\[ \\frac{a}{b} \\]") == "\\frac{a}{b}"
    assert _strip_math_wrappers("$x+y$") == "x+y"
    assert _strip_math_wrappers("E=mc^2") == "E=mc^2"


def test_extract_table_html_passthrough_and_fallback():
    html = "<table><tr><td>a</td></tr></table>"
    assert _extract_table_html(f"some text\n{html}\nmore") == html
    markdown = "| A | B |\n|---|---|\n| 1 | 2 |"
    converted = _extract_table_html(markdown)
    assert converted.startswith("<table>") and "<td>1</td>" in converted


def test_strip_code_fences():
    assert _strip_code_fences("```python\nprint(1)\n```") == "print(1)"
    assert _strip_code_fences("plain") == "plain"


def test_flatten_markdown_cjk_join():
    text = "## 标题\n这是第一行，\n这是第二行。\n\nEnglish line\nsecond line"
    flattened = _flatten_markdown(text)
    assert "这是第一行，这是第二行。" in flattened  # 中文行间不加空格
    assert "English line second line" in flattened  # 英文行间加空格
    assert "#" not in flattened


def test_flatten_markdown_removes_img_placeholder():
    result = _flatten_markdown('前文\n<img src="images/bbox_1_2_3_4.jpg" />\n后文')
    assert "img" not in result
    assert "前文" in result and "后文" in result


def test_postprocess_by_type():
    assert postprocess_block_content("equation", "$$a+b$$") == "a+b"
    assert postprocess_block_content("table", "<table><tr><td>x</td></tr></table>").startswith("<table>")
    assert postprocess_block_content("code", "```\nx=1\n```") == "x=1"
    assert postprocess_block_content("text", "**bold** text") == "bold text"


# ---------------------------------------------------------------------------
# 图像准备
# ---------------------------------------------------------------------------

def test_prepare_block_image_rotation_and_padding():
    recognizer = OvisOcrContentRecognizer(server_url="http://fake:8000", min_edge=100)
    page = Image.new("RGB", (1000, 800), (200, 200, 200))
    block = ContentBlock("table", [0.1, 0.1, 0.5, 0.3], angle=270)
    crop = recognizer._prepare_block_image(page, block)
    # 原裁块 400x160，270度旋转后宽高互换为 160x400，再补白到 min_edge
    assert crop.height == 400
    assert crop.width == max(160, 100)

    small_block = ContentBlock("text", [0.0, 0.0, 0.05, 0.05], angle=0)
    padded = recognizer._prepare_block_image(page, small_block)
    assert padded.width >= 100 and padded.height >= 100


def test_server_url_normalization():
    assert OvisOcrContentRecognizer("http://h:8000").base_url == "http://h:8000/v1"
    assert OvisOcrContentRecognizer("http://h:8000/").base_url == "http://h:8000/v1"
    assert OvisOcrContentRecognizer("http://h:8000/v1").base_url == "http://h:8000/v1"


# ---------------------------------------------------------------------------
# mock模型调用的批量识别
# ---------------------------------------------------------------------------

def test_batch_recognize_fills_content_and_skips():
    recognizer = OvisOcrContentRecognizer(server_url="http://fake:8000", max_concurrency=2)
    answers = {
        "text": "识别的正文",
        "equation": "$$E=mc^2$$",
        "table": "<table><tr><td>1</td></tr></table>",
    }
    recognizer._complete = None  # 确保不会走真实HTTP

    def fake_recognize_one(page_image, block):
        block["content"] = postprocess_block_content(block["type"], answers[block["type"]])

    recognizer._recognize_one = fake_recognize_one

    page = Image.new("RGB", (800, 1000), (255, 255, 255))
    blocks = [
        ContentBlock("text", [0.1, 0.1, 0.9, 0.2]),
        ContentBlock("equation", [0.1, 0.3, 0.9, 0.4]),
        ContentBlock("table", [0.1, 0.5, 0.9, 0.6]),
        ContentBlock("image", [0.1, 0.7, 0.9, 0.8]),          # skip类型
        ContentBlock("text", [0.1, 0.85, 0.9, 0.9], content="已有内容"),  # 已带内容
    ]
    results = recognizer.batch_recognize([page], [blocks])

    assert results[0][0]["content"] == "识别的正文"
    assert results[0][1]["content"] == "E=mc^2"
    assert results[0][2]["content"].startswith("<table>")
    assert results[0][3]["content"] is None       # image 不识别
    assert results[0][4]["content"] == "已有内容"  # 不覆盖已有内容


def test_full_chain_with_fake_layout(tmp_path):
    """FakeLayout + Ovis适配器(mock调用) 走完整 doc_analyze 解耦链路。"""
    from mineru.backend.vlm import vlm_analyze
    from mineru.backend.vlm.stages import LayoutDetector

    class OnePerPageLayout(LayoutDetector):
        name = "fake"

        def batch_detect(self, images, start_page_idx=0):
            return [
                [ContentBlock("text", [0.1, 0.1, 0.9, 0.3])]
                for _ in images
            ]

    recognizer = OvisOcrContentRecognizer(server_url="http://fake:8000")
    recognizer._complete = lambda img: "## 标题\nOvis识别内容"

    middle_json, results = vlm_analyze.doc_analyze(
        SMALL_OCR_PDF.read_bytes(),
        None,
        layout_detector=OnePerPageLayout(),
        content_recognizer=recognizer,
    )
    assert len(middle_json["pdf_info"]) == 8
    for page_results in results:
        assert page_results[0]["content"] == "标题 Ovis识别内容"
