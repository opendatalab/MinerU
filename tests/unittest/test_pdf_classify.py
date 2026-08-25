from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    NumberObject,
)

from mineru.utils import pdf_classify
from mineru.utils.pdf_document import PDFDocument

REPO_ROOT = Path(__file__).resolve().parents[2]
MIXED_ELEMENTS_PDF = REPO_ROOT / "demo" / "pdfs" / "mixed_elements_pages_07_10.pdf"
SYNTHETIC_FLASH_PDF = REPO_ROOT / "tests" / "unittest" / "pdfs" / "flash_table_annotations_synthetic.pdf"


def _write_single_sample_page(content_data: bytes) -> bytes:
    """复制真实样本第二页，并用指定内容流替换页面内容。"""
    reader = PdfReader(MIXED_ELEMENTS_PDF)
    writer = PdfWriter()
    page = writer.add_page(reader.pages[1])
    content = DecodedStreamObject()
    content.set_data(content_data)
    page[NameObject("/Contents")] = writer._add_object(content)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _get_exact_cid_usage(pdf_bytes: bytes) -> dict[int, dict[str, Any]]:
    """返回全部页面按字体资源对象精确统计的坏 CID 用量。"""
    reader = PdfReader(BytesIO(pdf_bytes))
    signals = pdf_classify._get_font_resource_signals_pypdf(
        pdf_bytes,
        list(range(len(reader.pages))),
    )
    return signals["cid_without_to_unicode_usage"]


def _make_form_xobject(
    writer: PdfWriter,
    content_data: bytes,
    resources: DictionaryObject,
) -> tuple[Any, DecodedStreamObject]:
    """构造带局部资源的最小 Form XObject 并加入 writer。"""
    form = DecodedStreamObject()
    form.set_data(content_data)
    form.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Form"),
            NameObject("/FormType"): NumberObject(1),
            NameObject("/BBox"): ArrayObject(
                [
                    FloatObject(0),
                    FloatObject(0),
                    FloatObject(100),
                    FloatObject(100),
                ]
            ),
            NameObject("/Resources"): resources,
        }
    )
    return writer._add_object(form), form


def _build_nested_form_pdf() -> bytes:
    """构造两次调用嵌套 Form 的 PDF，用于验证递归计数按调用次数累加。"""
    reader = PdfReader(MIXED_ELEMENTS_PDF)
    writer = PdfWriter()
    page = writer.add_page(reader.pages[1])
    page_resources = page["/Resources"].get_object()
    page_fonts = page_resources["/Font"].get_object()
    bad_font_ref = page_fonts["/C2_0"]

    inner_resources = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/C2_0"): bad_font_ref}
            )
        }
    )
    inner_ref, _inner = _make_form_xobject(
        writer,
        b"BT /C2_0 12 Tf <01380138> Tj ET",
        inner_resources,
    )
    outer_resources = DictionaryObject(
        {
            NameObject("/XObject"): DictionaryObject(
                {NameObject("/Inner"): inner_ref}
            )
        }
    )
    outer_ref, _outer = _make_form_xobject(
        writer,
        b"/Inner Do",
        outer_resources,
    )
    page_resources[NameObject("/XObject")] = DictionaryObject(
        {NameObject("/Outer"): outer_ref}
    )
    page_content = DecodedStreamObject()
    page_content.set_data(b"/Outer Do /Outer Do")
    page[NameObject("/Contents")] = writer._add_object(page_content)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _build_cyclic_form_pdf() -> bytes:
    """构造自引用 Form，用于验证循环内容流会进入保守失败路径。"""
    reader = PdfReader(MIXED_ELEMENTS_PDF)
    writer = PdfWriter()
    page = writer.add_page(reader.pages[1])
    page_resources = page["/Resources"].get_object()
    page_fonts = page_resources["/Font"].get_object()
    bad_font_ref = page_fonts["/C2_0"]

    loop_resources = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/C2_0"): bad_font_ref}
            ),
            NameObject("/XObject"): DictionaryObject(),
        }
    )
    loop_ref, loop_form = _make_form_xobject(
        writer,
        b"BT /C2_0 12 Tf <0138> Tj ET /Self Do",
        loop_resources,
    )
    loop_resources["/XObject"].get_object()[NameObject("/Self")] = loop_ref
    loop_form[NameObject("/Resources")] = loop_resources
    page_resources[NameObject("/XObject")] = DictionaryObject(
        {NameObject("/Loop"): loop_ref}
    )
    page_content = DecodedStreamObject()
    page_content.set_data(b"/Loop Do")
    page[NameObject("/Contents")] = writer._add_object(page_content)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_mixed_elements_uses_exact_cid_counts_and_classifies_as_txt() -> None:
    """验证真实样本的小规模坏 CID 字形不再被同名正文放大成 OCR。"""
    pdf_bytes = MIXED_ELEMENTS_PDF.read_bytes()
    usage = _get_exact_cid_usage(pdf_bytes)

    assert [usage[index]["cid_font_char_count"] for index in range(4)] == [
        5,
        11,
        4,
        5,
    ]
    assert usage[1]["font_names"] == [
        "MSUIGothic-90ms-RKSJ-H-Identity-H",
        "SimSun-GBK-EUC-H-Identity-H",
        "TimesNewRoman",
    ]
    with PDFDocument(pdf_bytes) as pdf_doc:
        assert pdf_doc.classify() == "txt"


def test_dominant_bad_cid_font_still_classifies_as_ocr_with_same_name_resource() -> None:
    """验证同名正常字体存在时，大量实际使用的坏 CID 字体仍触发 OCR。"""
    pdf_bytes = _write_single_sample_page(
        b"BT /C2_0 12 Tf 72 720 Td <" + b"0138" * 64 + b"> Tj ET"
    )
    usage = _get_exact_cid_usage(pdf_bytes)

    assert usage[0] == {
        "font_names": ["TimesNewRoman"],
        "cid_font_char_count": 64,
    }
    with PDFDocument(pdf_bytes) as pdf_doc:
        assert pdf_doc.classify() == "ocr"


def test_exact_cid_count_tracks_text_operators_and_graphics_state() -> None:
    """验证四类文本操作符及 q/Q 字体恢复均按具体资源统计。"""
    pdf_bytes = _write_single_sample_page(
        b"q "
        b"BT /C2_0 12 Tf 72 720 Td <0138> Tj "
        b"[<0138> 10 <0138>] TJ <0138> ' 0 0 <0138> \" ET "
        b"q BT /TT2 12 Tf 72 700 Td (normal text) Tj ET Q "
        b"BT 72 680 Td <0138> Tj ET Q"
    )

    assert _get_exact_cid_usage(pdf_bytes)[0] == {
        "font_names": ["TimesNewRoman"],
        "cid_font_char_count": 6,
    }


def test_exact_cid_count_recurses_nested_forms_per_invocation() -> None:
    """验证嵌套 Form 的局部字体资源按实际两次调用累计。"""
    assert _get_exact_cid_usage(_build_nested_form_pdf())[0] == {
        "font_names": ["TimesNewRoman"],
        "cid_font_char_count": 4,
    }


def test_unused_bad_cid_resource_is_not_counted() -> None:
    """验证页面资源中存在但内容流未选择的坏 CID 字体用量为零。"""
    pdf_bytes = _write_single_sample_page(
        b"BT /TT2 12 Tf 72 720 Td (normal TimesNewRoman text) Tj ET"
    )

    assert _get_exact_cid_usage(pdf_bytes)[0] == {
        "font_names": [],
        "cid_font_char_count": 0,
    }


@pytest.mark.parametrize(
    ("content_data", "error_pattern"),
    [
        (b"BT /C2_0 12 Tf <013801> Tj ET", "odd byte length"),
        (b"BT /Missing 12 Tf <0138> Tj ET", "font resource /Missing"),
    ],
)
def test_malformed_cid_content_raises_for_conservative_classification(
    content_data: bytes,
    error_pattern: str,
) -> None:
    """验证无法精确归因的坏 CID 内容不会静默退回字体名估算。"""
    with pytest.raises(ValueError, match=error_pattern):
        _get_exact_cid_usage(_write_single_sample_page(content_data))


def test_cyclic_form_raises_for_conservative_classification() -> None:
    """验证自引用 Form 在精确计数阶段被识别为循环。"""
    with pytest.raises(ValueError, match="Cyclic PDF Form XObject"):
        _get_exact_cid_usage(_build_cyclic_form_pdf())


def test_missing_raw_text_bytes_are_rejected() -> None:
    """验证坏 CID 字符串缺少原始字节时不会使用已解码文本猜测。"""
    with pytest.raises(ValueError, match="original bytes"):
        pdf_classify._get_pdf_string_raw_bytes("decoded text")


def test_classifier_returns_ocr_when_exact_content_analysis_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证内容流精确分析异常沿用分类器的保守 OCR 兜底。"""
    pdf_bytes = MIXED_ELEMENTS_PDF.read_bytes()

    def fail_font_analysis(_pdf_bytes: bytes, _page_indices: list[int]) -> dict[str, Any]:
        """模拟内容流无法解析。"""
        raise ValueError("broken content stream")

    monkeypatch.setattr(
        pdf_classify,
        "_get_font_resource_signals_pypdf",
        fail_font_analysis,
    )
    with PDFDocument(pdf_bytes) as pdf_doc:
        assert pdf_classify.classify(pdf_doc._pdf_doc, pdf_bytes) == "ocr"


@pytest.mark.parametrize(
    ("pdf_name", "expected_mode"),
    [
        ("demo1.pdf", "txt"),
        ("demo2.pdf", "txt"),
        ("demo3.pdf", "txt"),
        ("demo4.pdf", "txt"),
        ("demo6.pdf", "txt"),
        ("mixed_elements_pages_03_06.pdf", "txt"),
        ("mixed_elements_pages_07_10.pdf", "txt"),
        ("mixed_elements_pages_11_15.pdf", "txt"),
        ("mixed_elements_pages_39_40.pdf", "txt"),
        ("small_ocr.pdf", "ocr"),
    ],
)
def test_demo_pdf_classification_regressions(
    pdf_name: str,
    expected_mode: str,
) -> None:
    """验证仓库标准 PDF 分类仅修正目标样本且其余结果保持稳定。"""
    pdf_path = REPO_ROOT / "demo" / "pdfs" / pdf_name
    with PDFDocument(str(pdf_path)) as pdf_doc:
        assert pdf_doc.classify() == expected_mode


def test_synthetic_flash_fixture_classification_regression() -> None:
    """验证脱敏合成表格夹具继续稳定分类为 TXT。"""

    with PDFDocument(str(SYNTHETIC_FLASH_PDF)) as pdf_doc:
        assert pdf_doc.classify() == "txt"
