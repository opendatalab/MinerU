# Copyright (c) Opendatalab. All rights reserved.
import os
import re
from io import BytesIO

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from loguru import logger
from pypdf import PdfReader
from pdfminer.converter import PDFPageAggregator
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams, LTFigure, LTImage
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from mineru.utils.pdfium_guard import (
    close_pdfium_document,
    open_pdfium_document,
    pdfium_guard,
)

PDF_CLASSIFY_STRATEGY_ENV = "MINERU_PDF_CLASSIFY_STRATEGY"
PDF_CLASSIFY_STRATEGY_HYBRID = "hybrid"
PDF_CLASSIFY_STRATEGY_LEGACY = "legacy"

MAX_SAMPLE_PAGES = 10
CHARS_THRESHOLD = 50
HIGH_IMAGE_COVERAGE_THRESHOLD = 0.8
CID_RATIO_THRESHOLD = 0.05
TEXT_QUALITY_MIN_CHARS = 300
TEXT_QUALITY_BAD_THRESHOLD = 0.03
TEXT_QUALITY_GOOD_THRESHOLD = 0.005

_ALLOWED_CONTROL_CODES = {9, 10, 13}
_PRIVATE_USE_AREA_START = 0xE000
_PRIVATE_USE_AREA_END = 0xF8FF

def classify(pdf_bytes):
    """
    Classify a PDF as text-based or OCR-based.

    Returns:
        "txt" if the PDF can be parsed as text, otherwise "ocr".
    """

    strategy = get_pdf_classify_strategy()
    if strategy == PDF_CLASSIFY_STRATEGY_LEGACY:
        return classify_legacy(pdf_bytes)
    return classify_hybrid(pdf_bytes)


def get_pdf_classify_strategy() -> str:
    strategy = os.getenv(
        PDF_CLASSIFY_STRATEGY_ENV, PDF_CLASSIFY_STRATEGY_HYBRID
    ).strip().lower()
    if strategy not in {
        PDF_CLASSIFY_STRATEGY_HYBRID,
        PDF_CLASSIFY_STRATEGY_LEGACY,
    }:
        logger.warning(
            f"Invalid {PDF_CLASSIFY_STRATEGY_ENV} value: {strategy}, "
            f"fall back to {PDF_CLASSIFY_STRATEGY_HYBRID}"
        )
        return PDF_CLASSIFY_STRATEGY_HYBRID
    return strategy


def classify_hybrid(pdf_bytes):
    """
    Fast PDF classification path.

    The hybrid path uses pdfium + pypdf as the main path and falls back to
    pdfminer only for gray-zone samples.
    """

    pdf = None
    page_indices = []
    should_run_pdfminer_fallback = False

    try:
        with pdfium_guard():
            pdf = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
            page_count = len(pdf)
            if page_count == 0:
                return "ocr"

            page_indices = get_sample_page_indices(page_count, MAX_SAMPLE_PAGES)
            if not page_indices:
                return "ocr"

            if (
                get_avg_cleaned_chars_per_page_pdfium(pdf, page_indices)
                < CHARS_THRESHOLD
            ):
                return "ocr"

            if detect_cid_font_signal_pypdf(pdf_bytes, page_indices):
                return "ocr"

            text_quality_signal = get_text_quality_signal_pdfium(pdf, page_indices)
            total_chars = text_quality_signal["total_chars"]
            abnormal_ratio = text_quality_signal["abnormal_ratio"]

            if total_chars >= TEXT_QUALITY_MIN_CHARS:
                if abnormal_ratio >= TEXT_QUALITY_BAD_THRESHOLD:
                    return "ocr"
                should_run_pdfminer_fallback = abnormal_ratio > TEXT_QUALITY_GOOD_THRESHOLD
            else:
                should_run_pdfminer_fallback = True

            if (
                get_high_image_coverage_ratio_pdfium(pdf, page_indices)
                >= HIGH_IMAGE_COVERAGE_THRESHOLD
            ):
                return "ocr"

    except Exception as e:
        logger.error(f"Failed to classify PDF with hybrid strategy: {e}")
        return "ocr"

    finally:
        close_pdfium_document(pdf)

    if should_run_pdfminer_fallback:
        sample_pdf_bytes = extract_selected_pages(pdf_bytes, page_indices)
        if not sample_pdf_bytes:
            return "ocr"
        if detect_invalid_chars_pdfminer_fallback(sample_pdf_bytes):
            return "ocr"

    return "txt"


def classify_legacy(pdf_bytes):
    """
    Legacy classification path kept for rollback and A/B comparison.
    """

    sample_pdf_bytes = extract_pages(pdf_bytes)
    if not sample_pdf_bytes:
        return "ocr"
    pdf = None
    try:
        with pdfium_guard():
            pdf = open_pdfium_document(pdfium.PdfDocument, sample_pdf_bytes)
            page_count = len(pdf)
            if page_count == 0:
                return "ocr"

            pages_to_check = min(page_count, MAX_SAMPLE_PAGES)

            if (
                get_avg_cleaned_chars_per_page(pdf, pages_to_check) < CHARS_THRESHOLD
            ) or detect_invalid_chars(sample_pdf_bytes):
                return "ocr"

            if (
                get_high_image_coverage_ratio(sample_pdf_bytes, pages_to_check)
                >= HIGH_IMAGE_COVERAGE_THRESHOLD
            ):
                return "ocr"

            return "txt"

    except Exception as e:
        logger.warning(f"Failed to classify PDF with legacy strategy: {e}")
        return "ocr"

    finally:
        close_pdfium_document(pdf)


def get_sample_page_indices(page_count: int, max_pages: int = MAX_SAMPLE_PAGES):
    if page_count <= 0 or max_pages <= 0:
        return []

    sample_count = min(page_count, max_pages)
    if sample_count == page_count:
        return list(range(page_count))
    if sample_count == 1:
        return [0]

    indices = []
    seen = set()
    for i in range(sample_count):
        page_index = round(i * (page_count - 1) / (sample_count - 1))
        page_index = max(0, min(page_count - 1, page_index))
        if page_index not in seen:
            indices.append(page_index)
            seen.add(page_index)

    if len(indices) < sample_count:
        for page_index in range(page_count):
            if page_index in seen:
                continue
            indices.append(page_index)
            seen.add(page_index)
            if len(indices) == sample_count:
                break

    return sorted(indices)


def get_avg_cleaned_chars_per_page(pdf_doc, pages_to_check):
    total_chars = 0
    cleaned_total_chars = 0

    for i in range(pages_to_check):
        page = pdf_doc[i]
        text_page = page.get_textpage()
        text = text_page.get_text_bounded()
        total_chars += len(text)
        cleaned_text = re.sub(r"\s+", "", text)
        cleaned_total_chars += len(cleaned_text)

    avg_cleaned_chars_per_page = cleaned_total_chars / pages_to_check
    return avg_cleaned_chars_per_page


def get_avg_cleaned_chars_per_page_pdfium(pdf_doc, page_indices):
    cleaned_total_chars = 0

    for page_index in page_indices:
        page = pdf_doc[page_index]
        text_page = page.get_textpage()
        text = text_page.get_text_bounded()
        cleaned_total_chars += len(re.sub(r"\s+", "", text))

    if not page_indices:
        return 0.0
    return cleaned_total_chars / len(page_indices)


def get_text_quality_signal_pdfium(pdf_doc, page_indices):
    total_chars = 0
    null_char_count = 0
    replacement_char_count = 0
    control_char_count = 0
    private_use_char_count = 0

    for page_index in page_indices:
        page = pdf_doc[page_index]
        text_page = page.get_textpage()
        char_count = text_page.count_chars()
        total_chars += char_count

        for char_index in range(char_count):
            unicode_code = pdfium_c.FPDFText_GetUnicode(text_page, char_index)
            if unicode_code == 0:
                null_char_count += 1
            elif unicode_code == 0xFFFD:
                replacement_char_count += 1
            elif unicode_code < 32 and unicode_code not in _ALLOWED_CONTROL_CODES:
                control_char_count += 1
            elif _PRIVATE_USE_AREA_START <= unicode_code <= _PRIVATE_USE_AREA_END:
                private_use_char_count += 1

    abnormal_chars = (
        null_char_count
        + replacement_char_count
        + control_char_count
        + private_use_char_count
    )

    abnormal_ratio = 0.0
    if total_chars > 0:
        abnormal_ratio = abnormal_chars / total_chars

    return {
        "total_chars": total_chars,
        "abnormal_ratio": abnormal_ratio,
        "null_char_count": null_char_count,
        "replacement_char_count": replacement_char_count,
        "control_char_count": control_char_count,
        "private_use_char_count": private_use_char_count,
    }


def detect_cid_font_signal_pypdf(pdf_bytes, page_indices):
    reader = PdfReader(BytesIO(pdf_bytes))

    for page_index in page_indices:
        page = reader.pages[page_index]
        resources = _resolve_pdf_object(page.get("/Resources"))
        if not resources:
            continue

        fonts = _resolve_pdf_object(resources.get("/Font"))
        if not fonts:
            continue

        for _, font_ref in fonts.items():
            font = _resolve_pdf_object(font_ref)
            if not font:
                continue

            subtype = str(font.get("/Subtype"))
            encoding = str(font.get("/Encoding"))
            has_descendant_fonts = "/DescendantFonts" in font
            has_to_unicode = "/ToUnicode" in font

            if (
                subtype == "/Type0"
                and encoding in ("/Identity-H", "/Identity-V")
                and has_descendant_fonts
                and not has_to_unicode
            ):
                return True

    return False


def _resolve_pdf_object(obj):
    if hasattr(obj, "get_object"):
        return obj.get_object()
    return obj


def get_high_image_coverage_ratio(sample_pdf_bytes, pages_to_check):
    pdf_stream = BytesIO(sample_pdf_bytes)
    parser = PDFParser(pdf_stream)
    document = PDFDocument(parser)

    if not document.is_extractable:
        return 1.0

    rsrcmgr = PDFResourceManager()
    laparams = LAParams(
        line_overlap=0.5,
        char_margin=2.0,
        line_margin=0.5,
        word_margin=0.1,
        boxes_flow=None,
        detect_vertical=False,
        all_texts=False,
    )
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    high_image_coverage_pages = 0
    page_count = 0

    for page in PDFPage.create_pages(document):
        if page_count >= pages_to_check:
            break

        interpreter.process_page(page)
        layout = device.get_result()

        page_width = layout.width
        page_height = layout.height
        page_area = page_width * page_height

        image_area = 0
        for element in layout:
            if isinstance(element, (LTImage, LTFigure)):
                img_width = element.width
                img_height = element.height
                image_area += img_width * img_height

        coverage_ratio = min(image_area / page_area, 1.0) if page_area > 0 else 0
        if coverage_ratio >= HIGH_IMAGE_COVERAGE_THRESHOLD:
            high_image_coverage_pages += 1

        page_count += 1

    pdf_stream.close()

    if page_count == 0:
        return 0.0

    return high_image_coverage_pages / page_count


def get_high_image_coverage_ratio_pdfium(pdf_doc, page_indices):
    high_image_coverage_pages = 0

    for page_index in page_indices:
        page = pdf_doc[page_index]
        page_bbox = page.get_bbox()
        page_area = abs(
            (page_bbox[2] - page_bbox[0]) * (page_bbox[3] - page_bbox[1])
        )
        image_area = 0.0

        for page_object in page.get_objects(
            filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=3
        ):
            left, bottom, right, top = page_object.get_pos()
            image_area += max(0.0, right - left) * max(0.0, top - bottom)

        coverage_ratio = min(image_area / page_area, 1.0) if page_area > 0 else 0.0
        if coverage_ratio >= HIGH_IMAGE_COVERAGE_THRESHOLD:
            high_image_coverage_pages += 1

    if not page_indices:
        return 0.0
    return high_image_coverage_pages / len(page_indices)


def extract_pages(src_pdf_bytes: bytes) -> bytes:
    """
    Extract up to 10 random pages and return them as a new PDF.
    """

    pdf = None
    sample_docs = None
    try:
        with pdfium_guard():
            pdf = open_pdfium_document(pdfium.PdfDocument, src_pdf_bytes)
            total_page = len(pdf)
            if total_page == 0:
                logger.warning("PDF is empty, return empty document")
                return b""

            if total_page <= MAX_SAMPLE_PAGES:
                return src_pdf_bytes

            select_page_cnt = min(MAX_SAMPLE_PAGES, total_page)
            page_indices = np.random.choice(
                total_page, select_page_cnt, replace=False
            ).tolist()

            sample_docs = open_pdfium_document(pdfium.PdfDocument.new)
            sample_docs.import_pages(pdf, page_indices)

            output_buffer = BytesIO()
            sample_docs.save(output_buffer)
            return output_buffer.getvalue()
    except Exception as e:
        logger.exception(e)
        return src_pdf_bytes
    finally:
        close_pdfium_document(pdf)
        close_pdfium_document(sample_docs)


def extract_selected_pages(src_pdf_bytes: bytes, page_indices) -> bytes:
    """
    Extract specific pages and return them as a new PDF.
    """

    selected_page_indices = sorted(set(page_indices))
    if not selected_page_indices:
        return b""

    pdf = None
    sample_docs = None
    try:
        with pdfium_guard():
            pdf = open_pdfium_document(pdfium.PdfDocument, src_pdf_bytes)
            total_page = len(pdf)
            if total_page == 0:
                logger.warning("PDF is empty, return empty document")
                return b""

            selected_page_indices = [
                page_index
                for page_index in selected_page_indices
                if 0 <= page_index < total_page
            ]
            if not selected_page_indices:
                return b""

            if selected_page_indices == list(range(total_page)):
                return src_pdf_bytes

            sample_docs = open_pdfium_document(pdfium.PdfDocument.new)
            sample_docs.import_pages(pdf, selected_page_indices)

            output_buffer = BytesIO()
            sample_docs.save(output_buffer)
            return output_buffer.getvalue()
    except Exception as e:
        logger.exception(e)
        return src_pdf_bytes
    finally:
        close_pdfium_document(pdf)
        close_pdfium_document(sample_docs)


def detect_invalid_chars(sample_pdf_bytes: bytes) -> bool:
    """
    Detect whether a PDF contains invalid CID-style extracted text.
    """

    sample_pdf_file_like_object = BytesIO(sample_pdf_bytes)
    laparams = LAParams(
        line_overlap=0.5,
        char_margin=2.0,
        line_margin=0.5,
        word_margin=0.1,
        boxes_flow=None,
        detect_vertical=False,
        all_texts=False,
    )
    text = extract_text(pdf_file=sample_pdf_file_like_object, laparams=laparams)
    text = text.replace("\n", "")

    cid_pattern = re.compile(r"\(cid:\d+\)")
    matches = cid_pattern.findall(text)
    cid_count = len(matches)
    cid_len = sum(len(match) for match in matches)
    text_len = len(text)
    if text_len == 0:
        cid_chars_ratio = 0
    else:
        cid_chars_ratio = cid_count / (cid_count + text_len - cid_len)

    return cid_chars_ratio > CID_RATIO_THRESHOLD


def detect_invalid_chars_pdfminer_fallback(sample_pdf_bytes: bytes) -> bool:
    return detect_invalid_chars(sample_pdf_bytes)


if __name__ == "__main__":
    with open("/Users/myhloli/pdf/luanma2x10.pdf", "rb") as f:
        p_bytes = f.read()
        logger.info(f"PDF classify result: {classify(p_bytes)}")
