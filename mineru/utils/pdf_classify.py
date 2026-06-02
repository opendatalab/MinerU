# Copyright (c) Opendatalab. All rights reserved.
import re
from ctypes import byref, c_int, create_string_buffer
from io import BytesIO

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from loguru import logger
from pypdf import PdfReader
from mineru.utils.pdfium_guard import (
    close_pdfium_document,
    open_pdfium_document,
    pdfium_guard,
)

MAX_SAMPLE_PAGES = 10
CHARS_THRESHOLD = 50
HIGH_IMAGE_COVERAGE_THRESHOLD = 0.8
TEXT_QUALITY_MIN_CHARS = 300
TEXT_QUALITY_BAD_THRESHOLD = 0.03
UNICODE_MAP_ERROR_RATIO_THRESHOLD = 0.04
CID_FONT_USAGE_RATIO_THRESHOLD = 0.01
CID_FONT_USAGE_COUNT_THRESHOLD = 30
MAX_PAGE_ASPECT_RATIO = 10.0
SUSPICIOUS_CJK_72XX_START = 0x7280
SUSPICIOUS_CJK_72XX_END = 0x72DF
SUSPICIOUS_CJK_72XX_COUNT_THRESHOLD = 30
SUSPICIOUS_CJK_72XX_CJK_RATIO_THRESHOLD = 0.026
SUSPICIOUS_CJK_72XX_WHITELIST = set(
    "犀犁犄犊犒犟犬犯状犷犹狂狄狈狐狗狙狞"
)
ASCII_PUNCT_CHARS = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
ASCII_PUNCT_RUN_MIN_LENGTH = 4
SUSPICIOUS_ASCII_PUNCT_MIN_TEXT_CHARS = 100
SUSPICIOUS_ASCII_PUNCT_RATIO_THRESHOLD = 0.25
SUSPICIOUS_ASCII_PUNCT_RUN_RATIO_THRESHOLD = 0.10

_ALLOWED_CONTROL_CODES = {9, 10, 13}
_PRIVATE_USE_AREA_START = 0xE000
_PRIVATE_USE_AREA_END = 0xF8FF


def _is_disallowed_control_unicode(unicode_code: int) -> bool:
    return (
        (
            0 <= unicode_code < 32
            or 127 <= unicode_code <= 159
        )
        and unicode_code not in _ALLOWED_CONTROL_CODES
    )


def classify(pdf_bytes):
    """
    Fast PDF classification path.

    The path uses pdfium + pypdf to detect text PDFs and garbled PDFs.

    Returns:
        "txt" if the PDF can be parsed as text, otherwise "ocr".
    """

    pdf = None

    try:
        with pdfium_guard():
            pdf = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
            page_count = len(pdf)
            if page_count == 0:
                return "ocr"

            page_indices = get_sample_page_indices(page_count, MAX_SAMPLE_PAGES)
            if not page_indices:
                return "ocr"

            extreme_page_index, extreme_ratio = get_extreme_aspect_ratio_page_pdfium(
                pdf,
                page_indices,
            )
            if extreme_page_index is not None:
                logger.debug(
                    "Classify PDF as OCR due to extreme sampled-page aspect ratio: "
                    f"page={extreme_page_index + 1}, ratio={extreme_ratio:.2f}"
                )
                return "ocr"

            text_samples = _collect_pdfium_text_samples(pdf, page_indices)
            avg_cleaned_chars_per_page = _get_avg_cleaned_chars_per_page_from_samples(
                text_samples
            )
            if avg_cleaned_chars_per_page < CHARS_THRESHOLD:
                return "ocr"

            unicode_map_error_signal = _get_unicode_map_error_signal_from_samples(
                text_samples
            )
            if (
                unicode_map_error_signal["unicode_map_error_ratio"]
                >= UNICODE_MAP_ERROR_RATIO_THRESHOLD
            ):
                logger.debug(
                    "Classify PDF as OCR due to PDFium Unicode map errors: "
                    f"errors={unicode_map_error_signal['unicode_map_error_count']}, "
                    f"total={unicode_map_error_signal['total_chars']}, "
                    f"ratio={unicode_map_error_signal['unicode_map_error_ratio']:.4f}"
                )
                return "ocr"

            cid_font_signal = get_cid_font_signal_pypdf(pdf_bytes, page_indices)
            cid_font_usage_signal = _get_cid_font_usage_signal_from_samples(
                text_samples,
                cid_font_signal,
            )
            if cid_font_usage_signal["triggered"]:
                logger.debug(
                    "Classify PDF as OCR due to high CID font usage without ToUnicode: "
                    f"page={cid_font_usage_signal['page_index'] + 1}, "
                    f"fonts={cid_font_usage_signal['font_names']}, "
                    f"chars={cid_font_usage_signal['cid_font_char_count']}, "
                    f"total={cid_font_usage_signal['total_chars']}, "
                    f"ratio={cid_font_usage_signal['cid_font_usage_ratio']:.4f}"
                )
                return "ocr"

            text_quality_signal = _get_text_quality_signal_from_samples(text_samples)
            total_chars = text_quality_signal["total_chars"]
            abnormal_ratio = text_quality_signal["abnormal_ratio"]

            if (
                total_chars >= TEXT_QUALITY_MIN_CHARS
                and abnormal_ratio >= TEXT_QUALITY_BAD_THRESHOLD
            ):
                return "ocr"

            u72xx_signal = _get_u72xx_text_signal_from_samples(text_samples)
            if (
                u72xx_signal["u72xx_count"]
                >= SUSPICIOUS_CJK_72XX_COUNT_THRESHOLD
                and u72xx_signal["u72xx_cjk_ratio"]
                >= SUSPICIOUS_CJK_72XX_CJK_RATIO_THRESHOLD
            ):
                logger.debug(
                    "Classify PDF as OCR due to suspicious U+7280-U+72DF text: "
                    f"count={u72xx_signal['u72xx_count']}, "
                    f"cjk_ratio={u72xx_signal['u72xx_cjk_ratio']:.4f}"
                )
                return "ocr"

            ascii_punct_signal = _get_sampled_ascii_punct_signal_from_samples(
                text_samples
            )
            if ascii_punct_signal["triggered"]:
                logger.debug(
                    "Classify PDF as OCR due to suspicious sampled-page ASCII punctuation "
                    f"text: page={ascii_punct_signal['page_index'] + 1}, "
                    f"text_chars={ascii_punct_signal['cleaned_text_chars']}, "
                    f"ascii_punct_ratio="
                    f"{ascii_punct_signal['ascii_punct_ratio']:.4f}, "
                    f"punct_run_ratio={ascii_punct_signal['punct_run_ratio']:.4f}"
                )
                return "ocr"

            if (
                get_high_image_coverage_ratio_pdfium(pdf, page_indices)
                >= HIGH_IMAGE_COVERAGE_THRESHOLD
            ):
                return "ocr"

    except Exception as e:
        logger.error(f"Failed to classify PDF: {e}")
        return "ocr"

    finally:
        close_pdfium_document(pdf)

    return "txt"


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


def get_extreme_aspect_ratio_page_pdfium(
    pdf_doc,
    page_indices,
    max_page_aspect_ratio: float = MAX_PAGE_ASPECT_RATIO,
):
    with pdfium_guard():
        for page_index in page_indices:
            page = None
            try:
                page = pdf_doc[page_index]
                page_width, page_height = page.get_size()
                if page_width <= 0 or page_height <= 0:
                    continue

                aspect_ratio = max(page_width / page_height, page_height / page_width)
                if aspect_ratio > max_page_aspect_ratio:
                    return page_index, aspect_ratio
            finally:
                _close_pdfium_child(page)

    return None, None


def _close_pdfium_child(pdfium_obj) -> None:
    """显式关闭 PDFium 子对象，避免依赖 weakref/finalizer 延迟释放 native 资源。"""
    if pdfium_obj is None:
        return
    close = getattr(pdfium_obj, "close", None)
    if callable(close):
        close()


def _collect_pdfium_text_sample_from_page(page_index, page):
    """从单页 PDFium 对象提取纯 Python 文本统计，并在调用方释放子对象。"""
    text_page = None
    try:
        text_page = page.get_textpage()
        text = text_page.get_text_bounded()
        char_count = text_page.count_chars()
        null_char_count = 0
        replacement_char_count = 0
        control_char_count = 0
        private_use_char_count = 0
        unicode_map_error_count = 0
        font_name_counts = {}

        for char_index in range(char_count):
            unicode_code = pdfium_c.FPDFText_GetUnicode(text_page, char_index)
            if unicode_code == 0:
                null_char_count += 1
            elif unicode_code == 0xFFFD:
                replacement_char_count += 1
            elif _is_disallowed_control_unicode(unicode_code):
                control_char_count += 1
            elif _PRIVATE_USE_AREA_START <= unicode_code <= _PRIVATE_USE_AREA_END:
                private_use_char_count += 1

            if pdfium_c.FPDFText_HasUnicodeMapError(text_page, char_index):
                unicode_map_error_count += 1

            font_name = _normalize_pdf_font_name(
                _get_pdfium_char_font_name(text_page, char_index)
            )
            if font_name:
                font_name_counts[font_name] = font_name_counts.get(font_name, 0) + 1

        return {
            "page_index": page_index,
            "text": text,
            "cleaned_text": re.sub(r"\s+", "", text),
            "char_count": char_count,
            "null_char_count": null_char_count,
            "replacement_char_count": replacement_char_count,
            "control_char_count": control_char_count,
            "private_use_char_count": private_use_char_count,
            "unicode_map_error_count": unicode_map_error_count,
            "font_name_counts": font_name_counts,
        }
    finally:
        _close_pdfium_child(text_page)


def _collect_pdfium_text_samples(pdf_doc, page_indices):
    """一次性收集抽样页文本统计，返回纯 Python 数据，避免缓存 PDFium 子对象。"""
    text_samples = []

    with pdfium_guard():
        for page_index in page_indices:
            page = None
            try:
                page = pdf_doc[page_index]
                text_samples.append(
                    _collect_pdfium_text_sample_from_page(page_index, page)
                )
            finally:
                _close_pdfium_child(page)

    return text_samples


def _get_avg_cleaned_chars_per_page_from_samples(text_samples):
    """基于已缓存的抽样页文本计算平均有效字符数。"""
    cleaned_total_chars = 0

    for text_sample in text_samples:
        cleaned_total_chars += len(text_sample["cleaned_text"])

    if not text_samples:
        return 0.0
    return cleaned_total_chars / len(text_samples)


def get_avg_cleaned_chars_per_page_pdfium(pdf_doc, page_indices):
    text_samples = _collect_pdfium_text_samples(pdf_doc, page_indices)
    return _get_avg_cleaned_chars_per_page_from_samples(text_samples)


def _get_text_quality_signal_from_samples(text_samples):
    """基于已缓存的抽样页字符计数统计异常字符质量信号。"""
    total_chars = 0
    null_char_count = 0
    replacement_char_count = 0
    control_char_count = 0
    private_use_char_count = 0

    for text_sample in text_samples:
        total_chars += text_sample["char_count"]
        null_char_count += text_sample["null_char_count"]
        replacement_char_count += text_sample["replacement_char_count"]
        control_char_count += text_sample["control_char_count"]
        private_use_char_count += text_sample["private_use_char_count"]

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


def get_text_quality_signal_pdfium(pdf_doc, page_indices):
    text_samples = _collect_pdfium_text_samples(pdf_doc, page_indices)
    return _get_text_quality_signal_from_samples(text_samples)


def _get_unicode_map_error_signal_from_samples(text_samples):
    """统计 PDFium 字符级 Unicode 映射失败比例，用于识别无法可靠抽取的乱码文本。"""
    total_chars = 0
    unicode_map_error_count = 0

    for text_sample in text_samples:
        total_chars += text_sample["char_count"]
        unicode_map_error_count += text_sample["unicode_map_error_count"]

    unicode_map_error_ratio = 0.0
    if total_chars > 0:
        unicode_map_error_ratio = unicode_map_error_count / total_chars

    return {
        "total_chars": total_chars,
        "unicode_map_error_count": unicode_map_error_count,
        "unicode_map_error_ratio": unicode_map_error_ratio,
    }


def _normalize_pdf_font_name(font_name) -> str:
    """规范化 PDF 字体名，统一 pypdf 的 NameObject 和 PDFium 返回值格式。"""
    if font_name is None:
        return ""
    return str(font_name).strip().lstrip("/")


def _get_pdfium_char_font_name(text_page, char_index: int) -> str:
    """读取 PDFium 字符级字体名，用于统计可疑 CID 字体的实际使用比例。"""
    flags = c_int()
    buffer_length = pdfium_c.FPDFText_GetFontInfo(
        text_page,
        char_index,
        None,
        0,
        byref(flags),
    )
    if buffer_length <= 0:
        return ""

    font_name_buffer = create_string_buffer(buffer_length)
    actual_length = pdfium_c.FPDFText_GetFontInfo(
        text_page,
        char_index,
        font_name_buffer,
        buffer_length,
        byref(flags),
    )
    if actual_length <= 0:
        return ""

    return font_name_buffer.value.decode("utf-8", errors="ignore")


def _get_cid_font_usage_signal_from_samples(text_samples, cid_font_signal):
    """基于 PDFium 字符级字体名统计可疑 CID 字体在抽样页中的真实使用比例。"""
    best_signal = {
        "triggered": False,
        "page_index": None,
        "font_names": [],
        "cid_font_char_count": 0,
        "total_chars": 0,
        "cid_font_usage_ratio": 0.0,
    }
    if not cid_font_signal or not cid_font_signal.get("triggered"):
        return best_signal

    page_fonts = cid_font_signal.get("page_fonts") or {}
    for text_sample in text_samples:
        page_index = text_sample.get("page_index")
        cid_font_names = {
            _normalize_pdf_font_name(font_name)
            for font_name in page_fonts.get(page_index, set())
        }
        cid_font_names.discard("")
        if not cid_font_names:
            continue

        total_chars = text_sample["char_count"]
        if total_chars <= 0:
            continue

        font_name_counts = text_sample.get("font_name_counts") or {}
        matched_font_names = cid_font_names.intersection(font_name_counts)
        cid_font_char_count = sum(
            font_name_counts[font_name] for font_name in matched_font_names
        )

        cid_font_usage_ratio = cid_font_char_count / total_chars
        signal = {
            "triggered": False,
            "page_index": page_index,
            "font_names": sorted(matched_font_names),
            "cid_font_char_count": cid_font_char_count,
            "total_chars": total_chars,
            "cid_font_usage_ratio": cid_font_usage_ratio,
        }
        if (
            cid_font_char_count >= CID_FONT_USAGE_COUNT_THRESHOLD
            and cid_font_usage_ratio >= CID_FONT_USAGE_RATIO_THRESHOLD
        ):
            signal["triggered"] = True
            return signal

        if (
            signal["cid_font_usage_ratio"],
            signal["cid_font_char_count"],
        ) > (
            best_signal["cid_font_usage_ratio"],
            best_signal["cid_font_char_count"],
        ):
            best_signal = signal

    return best_signal


def _get_u72xx_text_signal_from_samples(text_samples):
    """基于已缓存的抽样页文本统计扣除常用字后的 U+7280-U+72DF 字符占比。"""
    cjk_chars = 0
    u72xx_count = 0

    for text_sample in text_samples:
        for char in text_sample["cleaned_text"]:
            unicode_code = ord(char)
            if 0x4E00 <= unicode_code <= 0x9FFF:
                cjk_chars += 1
            if (
                SUSPICIOUS_CJK_72XX_START
                <= unicode_code
                <= SUSPICIOUS_CJK_72XX_END
                and char not in SUSPICIOUS_CJK_72XX_WHITELIST
            ):
                u72xx_count += 1

    u72xx_cjk_ratio = 0.0
    if cjk_chars > 0:
        u72xx_cjk_ratio = u72xx_count / cjk_chars

    return {
        "cjk_chars": cjk_chars,
        "u72xx_count": u72xx_count,
        "u72xx_cjk_ratio": u72xx_cjk_ratio,
    }


def get_u72xx_text_signal_pdfium(pdf_doc, page_indices):
    """统计抽样页中扣除常用字后的 U+7280-U+72DF 字符占比，用于识别可疑 ToUnicode 映射。"""
    text_samples = _collect_pdfium_text_samples(pdf_doc, page_indices)
    return _get_u72xx_text_signal_from_samples(text_samples)


def _count_ascii_punct_run_chars(text: str) -> int:
    """统计连续 ASCII 标点字符数，仅累计长度达到阈值的 run。"""
    run_chars = 0
    current_run = 0

    for char in text:
        if char in ASCII_PUNCT_CHARS:
            current_run += 1
            continue

        if current_run >= ASCII_PUNCT_RUN_MIN_LENGTH:
            run_chars += current_run
        current_run = 0

    if current_run >= ASCII_PUNCT_RUN_MIN_LENGTH:
        run_chars += current_run

    return run_chars


def _get_sampled_ascii_punct_signal_from_samples(text_samples):
    """检查所有抽样页的 ASCII 标点密集度，用于识别无 ToUnicode 的乱码文本。"""
    best_signal = {
        "triggered": False,
        "page_index": None,
        "cleaned_text_chars": 0,
        "ascii_punct_count": 0,
        "ascii_punct_ratio": 0.0,
        "ascii_punct_run_chars": 0,
        "punct_run_ratio": 0.0,
    }

    for text_sample in text_samples:
        page_index = text_sample.get("page_index")
        cleaned_text = text_sample["cleaned_text"]
        cleaned_text_chars = len(cleaned_text)
        ascii_punct_count = sum(
            1 for char in cleaned_text if char in ASCII_PUNCT_CHARS
        )
        ascii_punct_run_chars = _count_ascii_punct_run_chars(cleaned_text)

        ascii_punct_ratio = 0.0
        punct_run_ratio = 0.0
        if cleaned_text_chars > 0:
            ascii_punct_ratio = ascii_punct_count / cleaned_text_chars
            punct_run_ratio = ascii_punct_run_chars / cleaned_text_chars

        signal = {
            "triggered": False,
            "page_index": page_index,
            "cleaned_text_chars": cleaned_text_chars,
            "ascii_punct_count": ascii_punct_count,
            "ascii_punct_ratio": ascii_punct_ratio,
            "ascii_punct_run_chars": ascii_punct_run_chars,
            "punct_run_ratio": punct_run_ratio,
        }
        if (
            cleaned_text_chars >= SUSPICIOUS_ASCII_PUNCT_MIN_TEXT_CHARS
            and ascii_punct_ratio >= SUSPICIOUS_ASCII_PUNCT_RATIO_THRESHOLD
            and punct_run_ratio >= SUSPICIOUS_ASCII_PUNCT_RUN_RATIO_THRESHOLD
        ):
            signal["triggered"] = True
            return signal

        # 未触发时保留最可疑的抽样页指标，方便日志扩展和后续排查阈值边界。
        if (
            signal["punct_run_ratio"],
            signal["ascii_punct_ratio"],
            signal["cleaned_text_chars"],
        ) > (
            best_signal["punct_run_ratio"],
            best_signal["ascii_punct_ratio"],
            best_signal["cleaned_text_chars"],
        ):
            best_signal = signal

    return best_signal


def get_cid_font_signal_pypdf(pdf_bytes, page_indices):
    """收集抽样页中无 ToUnicode 的 Identity CID 字体资源，供后续按实际字符使用量判定。"""
    reader = PdfReader(BytesIO(pdf_bytes))
    page_fonts = {}

    for page_index in page_indices:
        page = reader.pages[page_index]
        resources = _resolve_pdf_object(page.get("/Resources"))
        if not resources:
            continue

        fonts = _resolve_pdf_object(resources.get("/Font"))
        if not fonts:
            continue

        for font_key, font_ref in fonts.items():
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
                font_name = font.get("/BaseFont") or font_key
                page_fonts.setdefault(page_index, set()).add(
                    _normalize_pdf_font_name(font_name)
                )

    return {
        "triggered": bool(page_fonts),
        "page_fonts": page_fonts,
    }


def detect_cid_font_signal_pypdf(pdf_bytes, page_indices):
    """兼容旧接口：只返回是否存在无 ToUnicode 的 Identity CID 字体资源。"""
    return get_cid_font_signal_pypdf(pdf_bytes, page_indices)["triggered"]


def _resolve_pdf_object(obj):
    if hasattr(obj, "get_object"):
        return obj.get_object()
    return obj


def get_high_image_coverage_ratio_pdfium(pdf_doc, page_indices):
    high_image_coverage_pages = 0

    with pdfium_guard():
        for page_index in page_indices:
            page = None
            try:
                page = pdf_doc[page_index]
                page_bbox = page.get_bbox()
                page_area = abs(
                    (page_bbox[2] - page_bbox[0]) * (page_bbox[3] - page_bbox[1])
                )
                image_area = 0.0

                for page_object in page.get_objects(
                    filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=3
                ):
                    try:
                        left, bottom, right, top = page_object.get_pos()
                        image_area += max(0.0, right - left) * max(0.0, top - bottom)
                    finally:
                        _close_pdfium_child(page_object)

                coverage_ratio = (
                    min(image_area / page_area, 1.0) if page_area > 0 else 0.0
                )
                if coverage_ratio >= HIGH_IMAGE_COVERAGE_THRESHOLD:
                    high_image_coverage_pages += 1
            finally:
                _close_pdfium_child(page)

    if not page_indices:
        return 0.0
    return high_image_coverage_pages / len(page_indices)


if __name__ == "__main__":
    with open("/Users/myhloli/pdf/luanma2x10.pdf", "rb") as f:
        p_bytes = f.read()
        logger.info(f"PDF classify result: {classify(p_bytes)}")
