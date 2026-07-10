# Copyright (c) Opendatalab. All rights reserved.
import re
from ctypes import byref, c_int, create_string_buffer
from io import BytesIO

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from loguru import logger
from pypdf import PdfReader
from mineru.utils.pdfium_guard import (
    close_pdfium_child,
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
LATIN_CJK_FONT_USAGE_RATIO_THRESHOLD = 0.01
LATIN_CJK_FONT_USAGE_COUNT_THRESHOLD = 30
LATIN_CJK_FONT_CJK_RATIO_THRESHOLD = 0.8
LATIN_CHARSET_MIN_LATIN_GLYPHS = 10
LATIN_CHARSET_MIN_LATIN_RATIO = 0.5
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
SUSPICIOUS_CROSS_SCRIPT_MIN_TEXT_CHARS = 300
SUSPICIOUS_CROSS_SCRIPT_MIN_CJK_CHARS = 100
SUSPICIOUS_CROSS_SCRIPT_COUNT_THRESHOLD = 120
SUSPICIOUS_CROSS_SCRIPT_RATIO_THRESHOLD = 0.18
SUSPICIOUS_CROSS_SCRIPT_MIN_SCRIPT_COUNT = 3
SUSPICIOUS_CROSS_SCRIPT_SCRIPT_MIN_CHARS = 5
SUSPICIOUS_CROSS_SCRIPT_RANGES = (
    (0x0400, 0x052F, "Cyrillic"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0700, 0x074F, "Syriac"),
    (0x0750, 0x077F, "Arabic Supplement"),
    (0x0780, 0x07BF, "Thaana"),
    (0x07C0, 0x07FF, "NKo"),
    (0x0800, 0x083F, "Samaritan"),
    (0x0840, 0x085F, "Mandaic"),
    (0x0860, 0x086F, "Syriac Supplement"),
    (0x0870, 0x089F, "Arabic Extended-B"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x1000, 0x109F, "Myanmar"),
    (0x1100, 0x11FF, "Hangul Jamo"),
    (0x1200, 0x137F, "Ethiopic"),
    (0x13A0, 0x13FF, "Cherokee"),
    (0x1400, 0x167F, "Canadian Syllabics"),
    (0x1800, 0x18AF, "Mongolian"),
    (0x1A20, 0x1AAF, "Tai Tham"),
    (0x2C00, 0x2C5F, "Glagolitic"),
    (0xA000, 0xA48F, "Yi"),
)
CJK_TEXT_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2EBEF),
)

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

            font_resource_signals = _get_font_resource_signals_pypdf(
                pdf_bytes,
                page_indices,
            )
            cid_font_usage_signal = _get_cid_font_usage_signal_from_samples(
                text_samples,
                font_resource_signals["cid_without_to_unicode"],
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

            latin_cjk_font_usage_signal = _get_latin_font_cjk_usage_signal_from_samples(
                text_samples,
                font_resource_signals["latin_charset_with_to_unicode"],
                count_threshold=LATIN_CJK_FONT_USAGE_COUNT_THRESHOLD,
                usage_ratio_threshold=LATIN_CJK_FONT_USAGE_RATIO_THRESHOLD,
                cjk_ratio_threshold=LATIN_CJK_FONT_CJK_RATIO_THRESHOLD,
            )
            if latin_cjk_font_usage_signal["triggered"]:
                logger.debug(
                    "Classify PDF as OCR due to Latin CharSet font decoding as CJK: "
                    f"page={latin_cjk_font_usage_signal['page_index'] + 1}, "
                    f"fonts={latin_cjk_font_usage_signal['font_names']}, "
                    f"chars={latin_cjk_font_usage_signal['font_char_count']}, "
                    f"cjk={latin_cjk_font_usage_signal['cjk_char_count']}, "
                    f"total={latin_cjk_font_usage_signal['total_chars']}, "
                    f"usage_ratio="
                    f"{latin_cjk_font_usage_signal['font_usage_ratio']:.4f}, "
                    f"cjk_ratio="
                    f"{latin_cjk_font_usage_signal['font_cjk_ratio']:.4f}"
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

            cross_script_signal = _get_cross_script_text_signal_from_samples(
                text_samples
            )
            if cross_script_signal["triggered"]:
                logger.debug(
                    "Classify PDF as OCR due to suspicious cross-script text: "
                    f"chars={cross_script_signal['total_chars']}, "
                    f"cjk={cross_script_signal['cjk_chars']}, "
                    f"suspicious={cross_script_signal['suspicious_chars']}, "
                    f"ratio={cross_script_signal['suspicious_ratio']:.4f}, "
                    f"scripts={cross_script_signal['top_scripts']}"
                )
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
                close_pdfium_child(page)

    return None, None


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
        non_generated_char_count = 0
        font_non_generated_char_counts = {}
        font_non_generated_cjk_char_counts = {}

        for char_index in range(char_count):
            unicode_code = pdfium_c.FPDFText_GetUnicode(text_page, char_index)
            is_generated = pdfium_c.FPDFText_IsGenerated(text_page, char_index) == 1
            if not is_generated:
                non_generated_char_count += 1

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
                if not is_generated:
                    font_non_generated_char_counts[font_name] = (
                        font_non_generated_char_counts.get(font_name, 0) + 1
                    )
                    if _is_cjk_unicode_code(unicode_code):
                        font_non_generated_cjk_char_counts[font_name] = (
                            font_non_generated_cjk_char_counts.get(font_name, 0) + 1
                        )

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
            "non_generated_char_count": non_generated_char_count,
            "font_non_generated_char_counts": font_non_generated_char_counts,
            "font_non_generated_cjk_char_counts": (
                font_non_generated_cjk_char_counts
            ),
        }
    finally:
        close_pdfium_child(text_page)


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
                close_pdfium_child(page)

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


def _is_cjk_unicode_code(unicode_code: int) -> bool:
    """判断 Unicode 码点是否属于分类器认可的 CJK 文本范围。"""
    return any(start <= unicode_code <= end for start, end in CJK_TEXT_RANGES)


def _get_cjk_glyph_name_code(glyph_name: str) -> int | None:
    """解析 uniXXXX/uXXXXX 形式的 CJK glyph name，其他名称返回 None。"""
    match = re.fullmatch(
        r"(?:uni([0-9A-Fa-f]{4,6})|u([0-9A-Fa-f]{4,6}))",
        glyph_name,
    )
    if match is None:
        return None
    unicode_code = int(match.group(1) or match.group(2), 16)
    if not _is_cjk_unicode_code(unicode_code):
        return None
    return unicode_code


def _get_empty_latin_charset_with_to_unicode_signal() -> dict:
    """构造未触发的 Type1 Latin CharSet 字体候选信号。"""
    return {
        "triggered": False,
        "charset_glyph_count": 0,
        "latin_glyph_count": 0,
        "latin_glyph_ratio": 0.0,
        "cjk_charset_glyph_count": 0,
        "cjk_charset_glyph_ratio": 0.0,
    }


def _get_latin_charset_with_to_unicode_signal(font: object) -> dict:
    """识别带 ToUnicode 且 CharSet 明显为 Latin 的 Type1 字体候选。"""
    signal = _get_empty_latin_charset_with_to_unicode_signal()
    if str(font.get("/Subtype")) != "/Type1":
        return signal

    descriptor = _resolve_pdf_object(font.get("/FontDescriptor"))
    to_unicode = _resolve_pdf_object(font.get("/ToUnicode"))
    if descriptor is None or to_unicode is None:
        return signal

    charset = descriptor.get("/CharSet")
    if charset is None:
        return signal

    glyph_names = set(re.findall(r"/([^/\s]+)", str(charset)))
    charset_glyph_count = len(glyph_names)
    latin_glyph_count = sum(
        1 for glyph_name in glyph_names if re.fullmatch(r"[A-Za-z]", glyph_name)
    )
    cjk_charset_glyph_count = sum(
        1
        for glyph_name in glyph_names
        if _get_cjk_glyph_name_code(glyph_name) is not None
    )
    latin_glyph_ratio = (
        latin_glyph_count / charset_glyph_count if charset_glyph_count else 0.0
    )
    cjk_charset_glyph_ratio = (
        cjk_charset_glyph_count / charset_glyph_count
        if charset_glyph_count
        else 0.0
    )

    signal.update(
        {
            "charset_glyph_count": charset_glyph_count,
            "latin_glyph_count": latin_glyph_count,
            "latin_glyph_ratio": latin_glyph_ratio,
            "cjk_charset_glyph_count": cjk_charset_glyph_count,
            "cjk_charset_glyph_ratio": cjk_charset_glyph_ratio,
        }
    )
    signal["triggered"] = (
        latin_glyph_count >= LATIN_CHARSET_MIN_LATIN_GLYPHS
        and latin_glyph_ratio >= LATIN_CHARSET_MIN_LATIN_RATIO
        and cjk_charset_glyph_count == 0
    )
    return signal


def _normalize_pdf_font_name(font_name) -> str:
    """规范化 PDF 字体名，统一 pypdf 的 NameObject 和 PDFium 返回值格式。"""
    if font_name is None:
        return ""
    normalized_name = str(font_name).strip().lstrip("/")
    return re.sub(r"^[A-Z]{6}\+", "", normalized_name, count=1)


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


def _get_latin_font_cjk_usage_signal_from_samples(
    text_samples: list[dict],
    font_signal: dict,
    count_threshold: int,
    usage_ratio_threshold: float,
    cjk_ratio_threshold: float,
) -> dict:
    """按单个 Latin 候选字体统计实际使用量及 PDFium 解码后的 CJK 比例。"""
    best_signal = {
        "triggered": False,
        "page_index": None,
        "font_names": [],
        "font_char_count": 0,
        "cjk_char_count": 0,
        "total_chars": 0,
        "font_usage_ratio": 0.0,
        "font_cjk_ratio": 0.0,
    }
    if not font_signal or not font_signal.get("triggered"):
        return best_signal

    page_fonts = font_signal.get("page_fonts") or {}
    for text_sample in text_samples:
        page_index = text_sample.get("page_index")
        total_chars = text_sample.get("non_generated_char_count", 0)
        if total_chars <= 0:
            continue

        font_name_counts = text_sample.get("font_non_generated_char_counts") or {}
        font_cjk_char_counts = (
            text_sample.get("font_non_generated_cjk_char_counts") or {}
        )
        candidate_font_names = {
            _normalize_pdf_font_name(font_name)
            for font_name in page_fonts.get(page_index, set())
        }
        candidate_font_names.discard("")

        for font_name in sorted(candidate_font_names):
            font_char_count = font_name_counts.get(font_name, 0)
            cjk_char_count = font_cjk_char_counts.get(font_name, 0)
            font_usage_ratio = font_char_count / total_chars
            font_cjk_ratio = (
                cjk_char_count / font_char_count if font_char_count else 0.0
            )
            signal = {
                "triggered": False,
                "page_index": page_index,
                "font_names": [font_name] if font_char_count else [],
                "font_char_count": font_char_count,
                "cjk_char_count": cjk_char_count,
                "total_chars": total_chars,
                "font_usage_ratio": font_usage_ratio,
                "font_cjk_ratio": font_cjk_ratio,
            }
            if (
                font_char_count >= count_threshold
                and font_usage_ratio >= usage_ratio_threshold
                and font_cjk_ratio >= cjk_ratio_threshold
            ):
                signal["triggered"] = True
                return signal

            if (
                signal["font_cjk_ratio"],
                signal["font_usage_ratio"],
                signal["font_char_count"],
            ) > (
                best_signal["font_cjk_ratio"],
                best_signal["font_usage_ratio"],
                best_signal["font_char_count"],
            ):
                best_signal = signal

    return best_signal


def _is_cjk_text_char(char: str) -> bool:
    """判断字符是否属于中文文档中可接受的 CJK 文字范围。"""
    return _is_cjk_unicode_code(ord(char))


def _get_cross_script_name(char: str) -> str | None:
    """识别中文文档乱码中常见的跨脚本字符块名称。"""
    unicode_code = ord(char)
    for start, end, script_name in SUSPICIOUS_CROSS_SCRIPT_RANGES:
        if start <= unicode_code <= end:
            return script_name
    return None


def _get_cross_script_text_signal_from_samples(text_samples):
    """统计中文文档文本层中大比例跨脚本混入信号，用于识别合法 Unicode 错码。"""
    total_chars = 0
    cjk_chars = 0
    suspicious_chars = 0
    script_counts = {}

    for text_sample in text_samples:
        for char in text_sample["cleaned_text"]:
            total_chars += 1
            if _is_cjk_text_char(char):
                cjk_chars += 1

            script_name = _get_cross_script_name(char)
            if script_name is None:
                continue

            suspicious_chars += 1
            script_counts[script_name] = script_counts.get(script_name, 0) + 1

    suspicious_ratio = 0.0
    if total_chars > 0:
        suspicious_ratio = suspicious_chars / total_chars

    dense_script_count = sum(
        1
        for count in script_counts.values()
        if count >= SUSPICIOUS_CROSS_SCRIPT_SCRIPT_MIN_CHARS
    )
    top_scripts = sorted(
        script_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:5]
    triggered = (
        total_chars >= SUSPICIOUS_CROSS_SCRIPT_MIN_TEXT_CHARS
        and cjk_chars >= SUSPICIOUS_CROSS_SCRIPT_MIN_CJK_CHARS
        and suspicious_chars >= SUSPICIOUS_CROSS_SCRIPT_COUNT_THRESHOLD
        and suspicious_ratio >= SUSPICIOUS_CROSS_SCRIPT_RATIO_THRESHOLD
        and dense_script_count >= SUSPICIOUS_CROSS_SCRIPT_MIN_SCRIPT_COUNT
    )

    return {
        "triggered": triggered,
        "total_chars": total_chars,
        "cjk_chars": cjk_chars,
        "suspicious_chars": suspicious_chars,
        "suspicious_ratio": suspicious_ratio,
        "script_counts": script_counts,
        "top_scripts": top_scripts,
        "dense_script_count": dense_script_count,
    }


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


def _get_ascii_punct_run_signal(text: str) -> tuple[int, set[str]]:
    """统计长连续 ASCII 标点 run 的字符数和标点类型，用于区分目录点线和乱码。"""
    run_chars = 0
    run_punct_chars = set()
    current_run = 0
    current_punct_chars = set()

    for char in text:
        if char in ASCII_PUNCT_CHARS:
            current_run += 1
            current_punct_chars.add(char)
            continue

        if current_run >= ASCII_PUNCT_RUN_MIN_LENGTH:
            run_chars += current_run
            run_punct_chars.update(current_punct_chars)
        current_run = 0
        current_punct_chars.clear()

    if current_run >= ASCII_PUNCT_RUN_MIN_LENGTH:
        run_chars += current_run
        run_punct_chars.update(current_punct_chars)

    return run_chars, run_punct_chars


def _count_ascii_punct_run_chars(text: str) -> int:
    """统计连续 ASCII 标点字符数，仅累计长度达到阈值的 run。"""
    run_chars, _ = _get_ascii_punct_run_signal(text)

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
        ascii_punct_run_chars, ascii_punct_run_char_types = (
            _get_ascii_punct_run_signal(cleaned_text)
        )

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
            and len(ascii_punct_run_char_types) > 1
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


def _get_font_resource_cache_key(font_ref: object, font: object) -> tuple:
    """为 pypdf 字体资源生成页间可复用的分析缓存键。"""
    idnum = getattr(font_ref, "idnum", None)
    generation = getattr(font_ref, "generation", None)
    if idnum is not None:
        return "indirect", idnum, generation
    return "direct", id(font)


def _get_font_resource_signals_pypdf(
    pdf_bytes: bytes,
    page_indices: list[int],
) -> dict:
    """一次扫描抽样页字体资源，收集 CID 缺映射和 Type1 Latin 候选字体。"""
    reader = PdfReader(BytesIO(pdf_bytes))
    cid_page_fonts = {}
    latin_charset_page_fonts = {}
    font_analysis_cache = {}

    for page_index in page_indices:
        page = reader.pages[page_index]
        resources = _resolve_pdf_object(page.get("/Resources"))
        if not resources:
            continue

        fonts = _resolve_pdf_object(resources.get("/Font"))
        if not fonts:
            continue

        # PDFium 只返回规范化字体名；同名不同资源无法可靠归因到具体字体对象。
        page_latin_font_resources = {}
        for font_key, font_ref in fonts.items():
            font = _resolve_pdf_object(font_ref)
            if not font:
                continue

            font_name = _normalize_pdf_font_name(font.get("/BaseFont") or font_key)
            if not font_name:
                continue

            cache_key = _get_font_resource_cache_key(font_ref, font)
            analysis = font_analysis_cache.get(cache_key)
            if analysis is None:
                subtype = str(font.get("/Subtype"))
                encoding = str(font.get("/Encoding"))
                cid_without_to_unicode = (
                    subtype == "/Type0"
                    and encoding in ("/Identity-H", "/Identity-V")
                    and "/DescendantFonts" in font
                    and "/ToUnicode" not in font
                )
                latin_charset_signal = _get_latin_charset_with_to_unicode_signal(font)
                analysis = {
                    "cid_without_to_unicode": cid_without_to_unicode,
                    "latin_charset_with_to_unicode": latin_charset_signal["triggered"],
                }
                font_analysis_cache[cache_key] = analysis

            if analysis["cid_without_to_unicode"]:
                cid_page_fonts.setdefault(page_index, set()).add(font_name)

            page_latin_font_resources.setdefault(font_name, {})[cache_key] = analysis[
                "latin_charset_with_to_unicode"
            ]

        for font_name, resource_states in page_latin_font_resources.items():
            if len(resource_states) == 1 and set(resource_states.values()) == {True}:
                latin_charset_page_fonts.setdefault(page_index, set()).add(
                    font_name
                )

    return {
        "cid_without_to_unicode": {
            "triggered": bool(cid_page_fonts),
            "page_fonts": cid_page_fonts,
        },
        "latin_charset_with_to_unicode": {
            "triggered": bool(latin_charset_page_fonts),
            "page_fonts": latin_charset_page_fonts,
        },
    }


def get_cid_font_signal_pypdf(
    pdf_bytes: bytes,
    page_indices: list[int],
) -> dict:
    """兼容旧接口：返回抽样页无 ToUnicode 的 Identity CID 字体资源。"""
    return _get_font_resource_signals_pypdf(
        pdf_bytes,
        page_indices,
    )["cid_without_to_unicode"]


def detect_cid_font_signal_pypdf(
    pdf_bytes: bytes,
    page_indices: list[int],
) -> bool:
    """兼容旧接口：只返回是否存在无 ToUnicode 的 Identity CID 字体资源。"""
    return get_cid_font_signal_pypdf(pdf_bytes, page_indices)["triggered"]


def _resolve_pdf_object(obj):
    if hasattr(obj, "get_object"):
        return obj.get_object()
    return obj


def _get_pdfium_page_object_bounds(page_object):
    """兼容 pypdfium2 4.x/5.x，统一获取页面对象的边界坐标。"""
    get_bounds = getattr(page_object, "get_bounds", None)
    if callable(get_bounds):
        return get_bounds()

    get_pos = getattr(page_object, "get_pos", None)
    if callable(get_pos):
        return get_pos()

    raise AttributeError(
        "PDFium page object has neither get_bounds() nor get_pos()"
    )


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
                        left, bottom, right, top = _get_pdfium_page_object_bounds(
                            page_object
                        )
                        image_area += max(0.0, right - left) * max(0.0, top - bottom)
                    finally:
                        close_pdfium_child(page_object)

                coverage_ratio = (
                    min(image_area / page_area, 1.0) if page_area > 0 else 0.0
                )
                if coverage_ratio >= HIGH_IMAGE_COVERAGE_THRESHOLD:
                    high_image_coverage_pages += 1
            finally:
                close_pdfium_child(page)

    if not page_indices:
        return 0.0
    return high_image_coverage_pages / len(page_indices)


if __name__ == "__main__":
    with open("/Users/myhloli/pdf/luanma2x10.pdf", "rb") as f:
        p_bytes = f.read()
        logger.info(f"PDF classify result: {classify(p_bytes)}")
