from pathlib import Path

from loguru import logger
from magika import Magika


DEFAULT_LANG = "txt"
PDF_SIG_BYTES = b'%PDF'
magika = Magika()

def _normalize_text_for_language_guess(code: str) -> str:
    if not code:
        return ""

    normalized = []
    index = 0
    while index < len(code):
        current_char = code[index]
        current_ord = ord(current_char)

        if 0xD800 <= current_ord <= 0xDBFF:
            if index + 1 < len(code):
                next_char = code[index + 1]
                next_ord = ord(next_char)
                if 0xDC00 <= next_ord <= 0xDFFF:
                    pair = current_char + next_char
                    normalized.append(pair.encode("utf-16", "surrogatepass").decode("utf-16"))
                    index += 2
                    continue
            index += 1
            continue

        if 0xDC00 <= current_ord <= 0xDFFF:
            index += 1
            continue

        normalized.append(current_char)
        index += 1

    return "".join(normalized)


def guess_language_by_text(code):
    normalized_code = _normalize_text_for_language_guess(code)
    if not normalized_code:
        return DEFAULT_LANG

    try:
        codebytes = normalized_code.encode("utf-8", errors="replace")
        lang = magika.identify_bytes(codebytes).prediction.output.label
    except Exception:
        return DEFAULT_LANG

    return lang if lang != "unknown" else DEFAULT_LANG


def guess_suffix_by_bytes(file_bytes, file_path=None) -> str:
    suffix = magika.identify_bytes(file_bytes).prediction.output.label
    if file_path and suffix in ["ai", "html"] and Path(file_path).suffix.lower() in [".pdf"] and file_bytes[:4] == PDF_SIG_BYTES:
        suffix = "pdf"
    return suffix


def guess_suffix_by_path(file_path) -> str:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    suffix = magika.identify_path(file_path).prediction.output.label
    if suffix in ["ai", "html"] and file_path.suffix.lower() in [".pdf"]:
        try:
            with open(file_path, 'rb') as f:
                if f.read(4) == PDF_SIG_BYTES:
                    suffix = "pdf"
        except Exception as e:
            logger.warning(f"Failed to read file {file_path} for PDF signature check: {e}")
    return suffix
