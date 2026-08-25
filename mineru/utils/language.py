# Copyright (c) Opendatalab. All rights reserved.
import os
from pathlib import Path
import unicodedata

DEFAULT_CODE_LANGUAGE = "txt"


def _detect_language(text: str) -> object:
    """首次检测时配置本地模型缓存并惰性加载语言识别器。"""
    if not os.getenv("FTLANG_CACHE"):
        cache_dir = Path(__file__).resolve().parents[1] / "resources" / "fasttext-langdetect"
        os.environ["FTLANG_CACHE"] = str(cache_dir)
    from fast_langdetect import detect_language

    return detect_language(text)


def remove_invalid_surrogates(text: str) -> str:
    # 移除无效的 UTF-16 代理对
    return "".join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))


def detect_lang(text: str) -> str:
    if len(text) == 0:
        return ""

    text = text.replace("\n", "")
    text = remove_invalid_surrogates(text)

    try:
        lang_upper = _detect_language(text)
    except Exception:
        html_no_ctrl_chars = "".join([c for c in text if unicodedata.category(c)[0] not in ["C"]])
        lang_upper = _detect_language(html_no_ctrl_chars)

    try:
        lang = lang_upper.lower()
    except Exception:
        lang = ""
    return lang


def _normalize_text_for_language_guess(code: str) -> str:
    """移除孤立代理字符并还原合法代理对，供代码语言识别使用。"""
    if not code:
        return ""
    normalized: list[str] = []
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


def guess_code_language(code: str) -> str:
    """使用 Magika 推断代码块语言，失败时返回纯文本类型。"""
    normalized_code = _normalize_text_for_language_guess(code)
    if not normalized_code:
        return DEFAULT_CODE_LANGUAGE
    try:
        from magika import Magika

        lang = Magika().identify_bytes(normalized_code.encode("utf-8", errors="replace")).prediction.output.label
    except Exception:
        return DEFAULT_CODE_LANGUAGE
    return lang if lang != "unknown" else DEFAULT_CODE_LANGUAGE


if __name__ == "__main__":
    print(os.getenv("FTLANG_CACHE"))
    print(detect_lang("This is a test."))
    print(detect_lang("<html>This is a test</html>"))
    print(detect_lang("这个是中文测试。"))
    print(detect_lang("<html>这个是中文测试。</html>"))
    print(detect_lang("〖\ud835\udc46\ud835〗这是个包含utf-16的中文测试"))


__all__ = ["DEFAULT_CODE_LANGUAGE", "detect_lang", "guess_code_language", "remove_invalid_surrogates"]
