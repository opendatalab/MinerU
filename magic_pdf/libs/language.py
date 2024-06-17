import regex
import unicodedata
from fast_langdetect import detect_langs

RE_BAD_CHARS = regex.compile(r"\p{Cc}|\p{Cs}")


def remove_bad_chars(text):
    return RE_BAD_CHARS.sub("", text)


def detect_lang(text: str) -> str:
    if len(text) == 0:
        return ""
    try:
        lang_upper = detect_langs(text)
    except:
        html_no_ctrl_chars = ''.join([l for l in text if unicodedata.category(l)[0] not in ['C', ]])
        lang_upper = detect_langs(html_no_ctrl_chars)
    try:
        lang = lang_upper.lower()
    except:
        lang = ""
    return lang


if __name__ == '__main__':
    print(detect_lang("This is a test."))
    print(detect_lang("<html>This is a test</html>"))
    print(detect_lang("这个是中文测试。"))
    print(detect_lang("<html>这个是中文测试。</html>"))
