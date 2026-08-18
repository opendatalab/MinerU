# Copyright (c) Opendatalab. All rights reserved.
"""跨模型共享的文本字符规范化与换行连接规则。"""

import re

# 中日韩文本的物理换行通常不需要插入额外空格，集中定义以供各后端和渲染器共享。
CJK_LANGS = frozenset({"zh", "ja", "ko"})

# PDF 文本抽取时，英文跨行断词可能被编码为多种 hyphen 字符。
# 这里只用于判断“行末英文断词符”，不要扩展到 en/em dash 等普通破折号。
LINE_END_HYPHEN_CHARS = "-\u00ad\u2010\u2011\u2043"
LINE_END_HYPHEN_RE = re.compile(rf"[A-Za-z]+[{re.escape(LINE_END_HYPHEN_CHARS)}]\s*$")

# URL 候选仅允许 RFC 3986 常见 ASCII 字符，避免把中文正文吞入链接。
_URL_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:(?:https?|ftp)://|www\.)[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+",
    re.ASCII | re.IGNORECASE,
)
# 下一行自身以完整 URL 开头时，必须保留边界，避免两条独立链接相连。
_URL_AT_LINE_START_RE = re.compile(
    r"(?:(?:https?|ftp)://|www\.)",
    re.ASCII | re.IGNORECASE,
)


def is_hyphen_at_line_end(line: str) -> bool:
    """判断文本行是否以英文单词的跨行断词符结尾。

    只识别字母后紧跟行末 hyphen 的断词场景，不处理词内连字符或普通破折号。
    """
    return bool(LINE_END_HYPHEN_RE.search(line))


def _url_spans_line_boundary(previous_content: str, next_content: str) -> bool:
    """判断无空格候选中是否存在严格横跨当前物理行边界的 URL。"""
    stripped_previous = previous_content.rstrip()
    stripped_next = next_content.lstrip()
    if not stripped_previous or not stripped_next:
        return False
    candidate = f"{stripped_previous}{stripped_next}"
    boundary = len(stripped_previous)
    return any(match.start() < boundary < match.end() for match in _URL_CANDIDATE_RE.finditer(candidate))


def resolve_text_line_boundary(
    previous_content: str,
    *,
    block_language: str,
    next_content: str,
) -> tuple[str, str]:
    """返回处理后的上一行内容和本次物理行边界分隔符。

    严格横跨边界的 URL 候选直接连接，但下一行自身为完整 URL 时保留空格。
    其余 CJK 文本直接连接物理行，普通西文行插入一个空格。西文行末如果是
    合法的 hyphen，则始终直接连接下一行，并仅在下一行以小写字母开头时删除
    hyphen。
    """
    processed_content = previous_content.rstrip()
    if not processed_content:
        return "", ""
    stripped_next = next_content.lstrip()
    if _url_spans_line_boundary(processed_content, stripped_next):
        if _URL_AT_LINE_START_RE.match(stripped_next):
            return processed_content, " "
        return processed_content, ""
    if block_language in CJK_LANGS:
        return processed_content, ""
    if not is_hyphen_at_line_end(processed_content):
        return processed_content, " "
    if stripped_next and stripped_next[0].islower():
        return processed_content[:-1], ""
    return processed_content, ""


def full_to_half_exclude_marks(text: str) -> str:
    """将全角英文字母和数字转换为半角形式，同时保留全角标点。"""
    result = []
    for char in text:
        code = ord(char)
        # Full-width letters and numbers (FF21-FF3A for A-Z, FF41-FF5A for a-z, FF10-FF19 for 0-9)
        if (0xFF21 <= code <= 0xFF3A) or (0xFF41 <= code <= 0xFF5A) or (0xFF10 <= code <= 0xFF19):
            result.append(chr(code - 0xFEE0))  # Shift to ASCII range
        else:
            result.append(char)
    return "".join(result)


def full_to_half(text: str) -> str:
    """将全角 ASCII 字母、数字和标点统一转换为半角形式。"""
    result = []
    for char in text:
        code = ord(char)
        # Full-width letters, numbers and punctuation (FF01-FF5E)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))  # Shift to ASCII range
        else:
            result.append(char)
    return "".join(result)


def clean_isolated_formula(content: str) -> str:
    """移除行间公式外层的反斜杠方括号并清理首尾空白。"""
    latex = content[:]
    if latex.startswith("\\["):
        latex = latex[2:]
    if latex.endswith("\\]"):
        latex = latex[:-2]
    return latex.strip()


def normalize_formula_tag_content(tag_content: str) -> str:
    """归一化公式编号文本，去掉全角字符和包裹括号后用于 \\tag{}。"""
    tag_content = full_to_half(str(tag_content or "").strip())
    if tag_content.startswith(("(", "﹙")):
        tag_content = tag_content[1:].strip()
    if tag_content.endswith((")", "﹚")):
        tag_content = tag_content[:-1].strip()
    return tag_content


def normalize_formula_content_for_tag(formula_content: str) -> str:
    """归一化待合并编号的公式正文，去掉模型可能携带的展示公式分隔符。"""
    return clean_isolated_formula(str(formula_content or ""))


def build_tagged_formula_content(formula_content: str, tag_content: str) -> str | None:
    """将公式正文和编号文本合成为带 LaTeX tag 的纯公式内容。"""
    formula_content = normalize_formula_content_for_tag(formula_content)
    tag_content = normalize_formula_tag_content(tag_content)
    if not formula_content or not tag_content:
        return None
    return f"{formula_content}\\tag{{{tag_content}}}"
