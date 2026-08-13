# Copyright (c) Opendatalab. All rights reserved.
"""raw block 的文本、代码和公式内容清理规则。"""

from __future__ import annotations

import re


def code_content_clean(content: str | None) -> str:
    """去除代码块外层 Markdown 围栏并保留代码正文。"""
    if not content:
        return ""
    lines = content.splitlines()
    start_idx = 1 if lines and lines[0].startswith("```") else 0
    end_idx = len(lines)
    if lines and end_idx > start_idx and lines[end_idx - 1].strip() == "```":
        end_idx -= 1
    if start_idx < end_idx:
        return "\n".join(lines[start_idx:end_idx]).strip()
    return ""


def clean_content(content: str | None) -> str | None:
    """将成对的行间公式分隔符改为兼容文本清理的方括号。"""
    if content and content.count("\\[") == content.count("\\]") and content.count("\\[") > 0:

        def replace_pattern(match: re.Match[str]) -> str:
            """替换单个成对公式片段。"""
            return f"[{match.group(1)}]"

        content = re.sub(r"\\\[(.*?)\\\]", replace_pattern, content)
    return content
