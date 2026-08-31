from __future__ import annotations

from typing import Any


def inline(content: str, *, styles: list[str] | None = None) -> list[dict[str, Any]]:
    """把测试文字构造成最小 Middle JSON 2.0 TextSpan 列表。"""
    span: dict[str, Any] = {"type": "text", "content": content}
    if styles:
        span["styles"] = styles
    return [span]


def equation(latex: str) -> dict[str, str]:
    """构造测试用行内公式 Span。"""
    return {"type": "equation_inline", "content": latex}


def code(content: str) -> dict[str, str]:
    """构造测试用行内代码 Span。"""
    return {"type": "code_inline", "content": content}


def hyperlink(url: str, content: str) -> dict[str, Any]:
    """构造测试用超链接 Span。"""
    return {"type": "hyperlink", "url": url, "content": inline(content)}


def inline_text(spans: Any) -> str:
    """提取 raw 或 Pydantic InlineSpan 序列的可见文字，供断言复用。"""
    if not isinstance(spans, list):
        return ""
    parts: list[str] = []
    for span in spans:
        if isinstance(span, dict):
            span_type = span.get("type")
            content = span.get("content")
        else:
            span_type = getattr(span, "type", None)
            content = getattr(span, "content", None)
        if str(span_type) == "hyperlink":
            parts.append(inline_text(content))
        elif isinstance(content, str):
            parts.append(content)
    return "".join(parts)


def inline_items(spans: Any) -> list[Any]:
    """深度优先展开 InlineSpan，保留 HyperlinkSpan 本身及其子 Span。"""
    if not isinstance(spans, list):
        return []
    output: list[Any] = []
    for span in spans:
        output.append(span)
        content = span.get("content") if isinstance(span, dict) else getattr(span, "content", None)
        span_type = span.get("type") if isinstance(span, dict) else getattr(span, "type", None)
        if str(span_type) == "hyperlink":
            output.extend(inline_items(content))
    return output


def inline_urls(spans: Any) -> list[str]:
    """提取 InlineSpan 序列中的全部超链接目标。"""
    urls: list[str] = []
    for span in inline_items(spans):
        span_type = span.get("type") if isinstance(span, dict) else getattr(span, "type", None)
        url = span.get("url") if isinstance(span, dict) else getattr(span, "url", None)
        if str(span_type) == "hyperlink" and isinstance(url, str):
            urls.append(url)
    return urls


def visible_content(content: Any) -> str:
    """统一提取专用字符串内容或 InlineSpan 内容的可见文字。"""
    return content if isinstance(content, str) else inline_text(content)


__all__ = [
    "code",
    "equation",
    "hyperlink",
    "inline",
    "inline_items",
    "inline_text",
    "inline_urls",
    "visible_content",
]
