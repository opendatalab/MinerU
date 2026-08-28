# Copyright (c) Opendatalab. All rights reserved.
"""HTML Flash 解析的稳定异常类型。"""


class HtmlParseError(ValueError):
    """表示 HTML 字节无法形成可用静态 DOM。"""


class HtmlResourceLimitError(HtmlParseError):
    """表示 HTML 输入、DOM 或资源超过固定安全预算。"""


__all__ = ["HtmlParseError", "HtmlResourceLimitError"]
