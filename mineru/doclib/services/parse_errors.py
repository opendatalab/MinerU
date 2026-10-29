"""Doclib 解析请求与执行共享的稳定失败类型。"""

from __future__ import annotations

from ...errors import MineruError


class ParseFailure(MineruError):
    """记录解析失败的稳定错误信息，供队列和执行路径共同使用。"""

    def __init__(self, code: str, message: str, param: str | None = None) -> None:
        """保存稳定错误码、消息和可选参数路径。"""
        super().__init__(code, message, param)
        self.message = message


__all__ = ["ParseFailure"]
