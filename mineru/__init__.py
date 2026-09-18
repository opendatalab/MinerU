# Copyright (c) Opendatalab. All rights reserved.
"""MinerU 公开入口：文档类型即时导入，解析 API 与 SDK 客户端惰性导出。

类型（Block/PageInfo/...）保持 `import mineru` 即可用且不触发依赖加载；
解析入口（parse/MinerUParser/...）与 SDK 客户端（DoclibClient）通过 PEP 562
的显式映射惰性提供，首次访问才导入对应子包（如 mineru.parser、mineru.doclib）。
"""

from typing import TYPE_CHECKING, Any as _Any

from .types import (
    Block,
    CodeInlineSpan,
    EquationInlineSpan,
    HyperlinkSpan,
    InlineSpan,
    MiddleJson,
    ModelJson,
    PageInfo,
    TextSpan,
)

if TYPE_CHECKING:
    from .doclib import DoclibClient
    from .parser import DocumentParser, MinerUApiParser, MinerUParser, ParseResult, parse, parse_async

__all__ = [
    "Block",
    "CodeInlineSpan",
    "DoclibClient",
    "DocumentParser",
    "EquationInlineSpan",
    "HyperlinkSpan",
    "InlineSpan",
    "MiddleJson",
    "MinerUApiParser",
    "MinerUParser",
    "ModelJson",
    "PageInfo",
    "ParseResult",
    "TextSpan",
    "parse",
    "parse_async",
]

# PEP 562 惰性导出的显式映射：名字 -> mineru 包内子模块。静态可查，
# 新增导出时在此登记，不要在模块体顶部直接 import 以免引入重依赖。
_LAZY_EXPORTS: dict[str, str] = {
    "DoclibClient": "doclib",
    "DocumentParser": "parser",
    "MinerUApiParser": "parser",
    "MinerUParser": "parser",
    "ParseResult": "parser",
    "parse": "parser",
    "parse_async": "parser",
}


def __getattr__(name: str) -> _Any:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f".{module}", package=__name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
