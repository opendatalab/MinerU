# Copyright (c) Opendatalab. All rights reserved.
"""SDK 入口契约：顶层惰性导出与 doclib 公共面。"""

import subprocess
import sys

import pytest

import mineru


LAZY_PARSER_EXPORTS = ("DocumentParser", "MinerUApiParser", "MinerUParser", "ParseResult", "parse", "parse_async")


@pytest.mark.parametrize("name", LAZY_PARSER_EXPORTS)
def test_top_level_parser_exports_resolve_to_parser_package(name: str) -> None:
    """顶层解析 API 惰性导出解析到 mineru.parser 的同一符号。"""
    import mineru.parser

    assert getattr(mineru, name) is getattr(mineru.parser, name)


def test_top_level_doclib_client_is_lazy_and_correct() -> None:
    """mineru.DoclibClient 经惰性映射解析到 doclib.client 的同一个类。"""
    from mineru.doclib.client import DoclibClient

    assert mineru.DoclibClient is DoclibClient
    # 首次访问后缓存进 globals，二次访问是同一个对象。
    assert mineru.DoclibClient is mineru.DoclibClient


def test_dir_includes_lazy_exports() -> None:
    assert "DoclibClient" in dir(mineru)


def test_lazy_name_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        mineru.NotARealExport  # noqa: B018


def test_import_mineru_stays_dependency_free() -> None:
    """`import mineru` 不得拉起 httpx/doclib 等依赖（惰性导出的核心约束）。"""
    code = "import mineru, sys; assert 'httpx' not in sys.modules, 'import mineru pulled httpx'"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_doclib_exports_are_importable() -> None:
    """doclib 公共面的每个 __all__ 名字都可导入。"""
    from mineru import doclib

    for name in doclib.__all__:
        assert getattr(doclib, name) is not None
