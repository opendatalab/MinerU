"""使普通回归测试独立于开发者本地的可选 LLM 连接配置。"""

from __future__ import annotations

import importlib
import os

import pytest

# Typer 的 help 文案在 import 期求值,必须在任何 mineru 模块导入前固定 CLI 语言,
# 否则中文 locale 机器上既有的英文输出断言会失败。i18n 专项测试自行 monkeypatch 覆盖。
os.environ["MINERU_LANG"] = "en"

# GitHub Actions 设置 GITHUB_ACTIONS=true 会使 typer/rich 强制彩色输出,
# OptionHighlighter 会把 "--option" 拆成多段 ANSI 样式,破坏测试的子串断言;
# 在导入 typer 前统一关闭终端强制模式,保证跨环境输出确定。
os.environ.setdefault("_TYPER_FORCE_DISABLE_TERMINAL", "1")


@pytest.fixture(autouse=True)
def pin_cli_language() -> None:
    """每个测试前重置语言缓存,保证运行期 t() 调用也固定为英文。"""
    from mineru.utils import i18n

    i18n.reset_language_cache()


@pytest.fixture(autouse=True)
def isolate_optional_llm_configuration(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """普通测试默认关闭外部增强；需要增强的测试仍可显式配置其替身。"""
    if request.node.get_closest_marker("remote") or request.node.get_closest_marker("full_stack"):
        return
    module = importlib.import_module("mineru.config")
    configuration = module.config.llm_aided.model_copy(deep=True)
    configuration.features.title_leveling = False
    configuration.features.cross_page_table_cell_merge = False
    monkeypatch.setattr(module.config, "llm_aided", configuration)
