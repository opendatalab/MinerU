"""使普通回归测试独立于开发者本地的可选 LLM 连接配置。"""

from __future__ import annotations

import importlib

import pytest


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
