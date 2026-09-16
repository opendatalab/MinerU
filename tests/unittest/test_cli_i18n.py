"""mineru / mineru-kit CLI 双语文案:语言检测、t() 语义与词典完整性守卫。"""

from __future__ import annotations

import ast
import string
from pathlib import Path

import pytest

from mineru.cli.telemetry import TELEMETRY_CONSENT_MESSAGE
from mineru.utils import i18n
from mineru.utils.i18n import detect_language, reset_language_cache, t
from mineru.utils.translations import ZH_MESSAGES

REPO_ROOT = Path(__file__).resolve().parents[2]

# 两个 CLI 工具的用户可见表层;gradio(自带 i18n)与 vlm_server 透传服务不在范围内。
SCAN_FILES = sorted(
    {
        *REPO_ROOT.glob("mineru/cli/**/*.py"),
        REPO_ROOT / "mineru/kit/main.py",
        REPO_ROOT / "mineru/kit/common.py",
        *REPO_ROOT.glob("mineru/kit/commands/*.py"),
        REPO_ROOT / "mineru/kit/router/cli.py",
    }
)


def _use_language(monkeypatch: pytest.MonkeyPatch, language: str) -> None:
    monkeypatch.setenv("MINERU_LANG", language)
    reset_language_cache()


class TestDetectLanguage:
    def test_mineru_lang_overrides_everything(self) -> None:
        assert detect_language({"MINERU_LANG": "zh", "LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}) == "zh"
        assert detect_language({"MINERU_LANG": "en", "LC_ALL": "zh_CN.UTF-8"}) == "en"

    def test_posix_priority_chain(self) -> None:
        assert detect_language({"LC_ALL": "zh_CN.UTF-8", "LC_MESSAGES": "en_US.UTF-8", "LANG": "en_US.UTF-8"}) == "zh"
        assert detect_language({"LC_MESSAGES": "zh_TW.UTF-8", "LANG": "en_US.UTF-8"}) == "zh"
        assert detect_language({"LANG": "en_US.UTF-8"}) == "en"

    def test_c_locale_decides_english(self) -> None:
        assert detect_language({"LC_ALL": "C", "LANG": "zh_CN.UTF-8"}) == "en"
        assert detect_language({"LC_ALL": "POSIX", "LANG": "zh_CN.UTF-8"}) == "en"

    def test_c_utf8_is_not_a_language_choice(self) -> None:
        """C.UTF-8 由 IDE/容器注入以保证编码,不代表语言偏好,应跳过继续探测。"""
        assert detect_language({"LC_ALL": "C.UTF-8", "LANG": "zh_CN.UTF-8"}) == "zh"
        assert detect_language({"LANG": "C.UTF-8"}) == "en"

    def test_c_utf8_falls_through_to_macos_system_locale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(i18n.sys, "platform", "darwin")
        monkeypatch.setattr(i18n, "_macos_default_language", lambda: "zh_CN")
        for name in i18n.LANGUAGE_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("LANG", "C.UTF-8")
        reset_language_cache()
        assert detect_language() == "zh"

    def test_chinese_variants(self) -> None:
        for value in ("zh", "zh_CN.UTF-8", "zh_TW", "zh-HK", "zh_Hans@pinyin", "ZH_CN.UTF-8"):
            assert detect_language({"LANG": value}) == "zh", value

    def test_unset_falls_back_to_english(self) -> None:
        assert detect_language({}) == "en"
        assert detect_language({"LANG": ""}) == "en"

    def test_macos_system_locale_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """环境变量全未设置时,macOS 回退系统 UI 语言(IDE 内置终端不设 LANG)。"""
        for name in i18n.LANGUAGE_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(i18n.sys, "platform", "darwin")
        monkeypatch.setattr(i18n, "_macos_default_language", lambda: "zh_Hans_CN")
        reset_language_cache()
        assert detect_language() == "zh"

        monkeypatch.setattr(i18n, "_macos_default_language", lambda: "en_US")
        reset_language_cache()
        assert detect_language() == "en"

    def test_reads_process_env_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINERU_LANG", "zh")
        reset_language_cache()
        try:
            assert t("JSON output") == "JSON 输出"
        finally:
            reset_language_cache()


class TestTranslationFunction:
    def test_english_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_language(monkeypatch, "en")
        assert t("JSON output") == "JSON output"
        assert t("Never translated key.") == "Never translated key."

    def test_chinese_lookup_and_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_language(monkeypatch, "zh")
        assert t("JSON output") == "JSON 输出"
        assert t("Never translated key.") == "Never translated key."

    def test_format_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_language(monkeypatch, "zh")
        assert t("Written to {path}", path="out/a.md") == "已写入 out/a.md"
        _use_language(monkeypatch, "en")
        assert t("Written to {path}", path="out/a.md") == "Written to out/a.md"


class TestCatalogIntegrity:
    def test_every_entry_keeps_placeholders(self) -> None:
        formatter = string.Formatter()
        for key, value in ZH_MESSAGES.items():
            expected = {name for _, name, _, _ in formatter.parse(key) if name is not None}
            actual = {name for _, name, _, _ in formatter.parse(value) if name is not None}
            assert expected == actual, f"占位符不一致: {key!r}"

    def test_telemetry_consent_block_is_translated(self) -> None:
        assert TELEMETRY_CONSENT_MESSAGE in ZH_MESSAGES

    def test_every_t_literal_has_catalog_entry(self) -> None:
        keys = {key for path, key in _iter_t_literals()}
        missing = keys - set(ZH_MESSAGES)
        assert not missing, f"t() 字面量缺少中文词条: {sorted(missing)}"

    def test_no_raw_help_literals(self) -> None:
        violations = [f"{path.relative_to(REPO_ROOT)}:{node.lineno}" for path, node in _iter_raw_help_keywords()]
        assert not violations, f"发现未包 t() 的 help= 字面量: {violations}"


def _iter_ast() -> list[tuple[Path, ast.AST]]:
    trees: list[tuple[Path, ast.AST]] = []
    for path in SCAN_FILES:
        trees.append((path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))))
    return trees


def _iter_t_literals() -> list[tuple[Path, str]]:
    literals: list[tuple[Path, str]] = []
    for path, tree in _iter_ast():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Name) and node.func.id == "t"):
                continue
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                literals.append((path, node.args[0].value))
    return literals


def _iter_raw_help_keywords() -> list[tuple[Path, ast.keyword]]:
    hits: list[tuple[Path, ast.keyword]] = []
    for path, tree in _iter_ast():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != "help" or not isinstance(keyword.value, ast.Constant):
                    continue
                if isinstance(keyword.value.value, str):
                    hits.append((path, keyword))
    return hits
