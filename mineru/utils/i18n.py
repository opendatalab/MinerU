# Copyright (c) Opendatalab. All rights reserved.
"""CLI 双语文案:按系统语言自动选择英文或简体中文。

英文原文即词条 key,漏译时回退英文原文,不会报错。语言在首次使用时
通过环境变量检测并缓存;Typer 的 help 文案在装饰期求值,属于已知的
只读查询例外(见 AGENTS.md 副作用隔离一节的实现取舍)。
"""

from __future__ import annotations

import locale
import os
import sys
from collections.abc import Mapping
from typing import Any, Literal

from .translations import ZH_MESSAGES

Language = Literal["en", "zh"]

# POSIX 语义:LC_ALL 覆盖 LC_MESSAGES 覆盖 LANG;MINERU_LANG 是产品级强制覆盖。
LANGUAGE_ENV_VARS = ("MINERU_LANG", "LC_ALL", "LC_MESSAGES", "LANG")

_cached_language: Language | None = None


def _normalize_locale_code(raw: str) -> str:
    code = raw.split("@", 1)[0].split(".", 1)[0]
    return code.strip().lower().replace("-", "_")


def _windows_default_language() -> str:
    import ctypes

    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return ""
    try:
        lang_id = windll.kernel32.GetUserDefaultUILanguage()
    except Exception:  # noqa: BLE001 — Windows API 探测失败时静默回退英文
        return ""
    return locale.windows_locale.get(int(lang_id), "")


def detect_language(env: Mapping[str, str] | None = None) -> Language:
    """按 MINERU_LANG > LC_ALL > LC_MESSAGES > LANG 检测界面语言,纯函数。

    第一个非空变量即决定结果(``C``/``POSIX`` 也视为该层决定,结果为英文);
    全部未设置时 Windows 上回退到系统 UI 语言,其余平台为英文。
    """
    source: Mapping[str, str] = os.environ if env is None else env
    for name in LANGUAGE_ENV_VARS:
        raw = source.get(name, "").strip()
        if not raw:
            continue
        if _normalize_locale_code(raw).startswith("zh"):
            return "zh"
        return "en"
    if sys.platform == "win32" and env is None:
        if _normalize_locale_code(_windows_default_language()).startswith("zh"):
            return "zh"
    return "en"


def current_language() -> Language:
    global _cached_language
    if _cached_language is None:
        _cached_language = detect_language()
    return _cached_language


def reset_language_cache() -> None:
    global _cached_language
    _cached_language = None


def t(key: str, /, **fmt: Any) -> str:
    """返回 key 的当前语言文案;未收录时原样返回英文 key,有 fmt 时做 format。"""
    text = ZH_MESSAGES.get(key, key) if current_language() == "zh" else key
    return text.format(**fmt) if fmt else text
