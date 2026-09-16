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


def _locale_language_choice(raw: str) -> Language | None:
    """解析一个 locale 值的语言偏好;返回 None 表示该值不表达语言偏好。

    ``C``/``POSIX``(不带编码后缀)按 POSIX 语义视为明确选择英文;
    ``C.UTF-8`` 这类带编码后缀的值通常由 IDE 内置终端或容器注入以保证
    UTF-8 输出,不代表语言选择,返回 None 让后续探测继续。
    """
    modifier_stripped = raw.split("@", 1)[0]
    language_part, _, encoding = modifier_stripped.partition(".")
    language = language_part.strip().lower().replace("-", "_")
    if not language:
        return None
    if language in ("c", "posix"):
        return None if encoding else "en"
    return "zh" if language.startswith("zh") else "en"


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


# kCFStringEncodingUTF8
_CF_STRING_ENCODING_UTF8 = 0x08000100


def _macos_default_language() -> str:
    import ctypes
    import ctypes.util

    cf_path = ctypes.util.find_library("CoreFoundation")
    if cf_path is None:
        return ""
    cf = ctypes.cdll.LoadLibrary(cf_path)
    cf.CFLocaleCopyCurrent.restype = ctypes.c_void_p
    cf.CFLocaleGetIdentifier.restype = ctypes.c_void_p
    cf.CFLocaleGetIdentifier.argtypes = [ctypes.c_void_p]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    cf.CFRelease.argtypes = [ctypes.c_void_p]

    locale_ref = cf.CFLocaleCopyCurrent()
    if not locale_ref:
        return ""
    try:
        identifier_ref = cf.CFLocaleGetIdentifier(locale_ref)
        if not identifier_ref:
            return ""
        buffer = ctypes.create_string_buffer(256)
        if not cf.CFStringGetCString(identifier_ref, buffer, len(buffer), _CF_STRING_ENCODING_UTF8):
            return ""
        return buffer.value.decode("utf-8", errors="replace")
    finally:
        cf.CFRelease(locale_ref)


def detect_language(env: Mapping[str, str] | None = None) -> Language:
    """按 MINERU_LANG > LC_ALL > LC_MESSAGES > LANG 检测界面语言,纯函数。

    第一个表达了语言偏好的变量即决定结果(裸 ``C``/``POSIX`` 视为该层
    决定英文;``C.UTF-8`` 不表达偏好,跳过);全部无偏好/未设置时
    Windows/macOS 回退到系统 UI 语言(IDE 内置终端常不设置 LANG),
    其余平台为英文。
    """
    source: Mapping[str, str] = os.environ if env is None else env
    for name in LANGUAGE_ENV_VARS:
        raw = source.get(name, "").strip()
        if not raw:
            continue
        choice = _locale_language_choice(raw)
        if choice is not None:
            return choice
    if env is None:
        system_language = _windows_default_language() if sys.platform == "win32" else ""
        if sys.platform == "darwin":
            system_language = _macos_default_language()
        if _normalize_locale_code(system_language).startswith("zh"):
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
