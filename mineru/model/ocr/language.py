# Copyright (c) Opendatalab. All rights reserved.

from collections.abc import Collection

_CH_LANG_ALIASES = {"en", "japan", "chinese_cht", "latin"}
_ARABIC_LANG_ALIASES = {"ar", "fa", "ug", "ur", "ps", "ku", "sd", "bal"}
_EAST_SLAVIC_LANG_ALIASES = {"ru", "be", "uk"}
_CYRILLIC_LANG_ALIASES = {
    "rs_cyrillic",
    "bg",
    "mn",
    "abq",
    "ady",
    "kbd",
    "ava",
    "dar",
    "inh",
    "che",
    "lbe",
    "lez",
    "tab",
    "kk",
    "ky",
    "tg",
    "mk",
    "tt",
    "cv",
    "ba",
    "mhr",
    "mo",
    "udm",
    "kv",
    "os",
    "bua",
    "xal",
    "tyv",
    "sah",
    "kaa",
}
_DEVANAGARI_LANG_ALIASES = {
    "hi",
    "mr",
    "ne",
    "bh",
    "mai",
    "ang",
    "bho",
    "mah",
    "sck",
    "new",
    "gom",
    "sa",
    "bgc",
}


def normalize_ocr_model_lang(
    lang: str | None,
    *,
    device: str | None = None,
    supported_langs: Collection[str] | None = None,
) -> str:
    """将 OCR 语言参数归一为模型配置 key，保留内部 seal 与语系短码能力。"""
    normalized_lang = lang or "ch"
    if normalized_lang in _CH_LANG_ALIASES:
        normalized_lang = "ch"
    # elif device == "cpu" and normalized_lang == "seal":
    #     normalized_lang = "seal_lite"
    elif normalized_lang in _EAST_SLAVIC_LANG_ALIASES:
        normalized_lang = "east_slavic"
    elif normalized_lang in _ARABIC_LANG_ALIASES:
        normalized_lang = "arabic"
    elif normalized_lang in _CYRILLIC_LANG_ALIASES:
        normalized_lang = "cyrillic"
    elif normalized_lang in _DEVANAGARI_LANG_ALIASES:
        normalized_lang = "devanagari"

    if supported_langs is not None and normalized_lang not in supported_langs:
        raise ValueError(f"Language {lang} not supported")
    return normalized_lang
