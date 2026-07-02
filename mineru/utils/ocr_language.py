# Copyright (c) Opendatalab. All rights reserved.

PUBLIC_OCR_LANGUAGES = (
    "ch",
    "ch_server",
    "korean",
    "ta",
    "te",
    "ka",
    "th",
    "el",
    "arabic",
    "east_slavic",
    "cyrillic",
    "devanagari",
)

_PUBLIC_OCR_LANGUAGE_DESCRIPTIONS = {
    "ch": "Chinese, English, Japanese, Chinese Traditional, Latin",
    "ch_server": "Chinese, English, Japanese, Chinese Traditional, Latin",
    "korean": "Korean, English",
    "ta": "Tamil, English",
    "te": "Telugu, English",
    "ka": "Kannada",
    "th": "Thai, English",
    "el": "Greek, English",
    "arabic": (
        "Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi, English"
    ),
    "east_slavic": "Russian, Belarusian, Ukrainian, English",
    "cyrillic": (
        "Russian, Belarusian, Ukrainian, Serbian (Cyrillic), Bulgarian, Mongolian, "
        "Abkhazian, Adyghe, Kabardian, Avar, Dargin, Ingush, Chechen, Lak, Lezgin, "
        "Tabasaran, Kazakh, Kyrgyz, Tajik, Macedonian, Tatar, Chuvash, Bashkir, "
        "Malian, Moldovan, Udmurt, Komi, Ossetian, Buryat, Kalmyk, Tuvan, Sakha, "
        "Karakalpak, English"
    ),
    "devanagari": (
        "Hindi, Marathi, Nepali, Bihari, Maithili, Angika, Bhojpuri, Magahi, "
        "Santali, Newari, Konkani, Sanskrit, Haryanvi, English"
    ),
}

PUBLIC_OCR_LANGUAGE_CHOICES = tuple(
    f"{lang} ({_PUBLIC_OCR_LANGUAGE_DESCRIPTIONS[lang]})"
    for lang in PUBLIC_OCR_LANGUAGES
)

PUBLIC_OCR_LANGUAGE_SCHEMA_EXTRA = {"items": {"enum": list(PUBLIC_OCR_LANGUAGES)}}

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


def format_public_ocr_lang_description() -> str:
    """生成公开 API 使用的 OCR 语言说明，避免入口文案各自维护。"""
    option_lines = [
        f"- {lang}: {_PUBLIC_OCR_LANGUAGE_DESCRIPTIONS[lang]}."
        for lang in PUBLIC_OCR_LANGUAGES
    ]
    return (
        "(Adapted for local Hybrid OCR) Input the languages in the pdf "
        "to improve OCR accuracy. Options:\n"
        + "\n".join(option_lines)
    )


def validate_public_ocr_lang(lang: str) -> str:
    """校验公开入口允许的 OCR 语言，并将隐藏兼容别名归一到 ch。"""
    normalized_lang = "ch" if lang in _CH_LANG_ALIASES else lang
    if normalized_lang not in PUBLIC_OCR_LANGUAGES:
        raise ValueError(
            f"Language {lang} not supported. Allowed values: "
            + ", ".join(PUBLIC_OCR_LANGUAGES)
        )
    return normalized_lang


def validate_public_ocr_lang_list(lang_list: list[str]) -> list[str]:
    """校验公开 API 的语言列表，返回可安全传入下游的副本。"""
    effective_lang_list = lang_list or ["ch"]
    return [validate_public_ocr_lang(lang) for lang in effective_lang_list]


def normalize_ocr_model_lang(
    lang: str | None,
    *,
    device: str | None = None,
    supported_langs=None,
) -> str:
    """将 OCR 语言参数归一为模型配置 key，保留内部 seal 与语系短码能力。"""
    normalized_lang = lang or "ch"
    if normalized_lang in _CH_LANG_ALIASES:
        normalized_lang = "ch"
    elif device == "cpu" and normalized_lang == "seal":
        normalized_lang = "seal_lite"
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
