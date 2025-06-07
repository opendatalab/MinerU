"""MinerU支持的语言列表。"""

from typing import Dict, List

# 支持的语言列表
LANGUAGES: List[Dict[str, str]] = [
    {"name": "中文", "description": "Chinese & English", "code": "ch"},
    {"name": "英文", "description": "English", "code": "en"},
    {"name": "法文", "description": "French", "code": "fr"},
    {"name": "德文", "description": "German", "code": "german"},
    {"name": "日文", "description": "Japanese", "code": "japan"},
    {"name": "韩文", "description": "Korean", "code": "korean"},
    {"name": "中文繁体", "description": "Chinese Traditional", "code": "chinese_cht"},
    {"name": "意大利文", "description": "Italian", "code": "it"},
    {"name": "西班牙文", "description": "Spanish", "code": "es"},
    {"name": "葡萄牙文", "description": "Portuguese", "code": "pt"},
    {"name": "俄罗斯文", "description": "Russian", "code": "ru"},
    {"name": "阿拉伯文", "description": "Arabic", "code": "ar"},
    {"name": "印地文", "description": "Hindi", "code": "hi"},
    {"name": "维吾尔", "description": "Uyghur", "code": "ug"},
    {"name": "波斯文", "description": "Persian", "code": "fa"},
    {"name": "乌尔都文", "description": "Urdu", "code": "ur"},
    {"name": "塞尔维亚文（latin)", "description": "Serbian(latin)", "code": "rs_latin"},
    {"name": "欧西坦文", "description": "Occitan", "code": "oc"},
    {"name": "马拉地文", "description": "Marathi", "code": "mr"},
    {"name": "尼泊尔文", "description": "Nepali", "code": "ne"},
    {
        "name": "塞尔维亚文（cyrillic)",
        "description": "Serbian(cyrillic)",
        "code": "rs_cyrillic",
    },
    {"name": "毛利文", "description": "Maori", "code": "mi"},
    {"name": "马来文", "description": "Malay", "code": "ms"},
    {"name": "马耳他文", "description": "Maltese", "code": "mt"},
    {"name": "荷兰文", "description": "Dutch", "code": "nl"},
    {"name": "挪威文", "description": "Norwegian", "code": "no"},
    {"name": "波兰文", "description": "Polish", "code": "pl"},
    {"name": "罗马尼亚文", "description": "Romanian", "code": "ro"},
    {"name": "斯洛伐克文", "description": "Slovak", "code": "sk"},
    {"name": "斯洛文尼亚文", "description": "Slovenian", "code": "sl"},
    {"name": "阿尔巴尼亚文", "description": "Albanian", "code": "sq"},
    {"name": "瑞典文", "description": "Swedish", "code": "sv"},
    {"name": "西瓦希里文", "description": "Swahili", "code": "sw"},
    {"name": "塔加洛文", "description": "Tagalog", "code": "tl"},
    {"name": "土耳其文", "description": "Turkish", "code": "tr"},
    {"name": "乌兹别克文", "description": "Uzbek", "code": "uz"},
    {"name": "越南文", "description": "Vietnamese", "code": "vi"},
    {"name": "蒙古文", "description": "Mongolian", "code": "mn"},
    {"name": "车臣文", "description": "Chechen", "code": "che"},
    {"name": "哈里亚纳语", "description": "Haryanvi", "code": "bgc"},
    {"name": "保加利亚文", "description": "Bulgarian", "code": "bg"},
    {"name": "乌克兰文", "description": "Ukranian", "code": "uk"},
    {"name": "白俄罗斯文", "description": "Belarusian", "code": "be"},
    {"name": "泰卢固文", "description": "Telugu", "code": "te"},
    {"name": "阿巴扎文", "description": "Abaza", "code": "abq"},
    {"name": "泰米尔文", "description": "Tamil", "code": "ta"},
    {"name": "南非荷兰文", "description": "Afrikaans", "code": "af"},
    {"name": "阿塞拜疆文", "description": "Azerbaijani", "code": "az"},
    {"name": "波斯尼亚文", "description": "Bosnian", "code": "bs"},
    {"name": "捷克文", "description": "Czech", "code": "cs"},
    {"name": "威尔士文", "description": "Welsh", "code": "cy"},
    {"name": "丹麦文", "description": "Danish", "code": "da"},
    {"name": "爱沙尼亚文", "description": "Estonian", "code": "et"},
    {"name": "爱尔兰文", "description": "Irish", "code": "ga"},
    {"name": "克罗地亚文", "description": "Croatian", "code": "hr"},
    {"name": "匈牙利文", "description": "Hungarian", "code": "hu"},
    {"name": "印尼文", "description": "Indonesian", "code": "id"},
    {"name": "冰岛文", "description": "Icelandic", "code": "is"},
    {"name": "库尔德文", "description": "Kurdish", "code": "ku"},
    {"name": "立陶宛文", "description": "Lithuanian", "code": "lt"},
    {"name": "拉脱维亚文", "description": "Latvian", "code": "lv"},
    {"name": "达尔瓦文", "description": "Dargwa", "code": "dar"},
    {"name": "因古什文", "description": "Ingush", "code": "inh"},
    {"name": "拉克文", "description": "Lak", "code": "lbe"},
    {"name": "莱兹甘文", "description": "Lezghian", "code": "lez"},
    {"name": "塔巴萨兰文", "description": "Tabassaran", "code": "tab"},
    {"name": "比尔哈文", "description": "Bihari", "code": "bh"},
    {"name": "迈蒂利文", "description": "Maithili", "code": "mai"},
    {"name": "昂加文", "description": "Angika", "code": "ang"},
    {"name": "孟加拉文", "description": "Bhojpuri", "code": "bho"},
    {"name": "摩揭陀文", "description": "Magahi", "code": "mah"},
    {"name": "那格浦尔文", "description": "Nagpur", "code": "sck"},
    {"name": "尼瓦尔文", "description": "Newari", "code": "new"},
    {"name": "保加利亚文", "description": "Goan Konkani", "code": "gom"},
    {"name": "梵文", "description": "Sanskrit", "code": "sa"},
    {"name": "阿瓦尔文", "description": "Avar", "code": "ava"},
    {"name": "阿瓦尔文", "description": "Avar", "code": "ava"},
    {"name": "阿迪赫文", "description": "Adyghe", "code": "ady"},
    {"name": "巴利文", "description": "Pali", "code": "pi"},
    {"name": "拉丁文", "description": "Latin", "code": "la"},
]

# 构建语言代码到语言信息的映射字典，便于快速查找
LANGUAGES_DICT: Dict[str, Dict[str, str]] = {lang["code"]: lang for lang in LANGUAGES}


def get_language_list() -> List[Dict[str, str]]:
    """获取所有支持的语言列表。"""
    return LANGUAGES


def get_language_by_code(code: str) -> Dict[str, str]:
    """根据语言代码获取语言信息。"""
    return LANGUAGES_DICT.get(
        code, {"name": "未知", "description": "Unknown", "code": code}
    )
