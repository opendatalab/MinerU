# Copyright (c) Opendatalab. All rights reserved.
from typing import Literal, TypeAlias

# 这些类型只允许存在于 Analyze/MagicModel 的 raw dict 阶段。
RawBlockType: TypeAlias = Literal[
    "algorithm",
    "caption",
    "footnote",
    "title",
    "phonetic",
]

RAW_ALGORITHM: RawBlockType = "algorithm"
RAW_CAPTION: RawBlockType = "caption"
RAW_FOOTNOTE: RawBlockType = "footnote"
RAW_TITLE: RawBlockType = "title"
RAW_PHONETIC: RawBlockType = "phonetic"

RAW_ONLY_BLOCK_TYPES = frozenset(
    {RAW_ALGORITHM, RAW_CAPTION, RAW_FOOTNOTE, RAW_TITLE, RAW_PHONETIC}
)
