# Copyright (c) Opendatalab. All rights reserved.
from typing import Literal, TypeAlias

# 这些字符串不能作为公开 Block.type discriminator，只用于 raw 阶段或 Block 内部枚举值。
RawBlockType: TypeAlias = Literal[
    "algorithm",
    "caption",
    "footnote",
    "phonetic",
]

RAW_ALGORITHM: RawBlockType = "algorithm"
RAW_CAPTION: RawBlockType = "caption"
RAW_FOOTNOTE: RawBlockType = "footnote"
RAW_PHONETIC: RawBlockType = "phonetic"

RAW_ONLY_BLOCK_TYPES = frozenset(
    {
        RAW_ALGORITHM,
        RAW_CAPTION,
        RAW_FOOTNOTE,
        RAW_PHONETIC,
    }
)
