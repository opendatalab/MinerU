# Copyright (c) Opendatalab. All rights reserved.

class BlockType:
    TEXT = "text"  # 文本
    TITLE = "title"  # 段落标题
    TABLE = "table"  # 表格
    IMAGE = "image"  # 图像
    HEADER = "header"  # 页眉
    FOOTER = "footer"  # 页脚
    EQUATION = "equation"  # 公式(独立公式)
    LIST = "list"  # 列表块(无序/有序列表)

    # captions
    TABLE_CAPTION = "table_caption"  # 表格标题
    IMAGE_CAPTION = "image_caption"  # 图像标题

    UNKNOWN = "unknown"  # 未知块

BLOCK_TYPES = set(
    [
        BlockType.TEXT,
        BlockType.TITLE,
        BlockType.TABLE,
        BlockType.IMAGE,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.EQUATION,
        BlockType.TABLE_CAPTION,
        BlockType.IMAGE_CAPTION,
        BlockType.LIST,
        BlockType.UNKNOWN,
    ]
)


class ContentBlock(dict):
    def __init__(
        self,
        type: str,
        content: str | None = None,
    ):
        """
        Initialize a layout block.
        Args:
            type (str): Type of the block (e.g., 'text', 'image', 'table').
            content (str or None): The content of the block (if exists).
        """
        super().__init__()

        assert type in BLOCK_TYPES, f"Unknown type: {type}"
        assert content is None or isinstance(content, str), "Content must be a string or None"

        self["type"] = type
        self["content"] = content

    @property
    def type(self) -> str:
        return self["type"]

    @type.setter
    def type(self, value: str):
        assert value in BLOCK_TYPES, f"Unknown type: {value}"
        self["type"] = value

    @property
    def content(self) -> str | None:
        return self["content"]

    @content.setter
    def content(self, value: str | None):
        assert value is None or isinstance(value, str), "Content must be a string or None"
        self["content"] = value