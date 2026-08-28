# Copyright (c) Opendatalab. All rights reserved.
"""OFD 二进制流到分页 raw model-list 的转换入口。"""

from __future__ import annotations

from typing import BinaryIO

from .constants import MAX_TOTAL_BYTES
from .errors import OfdResourceLimitError
from .package import OfdPackage
from .reading_order import OfdReadingOrderProjector
from .scene import OfdSceneBuilder


class OfdConverter:
    """编排 OFD 包读取、场景构建和阅读顺序投影。"""

    def __init__(self) -> None:
        """初始化空的分页输出。"""
        self.pages: list[list[dict[str, object]]] = []

    def convert(self, file_binary: BinaryIO) -> None:
        """读取整份 OFD 并更新分页 model-list。"""
        file_bytes = file_binary.read(MAX_TOTAL_BYTES + 1)
        if len(file_bytes) > MAX_TOTAL_BYTES:
            raise OfdResourceLimitError(f"OFD resource limit exceeded: max_total_bytes={MAX_TOTAL_BYTES}")
        with OfdPackage(file_bytes) as package:
            scenes = OfdSceneBuilder(package).build()
            self.pages = OfdReadingOrderProjector(scenes).project()


__all__ = ["OfdConverter"]
