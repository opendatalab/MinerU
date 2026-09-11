# Copyright (c) Opendatalab. All rights reserved.
"""PDF 编排使用的显式 VLM 抽取协议。"""

from __future__ import annotations
from typing import TYPE_CHECKING, Protocol
from collections.abc import Sequence

if TYPE_CHECKING:
    from PIL.Image import Image
    from mineru_vl_utils.structs import ContentBlock, ExtractResult


class VlmPredictor(Protocol):
    """同步客户端与原生异步代理共同实现的最小文档抽取接口。"""

    def batch_extract_with_layout(
        self,
        images: list[Image],
        blocks_list: Sequence[Sequence[ContentBlock | dict]],
        *,
        not_extract_list: list[str] | None = None,
        image_analysis: bool | None = None,
    ) -> list[ExtractResult]:
        """同步入口通过常驻循环复用原生异步外部布局抽取。"""
        ...

    async def aio_batch_extract_with_layout(
        self,
        images: list[Image],
        blocks_list: Sequence[Sequence[ContentBlock | dict]],
        *,
        not_extract_list: list[str] | None = None,
        image_analysis: bool | None = None,
    ) -> list[ExtractResult]:
        """异步入口直接等待原生异步外部布局抽取。"""
        ...

    def batch_two_step_extract(
        self,
        images: list[Image],
        *,
        not_extract_list: list[str] | None = None,
        image_analysis: bool | None = None,
    ) -> list[ExtractResult]:
        """同步等待共享异步引擎完成两阶段抽取。"""
        ...

    async def aio_batch_two_step_extract(
        self,
        images: list[Image],
        *,
        not_extract_list: list[str] | None = None,
        image_analysis: bool | None = None,
    ) -> list[ExtractResult]:
        """异步等待共享引擎完成两阶段抽取。"""
        ...


__all__ = ["VlmPredictor"]
