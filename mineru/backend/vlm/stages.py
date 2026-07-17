# Copyright (c) Opendatalab. All rights reserved.
"""VLM后端两阶段（layout检测 / 内容识别）的可插拔接口层。

两阶段之间的交换格式为每页一组 ``mineru_vl_utils.structs.ContentBlock``
（dict子类，bbox为0-1归一化坐标），可经 layout doc JSON 序列化/反序列化，
从而支持两阶段分开独立运行。
"""
import asyncio
import json
import os

from loguru import logger

from mineru_vl_utils import MinerUClient
from mineru_vl_utils.structs import ContentBlock

LAYOUT_DOC_VERSION = 1
DEFAULT_LAYOUT_DOC_FILENAME = "layout.json"

# ContentBlock 构造函数直接接受的键，其余键（如 sub_type/cell_merge）需逐项赋值
_BLOCK_CTOR_KEYS = ("type", "bbox", "angle", "content", "merge_prev")


class LayoutDetector:
    """第一阶段：页面图片 -> 每页一组layout块。"""

    # 检测器会输出 formula_number 块时置True（如pipeline的PP-DocLayoutV2）；
    # 解析主流程会据此将编号块合并进相邻公式，vlm的MagicModel自身不处理该类型。
    emits_formula_number: bool = False
    name: str = "custom"

    def batch_detect(self, images, start_page_idx: int = 0) -> list[list[ContentBlock]]:
        raise NotImplementedError

    async def aio_batch_detect(self, images, start_page_idx: int = 0) -> list[list[ContentBlock]]:
        return await asyncio.to_thread(self.batch_detect, images, start_page_idx)


class ContentRecognizer:
    """第二阶段：页面图片 + layout块 -> 填充识别内容后的块。"""

    name: str = "custom"

    def batch_recognize(
        self,
        images,
        blocks_list,
        image_analysis: bool = True,
    ) -> list[list[ContentBlock]]:
        raise NotImplementedError

    async def aio_batch_recognize(
        self,
        images,
        blocks_list,
        image_analysis: bool = True,
    ) -> list[list[ContentBlock]]:
        return await asyncio.to_thread(self.batch_recognize, images, blocks_list, image_analysis)


class VlmLayoutDetector(LayoutDetector):
    """默认layout实现：MinerU VLM 自身的layout检测步骤。"""

    name = "mineru-vlm"

    def __init__(self, predictor: MinerUClient):
        self.predictor = predictor

    def batch_detect(self, images, start_page_idx: int = 0):
        return self.predictor.batch_layout_detect(images)

    async def aio_batch_detect(self, images, start_page_idx: int = 0):
        return await self.predictor.aio_batch_layout_detect(images)


class VlmContentRecognizer(ContentRecognizer):
    """默认识别实现：MinerU VLM 基于外部layout做逐块识别。"""

    name = "mineru-vlm"

    def __init__(self, predictor: MinerUClient):
        self.predictor = predictor

    def batch_recognize(self, images, blocks_list, image_analysis: bool = True):
        return self.predictor.batch_extract_with_layout(
            images,
            blocks_list,
            image_analysis=image_analysis,
        )

    async def aio_batch_recognize(self, images, blocks_list, image_analysis: bool = True):
        return await self.predictor.aio_batch_extract_with_layout(
            images,
            blocks_list,
            image_analysis=image_analysis,
        )


def block_to_jsonable(block) -> dict:
    """ContentBlock（dict子类）-> 可JSON化的普通dict。"""
    return dict(block)


def block_from_jsonable(data: dict) -> ContentBlock | None:
    """普通dict -> ContentBlock，非法块跳过并告警（与hybrid的容错策略一致）。"""
    try:
        block = ContentBlock(
            data["type"],
            [float(v) for v in data["bbox"]],
            angle=data.get("angle"),
            content=data.get("content"),
            merge_prev=bool(data.get("merge_prev", False)),
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        logger.warning(f"Skip invalid layout block: {data}, error: {exc}")
        return None
    for key, value in data.items():
        if key in _BLOCK_CTOR_KEYS:
            continue
        try:
            block[key] = value
        except Exception as exc:
            logger.warning(f"Skip layout block extra field {key!r}: {exc}")
    return block


def blocks_from_jsonable(blocks_data) -> list[ContentBlock]:
    blocks = []
    for data in blocks_data or []:
        block = block_from_jsonable(data)
        if block is not None:
            blocks.append(block)
    return blocks


def build_layout_doc(
    pages: list[dict],
    layout_backend: str = "custom",
    emits_formula_number: bool = False,
) -> dict:
    """将各页layout结果组装为layout doc（两阶段间的标准中间结果）。

    pages 元素: {"page_idx": int, "page_size": [w, h], "blocks": list[ContentBlock|dict]}
    """
    return {
        "version": LAYOUT_DOC_VERSION,
        "layout_backend": layout_backend,
        "page_count": len(pages),
        "emits_formula_number": bool(emits_formula_number),
        "pages": [
            {
                "page_idx": int(page["page_idx"]),
                "page_size": list(page["page_size"]),
                "blocks": [block_to_jsonable(block) for block in page["blocks"]],
            }
            for page in pages
        ],
    }


def layout_doc_to_json(layout_doc: dict) -> str:
    return json.dumps(layout_doc, ensure_ascii=False, indent=2)


def load_layout_doc(source) -> dict:
    """加载layout doc：接受dict、JSON字符串/bytes或文件路径。"""
    if isinstance(source, dict):
        layout_doc = source
    elif isinstance(source, bytes):
        layout_doc = json.loads(source)
    elif isinstance(source, (str, os.PathLike)):
        text = os.fspath(source) if isinstance(source, os.PathLike) else source
        if text.lstrip().startswith("{"):
            layout_doc = json.loads(text)
        else:
            with open(text, "r", encoding="utf-8") as f:
                layout_doc = json.load(f)
    else:
        raise TypeError(f"Unsupported layout doc source type: {type(source)}")

    version = layout_doc.get("version")
    if version != LAYOUT_DOC_VERSION:
        raise ValueError(f"Unsupported layout doc version: {version}, expected {LAYOUT_DOC_VERSION}")
    if not isinstance(layout_doc.get("pages"), list):
        raise ValueError("Invalid layout doc: missing 'pages' list")
    return layout_doc


class PrecomputedLayoutDetector(LayoutDetector):
    """从已落盘的layout doc读取layout —— 即"只跑第二阶段"的入口。

    用法：doc_analyze(..., layout_detector=PrecomputedLayoutDetector("layout.json"))
    """

    def __init__(self, layout_doc):
        doc = load_layout_doc(layout_doc)
        self.emits_formula_number = bool(doc.get("emits_formula_number", False))
        self.name = f"precomputed:{doc.get('layout_backend', 'unknown')}"
        self._pages_by_idx = {int(page["page_idx"]): page for page in doc["pages"]}

    def batch_detect(self, images, start_page_idx: int = 0) -> list[list[ContentBlock]]:
        blocks_list = []
        for offset in range(len(images)):
            page_idx = start_page_idx + offset
            page = self._pages_by_idx.get(page_idx)
            if page is None:
                raise ValueError(
                    f"Layout doc has no page_idx={page_idx} "
                    f"(available: {sorted(self._pages_by_idx)[:10]}...)"
                )
            # 每次调用都重新构造全新的ContentBlock，避免识别阶段的下游改动污染缓存
            blocks_list.append(blocks_from_jsonable(page["blocks"]))
        return blocks_list
