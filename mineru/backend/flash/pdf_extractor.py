# Copyright (c) Opendatalab. All rights reserved.

"""Flash PDF 公共提取入口。

原生文本处理由 FlashModel 提供；需要 OCR 时统一委托 Hybrid low。
"""

from __future__ import annotations

from typing import Any, Literal

from loguru import logger

from mineru.cli_old.common import read_fn
from mineru.utils.pdf_document import PDFDocument

__all__ = ["doc_analyze", "extract_pages_text"]


def extract_pages_text(
    filepath: str,
    start_page: int = 0,
    end_page: int | None = None,
) -> list[str]:
    """逐页提取 PDF 原生纯文本，并保留空白页。"""

    pages: list[str] = []
    with PDFDocument(filepath) as pdf_doc:
        page_count = pdf_doc.page_count
        end = page_count if end_page is None else min(end_page, page_count)

        # 保留旧 Flash parser 依赖的逐页纯文本接口。
        for page_idx in range(start_page, end):
            pages.append(pdf_doc.get_page_text(page_idx))
    return pages


def doc_analyze(
    pdf_bytes: bytes,
    parse_mode: Literal["auto", "txt", "ocr"] = "auto",
    page_index_map: list[int] | None = None,
) -> list[list[dict[str, Any]]]:
    """使用 Flash 提取原生文本，OCR 文档则委托 Hybrid low。"""

    if parse_mode not in {"auto", "txt", "ocr"}:
        raise ValueError(f"parse_mode {parse_mode} is not supported")

    with PDFDocument(pdf_bytes) as pdf_doc:
        page_count = pdf_doc.page_count
        if page_index_map and len(page_index_map) != page_count:
            raise ValueError(
                f"Flash page_index_map length mismatch: page_count={page_count}, page_index_map={len(page_index_map)}"
            )

        resolved_mode: Literal["txt", "ocr"]
        if parse_mode == "auto":
            resolved_mode = pdf_doc.classify()
        else:
            resolved_mode = parse_mode

        if resolved_mode == "txt":
            # 延迟加载 FlashModel，使逐页纯文本和 OCR 路径保持轻量。
            from mineru.model.flash import FlashModel

            return FlashModel().predict(pdf_doc)

    # OCR 路径延迟加载 Hybrid，保证原生 Flash 不引入本地视觉模型运行时。
    from mineru.backend.hybrid.analyze import doc_analyze as hybrid_doc_analyze

    _middle_json, model_list = hybrid_doc_analyze(
        pdf_bytes,
        effort="low",
        parse_mode="ocr",
        page_index_map=page_index_map,
    )
    return model_list


if __name__ == '__main__':
    if __name__ == "__main__":
        # pdf_path = "/Users/myhloli/pdf/截断合并/demo1-3.pdf"
        # pdf_path = "/Users/myhloli/pdf/png/seal4.png"  # shubiao.png
        pdf_path = "/Users/myhloli/projects/20240809magic_pdf/Magic-PDF/demo/pdfs/demo4.pdf"
        pdf_bytes = read_fn(pdf_path)
        model_list = doc_analyze(pdf_bytes, parse_mode="auto")
        logger.info(f"model_list: {model_list}")
