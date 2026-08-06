# Copyright (c) Opendatalab. All rights reserved.

"""Flash 原生 PDF 提取模型。"""

from __future__ import annotations

from typing import Any

from mineru.utils.pdf_document import PDFDocument

from .native_pdf import pipeline


class FlashModel:
    """将 Flash 原生 PDF 流水线包装为无状态模型。"""

    def predict(self, pdf_doc: PDFDocument) -> list[list[dict[str, Any]]]:
        """分析调用方持有的 PDFDocument，并原样返回分页 model_list。"""

        return pipeline._analyze_native_document(pdf_doc)
