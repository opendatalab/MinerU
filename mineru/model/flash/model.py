# Copyright (c) Opendatalab. All rights reserved.

"""Flash PDF 与 Office 文档模型。"""

from __future__ import annotations

from typing import Any, BinaryIO

from mineru.utils.pdf_document import PDFDocument

from .native_pdf import pipeline


class PdfModel:
    """将 Flash 原生 PDF 流水线包装为无状态模型。"""

    def predict(self, pdf_doc: PDFDocument) -> list[list[dict[str, Any]]]:
        """分析调用方持有的 PDFDocument，并原样返回分页 model_list。"""

        return pipeline._analyze_native_document(pdf_doc)


class DocxModel:
    """将 DOCX Converter 包装为无状态模型。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """转换调用方持有的 DOCX 二进制流，并返回分页 model_list。"""

        # 延迟加载 Converter，避免纯 PDF 路径提前加载 Office 依赖。
        from .docx.docx_converter import DocxConverter

        converter = DocxConverter()
        converter.convert(file_binary)
        return converter.pages


class PptxModel:
    """将 PPTX Converter 包装为无状态模型。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """转换调用方持有的 PPTX 二进制流，并返回分页 model_list。"""

        # 延迟加载 Converter，避免纯 PDF 路径提前加载 Office 依赖。
        from .pptx.pptx_converter import PptxConverter

        converter = PptxConverter()
        converter.convert(file_binary)
        return converter.pages


class PptModel:
    """将 PowerPoint 97–2003 Converter 包装为无状态模型。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """转换调用方持有的 PPT 二进制流，并返回逐幻灯片 model-list。"""

        # 延迟加载旧版 PPT 解析器，避免其他格式提前加载 olefile。
        from .ppt.ppt_converter import PptConverter

        converter = PptConverter()
        converter.convert(file_binary)
        return converter.pages


class XlsModel:
    """将 Excel 97–2003 Converter 包装为无状态模型。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """转换调用方持有的 XLS 二进制流，并返回逐工作表 model-list。"""

        # 延迟加载旧版 XLS 解析器，避免其他格式提前加载 olefile/openpyxl。
        from .xls.xls_converter import XlsConverter

        converter = XlsConverter()
        converter.convert(file_binary)
        return converter.pages


class XlsxModel:
    """将 XLSX Converter 包装为无状态模型。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """转换调用方持有的 XLSX 二进制流，并返回分页 model_list。"""

        # 延迟加载 Converter，避免纯 PDF 路径提前加载 Office 依赖。
        from .xlsx.xlsx_converter import XlsxConverter

        converter = XlsxConverter()
        converter.convert(file_binary)
        return converter.pages
