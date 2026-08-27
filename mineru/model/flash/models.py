# Copyright (c) Opendatalab. All rights reserved.

"""Flash PDF、EPUB、CSV 与 Office/RTF 文档模型。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    from .pdf.document import PDFDocument


class PdfModel:
    """将 Flash 原生 PDF 流水线包装为无状态模型。"""

    def predict(self, pdf_doc: PDFDocument) -> list[list[dict[str, Any]]]:
        """分析调用方持有的 PDFDocument，并原样返回分页 model_list。"""
        from .pdf import pipeline

        return pipeline._analyze_native_document(pdf_doc)


class CsvModel:
    """将 CSV 分隔符文本包装为无状态 Flash 模型。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """转换调用方持有的 CSV 二进制流，并返回单逻辑页 model_list。"""
        from .csv import convert_csv

        return convert_csv(file_binary)


class EpubModel:
    """将 EPUB OCF/OPF 文档包装为无状态 Flash 模型。"""

    def predict(
        self,
        file_binary: BinaryIO,
    ) -> list[list[dict[str, Any]]]:
        """转换调用方持有的整本 EPUB 流，并返回目录页和全部正文逻辑页。"""
        from .epub.converter import EpubConverter

        converter = EpubConverter()
        converter.convert(file_binary)
        return converter.pages


class RtfModel:
    """将 Rich Text Format 文档包装为无状态 Flash 模型。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """转换调用方持有的 RTF 二进制流，并返回单逻辑页 model_list。"""
        from .office.rtf.converter import RtfConverter

        converter = RtfConverter()
        converter.convert(file_binary)
        return converter.pages


class DocxModel:
    """将 DOCX Converter 包装为无状态模型。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """转换调用方持有的 DOCX 二进制流，并返回分页 model_list。"""

        # 延迟加载 Converter，避免纯 PDF 路径提前加载 Office 依赖。
        from .office.docx.docx_converter import DocxConverter

        converter = DocxConverter()
        converter.convert(file_binary)
        return converter.pages


class DocModel:
    """将 Word 97–2003 Converter 包装为无状态模型。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """转换调用方持有的 DOC 二进制流，并返回逐 section model-list。"""

        # 延迟加载旧版 DOC 解析器，避免其他格式提前加载 olefile。
        from .office.doc.doc_converter import DocConverter

        converter = DocConverter()
        converter.convert(file_binary)
        return converter.pages


class PptxModel:
    """将 PPTX Converter 包装为无状态模型。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """转换调用方持有的 PPTX 二进制流，并返回分页 model_list。"""

        # 延迟加载 Converter，避免纯 PDF 路径提前加载 Office 依赖。
        from .office.pptx.pptx_converter import PptxConverter

        converter = PptxConverter()
        converter.convert(file_binary)
        return converter.pages


class PptModel:
    """将 PowerPoint 97–2003 Converter 包装为无状态模型。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """转换调用方持有的 PPT 二进制流，并返回逐幻灯片 model-list。"""

        # 延迟加载旧版 PPT 解析器，避免其他格式提前加载 olefile。
        from .office.ppt.ppt_converter import PptConverter

        converter = PptConverter()
        converter.convert(file_binary)
        return converter.pages


class XlsModel:
    """将 Excel 97–2003 Converter 包装为无状态模型。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """转换调用方持有的 XLS 二进制流，并返回逐工作表 model-list。"""

        # 延迟加载旧版 XLS 解析器，避免其他格式提前加载 olefile/openpyxl。
        from .office.xls.xls_converter import XlsConverter

        converter = XlsConverter()
        converter.convert(file_binary)
        return converter.pages


class XlsxModel:
    """将 XLSX Converter 包装为无状态模型。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """转换调用方持有的 XLSX 二进制流，并返回分页 model_list。"""

        # 延迟加载 Converter，避免纯 PDF 路径提前加载 Office 依赖。
        from .office.xlsx.xlsx_converter import XlsxConverter

        converter = XlsxConverter()
        converter.convert(file_binary)
        return converter.pages


class OdtModel:
    """将 OpenDocument Text 包装为无状态 Flash 模型。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """转换调用方持有的 ODT 二进制流，并返回分页 model_list。"""
        from .office.odf.converters import OdtConverter

        converter = OdtConverter()
        converter.convert(file_binary)
        return converter.pages


class OdsModel:
    """将 OpenDocument Spreadsheet 包装为无状态 Flash 模型。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """转换调用方持有的 ODS 二进制流，并返回逐工作表 model_list。"""
        from .office.odf.converters import OdsConverter

        converter = OdsConverter()
        converter.convert(file_binary)
        return converter.pages


class OdpModel:
    """将 OpenDocument Presentation 包装为无状态 Flash 模型。"""

    def predict(self, file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
        """转换调用方持有的 ODP 二进制流，并返回逐幻灯片 model_list。"""
        from .office.odf.converters import OdpConverter

        converter = OdpConverter()
        converter.convert(file_binary)
        return converter.pages
