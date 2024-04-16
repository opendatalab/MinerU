from abc import ABC, abstractmethod

from magic_pdf.dict2md.mkcontent import mk_universal_format, mk_mm_markdown
from magic_pdf.dict2md.ocr_mkcontent import make_standard_format_with_para, ocr_mk_mm_markdown_with_para
from magic_pdf.filter.pdf_classify_by_type import classify
from magic_pdf.filter.pdf_meta_scan import pdf_meta_scan
from magic_pdf.io.AbsReaderWriter import AbsReaderWriter
from magic_pdf.libs.drop_reason import DropReason
from magic_pdf.libs.json_compressor import JsonCompressor


class AbsPipe(ABC):
    """
    txt和ocr处理的抽象类
    """

    def __init__(self, pdf_bytes: bytes, model_list: list, image_writer: AbsReaderWriter):
        self.pdf_bytes = pdf_bytes
        self.model_list = model_list
        self.image_writer = image_writer

    @abstractmethod
    def pipe_classify(self):
        """
        有状态的分类
        """
        raise NotImplementedError

    @abstractmethod
    def pipe_parse(self):
        """
        有状态的解析
        """
        raise NotImplementedError

    @abstractmethod
    def pipe_mk_uni_format(self):
        """
        有状态的组装统一格式
        """
        raise NotImplementedError

    @abstractmethod
    def pipe_mk_markdown(self):
        """
        有状态的组装markdown
        """
        raise NotImplementedError

    @staticmethod
    def classify(pdf_bytes: bytes) -> str:
        """
        根据pdf的元数据，判断是否是文本pdf，还是ocr pdf
        """
        pdf_meta = pdf_meta_scan(pdf_bytes)
        if pdf_meta.get("_need_drop", False):  # 如果返回了需要丢弃的标志，则抛出异常
            raise Exception(f"pdf meta_scan need_drop,reason is {pdf_meta['_drop_reason']}")
        else:
            is_encrypted = pdf_meta["is_encrypted"]
            is_needs_password = pdf_meta["is_needs_password"]
            if is_encrypted or is_needs_password:  # 加密的，需要密码的，没有页面的，都不处理
                raise Exception(f"pdf meta_scan need_drop,reason is {DropReason.ENCRYPTED}")
            else:
                is_text_pdf, results = classify(
                    pdf_meta["total_page"],
                    pdf_meta["page_width_pts"],
                    pdf_meta["page_height_pts"],
                    pdf_meta["image_info_per_page"],
                    pdf_meta["text_len_per_page"],
                    pdf_meta["imgs_per_page"],
                    pdf_meta["text_layout_per_page"],
                )
                if is_text_pdf:
                    return "txt"
                else:
                    return "ocr"

    @staticmethod
    def mk_uni_format(compressed_pdf_mid_data: str, img_buket_path: str) -> list:
        """
        根据pdf类型，生成统一格式content_list
        """
        pdf_mid_data = JsonCompressor.decompress_json(compressed_pdf_mid_data)
        parse_type = pdf_mid_data["_parse_type"]
        pdf_info_list = pdf_mid_data["pdf_info"]
        if parse_type == "txt":
            content_list = mk_universal_format(pdf_info_list, img_buket_path)
        elif parse_type == "ocr":
            content_list = make_standard_format_with_para(pdf_info_list, img_buket_path)
        return content_list

    @staticmethod
    def mk_markdown(compressed_pdf_mid_data: str, img_buket_path: str) -> list:
        """
        根据pdf类型，markdown
        """
        pdf_mid_data = JsonCompressor.decompress_json(compressed_pdf_mid_data)
        parse_type = pdf_mid_data["_parse_type"]
        pdf_info_list = pdf_mid_data["pdf_info"]
        if parse_type == "txt":
            content_list = mk_universal_format(pdf_info_list, img_buket_path)
            md_content = mk_mm_markdown(content_list)
        elif parse_type == "ocr":
            md_content = ocr_mk_mm_markdown_with_para(pdf_info_list, img_buket_path)
        return md_content


