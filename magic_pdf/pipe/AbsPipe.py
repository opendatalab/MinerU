from abc import ABC, abstractmethod

from magic_pdf.dict2md.ocr_mkcontent import union_make
from magic_pdf.filter.pdf_classify_by_type import classify
from magic_pdf.filter.pdf_meta_scan import pdf_meta_scan
from magic_pdf.libs.MakeContentConfig import MakeMode, DropMode
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.libs.drop_reason import DropReason
from magic_pdf.libs.json_compressor import JsonCompressor


class AbsPipe(ABC):
    """
    txt和ocr处理的抽象类
    """
    PIP_OCR = "ocr"
    PIP_TXT = "txt"

    def __init__(self, pdf_bytes: bytes, model_list: list, image_writer: AbsReaderWriter, is_debug: bool = False,
                 start_page_id=0, end_page_id=None):
        self.pdf_bytes = pdf_bytes
        self.model_list = model_list
        self.image_writer = image_writer
        self.pdf_mid_data = None  # 未压缩
        self.is_debug = is_debug
        self.start_page_id = start_page_id
        self.end_page_id = end_page_id
    
    def get_compress_pdf_mid_data(self):
        return JsonCompressor.compress_json(self.pdf_mid_data)

    @abstractmethod
    def pipe_classify(self):
        """
        有状态的分类
        """
        raise NotImplementedError

    @abstractmethod
    def pipe_analyze(self):
        """
        有状态的跑模型分析
        """
        raise NotImplementedError

    @abstractmethod
    def pipe_parse(self):
        """
        有状态的解析
        """
        raise NotImplementedError

    def pipe_mk_uni_format(self, img_parent_path: str, drop_mode=DropMode.WHOLE_PDF):
        content_list = AbsPipe.mk_uni_format(self.get_compress_pdf_mid_data(), img_parent_path, drop_mode)
        return content_list

    def pipe_mk_markdown(self, img_parent_path: str, drop_mode=DropMode.WHOLE_PDF, md_make_mode=MakeMode.MM_MD):
        md_content = AbsPipe.mk_markdown(self.get_compress_pdf_mid_data(), img_parent_path, drop_mode, md_make_mode)
        return md_content

    @staticmethod
    def classify(pdf_bytes: bytes) -> str:
        """
        根据pdf的元数据，判断是文本pdf，还是ocr pdf
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
                    pdf_meta["invalid_chars"],
                )
                if is_text_pdf:
                    return AbsPipe.PIP_TXT
                else:
                    return AbsPipe.PIP_OCR

    @staticmethod
    def mk_uni_format(compressed_pdf_mid_data: str, img_buket_path: str, drop_mode=DropMode.WHOLE_PDF) -> list:
        """
        根据pdf类型，生成统一格式content_list
        """
        pdf_mid_data = JsonCompressor.decompress_json(compressed_pdf_mid_data)
        pdf_info_list = pdf_mid_data["pdf_info"]
        content_list = union_make(pdf_info_list, MakeMode.STANDARD_FORMAT, drop_mode, img_buket_path)
        return content_list

    @staticmethod
    def mk_markdown(compressed_pdf_mid_data: str, img_buket_path: str, drop_mode=DropMode.WHOLE_PDF, md_make_mode=MakeMode.MM_MD) -> list:
        """
        根据pdf类型，markdown
        """
        pdf_mid_data = JsonCompressor.decompress_json(compressed_pdf_mid_data)
        pdf_info_list = pdf_mid_data["pdf_info"]
        md_content = union_make(pdf_info_list, md_make_mode, drop_mode, img_buket_path)
        return md_content


