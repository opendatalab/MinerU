import json

from loguru import logger

from magic_pdf.dict2md.mkcontent import mk_universal_format, mk_mm_markdown
from magic_pdf.dict2md.ocr_mkcontent import make_standard_format_with_para, ocr_mk_mm_markdown_with_para
from magic_pdf.filter.pdf_classify_by_type import classify
from magic_pdf.filter.pdf_meta_scan import pdf_meta_scan
from magic_pdf.io.AbsReaderWriter import AbsReaderWriter
from magic_pdf.io.DiskReaderWriter import DiskReaderWriter
from magic_pdf.libs.commons import join_path
from magic_pdf.libs.detect_language_from_model import get_language_from_model
from magic_pdf.libs.drop_reason import DropReason
from magic_pdf.libs.json_compressor import JsonCompressor
from magic_pdf.user_api import parse_union_pdf, parse_ocr_pdf


class UNIPipe:
    def __init__(self):
        pass

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

    def parse(self, pdf_bytes: bytes, image_writer, jso_useful_key) -> dict:
        """
        根据pdf类型，解析pdf
        """
        text_language = get_language_from_model(jso_useful_key['model_list'])
        allow_language = ["zh", "en"]  # 允许的语言,目前只允许简中和英文的
        logger.info(f"pdf text_language is {text_language}")
        if text_language not in allow_language:  # 如果语言不在允许的语言中，则drop
            raise Exception(f"pdf meta_scan need_drop,reason is {DropReason.NOT_ALLOW_LANGUAGE}")
        else:
            if jso_useful_key['_pdf_type'] == "txt":
                pdf_mid_data = parse_union_pdf(pdf_bytes, jso_useful_key['model_list'], image_writer)
            elif jso_useful_key['_pdf_type'] == "ocr":
                pdf_mid_data = parse_ocr_pdf(pdf_bytes, jso_useful_key['model_list'], image_writer)
            else:
                raise Exception(f"pdf type is not txt or ocr")
            return JsonCompressor.compress_json(pdf_mid_data)

    @staticmethod
    def mk_uni_format(pdf_mid_data: str, img_buket_path: str) -> list:
        """
        根据pdf类型，生成统一格式content_list
        """
        pdf_mid_data = JsonCompressor.decompress_json(pdf_mid_data)
        parse_type = pdf_mid_data["_parse_type"]
        pdf_info_list = pdf_mid_data["pdf_info"]
        if parse_type == "txt":
            content_list = mk_universal_format(pdf_info_list, img_buket_path)
        elif parse_type == "ocr":
            content_list = make_standard_format_with_para(pdf_info_list, img_buket_path)
        return content_list

    @staticmethod
    def mk_markdown(pdf_mid_data: str, img_buket_path: str) -> list:
        """
        根据pdf类型，markdown
        """
        pdf_mid_data = JsonCompressor.decompress_json(pdf_mid_data)
        parse_type = pdf_mid_data["_parse_type"]
        pdf_info_list = pdf_mid_data["pdf_info"]
        if parse_type == "txt":
            content_list = mk_universal_format(pdf_info_list, img_buket_path)
            md_content = mk_mm_markdown(content_list)
        elif parse_type == "ocr":
            md_content = ocr_mk_mm_markdown_with_para(pdf_info_list, img_buket_path)
        return md_content


if __name__ == '__main__':
    # 测试
    # file_path = r"tmp/unittest/download-pdfs/数学新星网/edu_00001236.pdf"
    drw = DiskReaderWriter(r"D:/project/20231108code-clean")
    # pdf_bytes = drw.read(path=file_path, mode=AbsReaderWriter.MODE_BIN)
    # pdf_type = UNIPipe.classify(pdf_bytes)
    # logger.info(f"pdf_type is {pdf_type}")

    pdf_file_path = r"linshixuqiu\25536-00.pdf"
    model_file_path = r"linshixuqiu\25536-00.json"
    pdf_bytes = drw.read(pdf_file_path, AbsReaderWriter.MODE_BIN)
    model_json_txt = drw.read(model_file_path, AbsReaderWriter.MODE_TXT)

    pdf_type = UNIPipe.classify(pdf_bytes)
    logger.info(f"pdf_type is {pdf_type}")
    jso_useful_key = {
        "_pdf_type": pdf_type,
        "model_list": json.loads(model_json_txt),
    }
    pipe = UNIPipe()
    write_path = r"D:\project\20231108code-clean\linshixuqiu\25536-00"
    img_buket_path = "imgs"
    img_writer = DiskReaderWriter(join_path(write_path, img_buket_path))
    pdf_mid_data = pipe.parse(pdf_bytes, img_writer, jso_useful_key)

    md_content = pipe.mk_markdown(pdf_mid_data, "imgs")
    md_writer = DiskReaderWriter(write_path)
    md_writer.write(md_content, "25536-00.md", AbsReaderWriter.MODE_TXT)
    md_writer.write(json.dumps(JsonCompressor.decompress_json(pdf_mid_data), ensure_ascii=False, indent=4), "25536-00.json", AbsReaderWriter.MODE_TXT)
