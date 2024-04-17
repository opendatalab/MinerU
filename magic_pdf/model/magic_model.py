import json

from magic_pdf.libs.commons import fitz
from loguru import logger

from magic_pdf.libs.commons import join_path
from magic_pdf.libs.coordinate_transform import get_scale_ratio
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter


class MagicModel():
    """
    每个函数没有得到元素的时候返回空list
    
    """

    def __fix_axis(self):
        for model_page_info in self.__model_list:
            page_no = model_page_info['page_info']['page_no']
            horizontal_scale_ratio, vertical_scale_ratio = get_scale_ratio(model_page_info, self.__docs[page_no])
            layout_dets = model_page_info["layout_dets"]
            for layout_det in layout_dets:
                x0, y0, _, _, x1, y1, _, _ = layout_det["poly"]
                bbox = [
                    int(x0 / horizontal_scale_ratio),
                    int(y0 / vertical_scale_ratio),
                    int(x1 / horizontal_scale_ratio),
                    int(y1 / vertical_scale_ratio),
                ]
                layout_det["bbox"] = bbox

    def __init__(self, model_list: list, docs: fitz.Document):
        self.__model_list = model_list
        self.__docs = docs
        self.__fix_axis()

    def get_imgs(self, page_no: int):  # @许瑞

        image_block = {

        }
        image_block['bbox'] = [x0, y0, x1, y1]  # 计算出来
        image_block['img_body_bbox'] = [x0, y0, x1, y1]
        image_blcok['img_caption_bbox'] = [x0, y0, x1, y1]  # 如果没有就是None，但是保证key存在
        image_blcok['img_caption_text'] = [x0, y0, x1, y1]  # 如果没有就是空字符串，但是保证key存在

        return [image_block, ]

    def get_tables(self, page_no: int) -> list:  # 3个坐标， caption, table主体，table-note
        pass  # 许瑞, 结构和image一样

    def get_equations(self, page_no: int) -> list:  # 有坐标，也有字
        return inline_equations, interline_equations  # @凯文

    def get_discarded(self, page_no: int) -> list:  # 自研模型，只有坐标
        pass  # @凯文

    def get_text_blocks(self, page_no: int) -> list:  # 自研模型搞的，只有坐标，没有字
        pass  # @凯文

    def get_title_blocks(self, page_no: int) -> list:  # 自研模型，只有坐标，没字
        pass  # @凯文

    def get_ocr_text(self, page_no: int) -> list:  # paddle 搞的，有字也有坐标
        pass  # @小蒙

    def get_ocr_spans(self, page_no: int) -> list:
        pass  # @小蒙


if __name__ == '__main__':
    drw = DiskReaderWriter(r"D:/project/20231108code-clean")
    pdf_file_path = r"linshixuqiu\19983-00.pdf"
    model_file_path = r"linshixuqiu\19983-00_new.json"
    pdf_bytes = drw.read(pdf_file_path, AbsReaderWriter.MODE_BIN)
    model_json_txt = drw.read(model_file_path, AbsReaderWriter.MODE_TXT)
    model_list = json.loads(model_json_txt)
    write_path = r"D:\project\20231108code-clean\linshixuqiu\19983-00"
    img_bucket_path = "imgs"
    img_writer = DiskReaderWriter(join_path(write_path, img_bucket_path))
    pdf_docs = fitz.open("pdf", pdf_bytes)
    magic_model = MagicModel(model_list, pdf_docs)
