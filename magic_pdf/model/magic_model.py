import json

from magic_pdf.libs.commons import fitz
from loguru import logger

from magic_pdf.libs.commons import join_path
from magic_pdf.libs.coordinate_transform import get_scale_ratio
from magic_pdf.libs.ocr_content_type import ContentType
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter


class MagicModel():
    """
    每个函数没有得到元素的时候返回空list
    
    """

    def __fix_axis(self):
        need_remove_list = []
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
                # 删除高度或者宽度为0的spans
                if bbox[2] - bbox[0] == 0 or bbox[3] - bbox[1] == 0:
                    need_remove_list.append(layout_det)
            for need_remove in need_remove_list:
                layout_dets.remove(need_remove)


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
        text_spans = []
        model_page_info = self.__model_list[page_no]
        layout_dets = model_page_info["layout_dets"]
        for layout_det in layout_dets:
            if layout_det["category_id"] == "15":
                span = {
                    "bbox": layout_det['bbox'],
                    "content": layout_det["text"],
                }
                text_spans.append(span)
        return text_spans

    def get_all_spans(self, page_no: int) -> list:
        all_spans = []
        model_page_info = self.__model_list[page_no]
        layout_dets = model_page_info["layout_dets"]
        allow_category_id_list = [3, 5, 13, 14, 15]
        """当成span拼接的"""
        #  3: 'image', # 图片
        #  4: 'table',       # 表格
        #  13: 'inline_equation',     # 行内公式
        #  14: 'interline_equation',      # 行间公式
        #  15: 'text',      # ocr识别文本
        for layout_det in layout_dets:
            category_id = layout_det["category_id"]
            if category_id in allow_category_id_list:
                span = {
                    "bbox": layout_det['bbox']
                }
                if category_id == 3:
                    span["type"] = ContentType.Image
                elif category_id == 5:
                    span["type"] = ContentType.Table
                elif category_id == 13:
                    span["content"] = layout_det["latex"]
                    span["type"] = ContentType.InlineEquation
                elif category_id == 14:
                    span["content"] = layout_det["latex"]
                    span["type"] = ContentType.InterlineEquation
                elif category_id == 15:
                    span["content"] = layout_det["text"]
                    span["type"] = ContentType.Text
                all_spans.append(span)
        return all_spans

    def get_page_size(self, page_no: int):  # 获取页面宽高
        # 获取当前页的page对象
        page = self.__docs[page_no]
        # 获取当前页的宽高
        page_w = page.rect.width
        page_h = page.rect.height
        return page_w, page_h


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
