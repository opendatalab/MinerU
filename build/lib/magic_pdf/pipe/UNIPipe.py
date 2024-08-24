import json

from loguru import logger

from magic_pdf.libs.MakeContentConfig import DropMode, MakeMode
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf.libs.commons import join_path
from magic_pdf.pipe.AbsPipe import AbsPipe
from magic_pdf.user_api import parse_union_pdf, parse_ocr_pdf


class UNIPipe(AbsPipe):

    def __init__(self, pdf_bytes: bytes, jso_useful_key: dict, image_writer: AbsReaderWriter, is_debug: bool = False):
        self.pdf_type = jso_useful_key["_pdf_type"]
        super().__init__(pdf_bytes, jso_useful_key["model_list"], image_writer, is_debug)
        if len(self.model_list) == 0:
            self.input_model_is_empty = True
        else:
            self.input_model_is_empty = False

    def pipe_classify(self):
        self.pdf_type = AbsPipe.classify(self.pdf_bytes)

    def pipe_analyze(self):
        if self.pdf_type == self.PIP_TXT:
            self.model_list = doc_analyze(self.pdf_bytes, ocr=False)
        elif self.pdf_type == self.PIP_OCR:
            self.model_list = doc_analyze(self.pdf_bytes, ocr=True)

    def pipe_parse(self):
        if self.pdf_type == self.PIP_TXT:
            self.pdf_mid_data = parse_union_pdf(self.pdf_bytes, self.model_list, self.image_writer,
                                                is_debug=self.is_debug, input_model_is_empty=self.input_model_is_empty)
        elif self.pdf_type == self.PIP_OCR:
            self.pdf_mid_data = parse_ocr_pdf(self.pdf_bytes, self.model_list, self.image_writer,
                                              is_debug=self.is_debug)

    def pipe_mk_uni_format(self, img_parent_path: str, drop_mode=DropMode.WHOLE_PDF):
        result = super().pipe_mk_uni_format(img_parent_path, drop_mode)
        logger.info("uni_pipe mk content list finished")
        return result

    def pipe_mk_markdown(self, img_parent_path: str, drop_mode=DropMode.WHOLE_PDF, md_make_mode=MakeMode.MM_MD):
        result = super().pipe_mk_markdown(img_parent_path, drop_mode, md_make_mode)
        logger.info(f"uni_pipe mk {md_make_mode} finished")
        return result


if __name__ == '__main__':
    # 测试
    drw = DiskReaderWriter(r"D:/project/20231108code-clean")

    pdf_file_path = r"linshixuqiu\19983-00.pdf"
    model_file_path = r"linshixuqiu\19983-00.json"
    pdf_bytes = drw.read(pdf_file_path, AbsReaderWriter.MODE_BIN)
    model_json_txt = drw.read(model_file_path, AbsReaderWriter.MODE_TXT)
    model_list = json.loads(model_json_txt)
    write_path = r"D:\project\20231108code-clean\linshixuqiu\19983-00"
    img_bucket_path = "imgs"
    img_writer = DiskReaderWriter(join_path(write_path, img_bucket_path))

    # pdf_type = UNIPipe.classify(pdf_bytes)
    # jso_useful_key = {
    #     "_pdf_type": pdf_type,
    #     "model_list": model_list
    # }

    jso_useful_key = {
        "_pdf_type": "",
        "model_list": model_list
    }
    pipe = UNIPipe(pdf_bytes, jso_useful_key, img_writer)
    pipe.pipe_classify()
    pipe.pipe_parse()
    md_content = pipe.pipe_mk_markdown(img_bucket_path)
    content_list = pipe.pipe_mk_uni_format(img_bucket_path)

    md_writer = DiskReaderWriter(write_path)
    md_writer.write(md_content, "19983-00.md", AbsReaderWriter.MODE_TXT)
    md_writer.write(json.dumps(pipe.pdf_mid_data, ensure_ascii=False, indent=4), "19983-00.json",
                    AbsReaderWriter.MODE_TXT)
    md_writer.write(str(content_list), "19983-00.txt", AbsReaderWriter.MODE_TXT)
