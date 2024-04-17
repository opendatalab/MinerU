from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.libs.json_compressor import JsonCompressor
from magic_pdf.pipe.AbsPipe import AbsPipe
from magic_pdf.user_api import parse_ocr_pdf


class OCRPipe(AbsPipe):

    def __init__(self, pdf_bytes: bytes, model_list: list, image_writer: AbsReaderWriter, img_parent_path: str):
        super().__init__(pdf_bytes, model_list, image_writer, img_parent_path)

    def pipe_classify(self):
        pass

    def pipe_parse(self):
        self.pdf_mid_data = parse_ocr_pdf(self.pdf_bytes, self.model_list, self.image_writer)

    def pipe_mk_uni_format(self):
        content_list = AbsPipe.mk_uni_format(self.get_compress_pdf_mid_data(), self.img_parent_path)
        return content_list

    def pipe_mk_markdown(self):
        md_content = AbsPipe.mk_markdown(self.get_compress_pdf_mid_data(), self.img_parent_path)
        return md_content
