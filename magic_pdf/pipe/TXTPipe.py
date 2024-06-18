from magic_pdf.libs.MakeContentConfig import DropMode
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.libs.json_compressor import JsonCompressor
from magic_pdf.pipe.AbsPipe import AbsPipe
from magic_pdf.user_api import parse_txt_pdf


class TXTPipe(AbsPipe):

    def __init__(self, pdf_bytes: bytes, model_list: list, image_writer: AbsReaderWriter, is_debug: bool=False):
        super().__init__(pdf_bytes, model_list, image_writer, is_debug)

    def pipe_classify(self):
        pass

    def pipe_analyze(self):
        self.model_list = doc_analyze(self.pdf_bytes, ocr=False)

    def pipe_parse(self):
        self.pdf_mid_data = parse_txt_pdf(self.pdf_bytes, self.model_list, self.image_writer, is_debug=self.is_debug)

    def pipe_mk_uni_format(self, img_parent_path: str, drop_mode=DropMode.WHOLE_PDF):
        content_list = AbsPipe.mk_uni_format(self.get_compress_pdf_mid_data(), img_parent_path, drop_mode)
        return content_list

    def pipe_mk_markdown(self, img_parent_path: str, drop_mode=DropMode.WHOLE_PDF):
        md_content = AbsPipe.mk_markdown(self.get_compress_pdf_mid_data(), img_parent_path, drop_mode)
        return md_content
