
import json
import os

from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.data.data_reader_writer import DataWriter
from magic_pdf.data.dataset import Dataset
from magic_pdf.dict2md.ocr_mkcontent import union_make
from magic_pdf.libs.draw_bbox import (draw_layout_bbox, draw_line_sort_bbox,
                                      draw_span_bbox)
from magic_pdf.libs.json_compressor import JsonCompressor


class PipeResult:
    def __init__(self, pipe_res, dataset: Dataset):
        self._pipe_res = pipe_res
        self._dataset = dataset

    def dump_md(self, writer: DataWriter, file_path: str, img_dir_or_bucket_prefix: str, drop_mode=DropMode.WHOLE_PDF, md_make_mode=MakeMode.MM_MD):
        pdf_info_list = self._pipe_res['pdf_info']
        md_content = union_make(pdf_info_list, md_make_mode, drop_mode, img_dir_or_bucket_prefix)
        writer.write_string(file_path, md_content)

    def dump_content_list(self, writer: DataWriter, file_path: str, image_dir_or_bucket_prefix: str, drop_mode=DropMode.NONE):
        pdf_info_list = self._pipe_res['pdf_info']
        content_list = union_make(pdf_info_list, MakeMode.STANDARD_FORMAT, drop_mode, image_dir_or_bucket_prefix)
        writer.write_string(file_path, json.dumps(content_list, ensure_ascii=False, indent=4))

    def dump_middle_json(self, writer: DataWriter, file_path: str):
        writer.write_string(file_path, json.dumps(self._pipe_res, ensure_ascii=False, indent=4))

    def draw_layout(self, file_path: str) -> None:
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        pdf_info = self._pipe_res['pdf_info']
        draw_layout_bbox(pdf_info, self._dataset.data_bits(), dir_name, base_name)

    def draw_span(self, file_path: str):
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        pdf_info = self._pipe_res['pdf_info']
        draw_span_bbox(pdf_info, self._dataset.data_bits(), dir_name, base_name)

    def draw_line_sort(self, file_path: str):
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        pdf_info = self._pipe_res['pdf_info']
        draw_line_sort_bbox(pdf_info, self._dataset.data_bits(), dir_name, base_name)

    def draw_content_list(self, writer: DataWriter, file_path: str, img_dir_or_bucket_prefix: str, drop_mode=DropMode.WHOLE_PDF):
        pdf_info_list = self._pipe_res['pdf_info']
        content_list = union_make(pdf_info_list, MakeMode.STANDARD_FORMAT, drop_mode, img_dir_or_bucket_prefix)
        writer.write_string(file_path, json.dumps(content_list, ensure_ascii=False, indent=4))

    def get_compress_pdf_mid_data(self):
        return JsonCompressor.compress_json(self.pdf_mid_data)
