import copy
import json
import os
from typing import Callable

from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import DataWriter
from magic_pdf.data.dataset import Dataset
from magic_pdf.filter import classify
from magic_pdf.libs.draw_bbox import draw_model_bbox
from magic_pdf.pdf_parse_union_core_v2 import pdf_parse_union
from magic_pdf.pipe.types import PipeResult


class InferenceResult:
    def __init__(self, inference_results: list, dataset: Dataset):
        self._infer_res = inference_results
        self._dataset = dataset

    def draw_model(self, file_path: str) -> None:
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        draw_model_bbox(
            copy.deepcopy(self._infer_res), self._dataset, dir_name, base_name
        )

    def dump_model(self, writer: DataWriter, file_path: str):
        writer.write_string(
            file_path, json.dumps(self._infer_res, ensure_ascii=False, indent=4)
        )

    def get_infer_res(self):
        return self._infer_res

    def apply(self, proc: Callable, *args, **kwargs):
        return proc(copy.deepcopy(self._infer_res), *args, **kwargs)

    def pipe_auto_mode(
        self,
        imageWriter: DataWriter,
        start_page_id=0,
        end_page_id=None,
        debug_mode=False,
        lang=None,
    ) -> PipeResult:
        def proc(*args, **kwargs) -> PipeResult:
            res = pdf_parse_union(*args, **kwargs)
            return PipeResult(res, self._dataset)

        pdf_proc_method = classify(self._dataset.data_bits())

        if pdf_proc_method == SupportedPdfParseMethod.TXT:
            return self.apply(
                proc,
                self._dataset,
                imageWriter,
                SupportedPdfParseMethod.TXT,
                start_page_id=0,
                end_page_id=None,
                debug_mode=False,
                lang=None,
            )
        else:
            return self.apply(
                proc,
                self._dataset,
                imageWriter,
                SupportedPdfParseMethod.OCR,
                start_page_id=0,
                end_page_id=None,
                debug_mode=False,
                lang=None,
            )

    def pipe_txt_mode(
        self,
        imageWriter: DataWriter,
        start_page_id=0,
        end_page_id=None,
        debug_mode=False,
        lang=None,
    ) -> PipeResult:
        def proc(*args, **kwargs) -> PipeResult:
            res = pdf_parse_union(*args, **kwargs)
            return PipeResult(res, self._dataset)

        return self.apply(
            proc,
            self._dataset,
            imageWriter,
            SupportedPdfParseMethod.TXT,
            start_page_id=0,
            end_page_id=None,
            debug_mode=False,
            lang=None,
        )

    def pipe_ocr_mode(
        self,
        imageWriter: DataWriter,
        start_page_id=0,
        end_page_id=None,
        debug_mode=False,
        lang=None,
    ) -> PipeResult:

        def proc(*args, **kwargs) -> PipeResult:
            res = pdf_parse_union(*args, **kwargs)
            return PipeResult(res, self._dataset)

        return self.apply(
            proc,
            self._dataset,
            imageWriter,
            SupportedPdfParseMethod.TXT,
            start_page_id=0,
            end_page_id=None,
            debug_mode=False,
            lang=None,
        )
