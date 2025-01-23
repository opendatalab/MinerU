import cv2
import numpy as np
import torch
from loguru import logger
from rapid_table import RapidTable, RapidTableInput
from rapid_table.main import ModelType

from magic_pdf.libs.config_reader import get_device


class RapidTableModel(object):
    def __init__(self, ocr_engine, table_sub_model_name):
        sub_model_list = [model.value for model in ModelType]
        if table_sub_model_name is None:
            input_args = RapidTableInput()
        elif table_sub_model_name in  sub_model_list:
            if torch.cuda.is_available() and table_sub_model_name == "unitable":
                input_args = RapidTableInput(model_type=table_sub_model_name, use_cuda=True, device=get_device())
            else:
                input_args = RapidTableInput(model_type=table_sub_model_name)
        else:
            raise ValueError(f"Invalid table_sub_model_name: {table_sub_model_name}. It must be one of {sub_model_list}")

        self.table_model = RapidTable(input_args)

        # if ocr_engine is None:
        #     self.ocr_model_name = "RapidOCR"
        #     if torch.cuda.is_available():
        #         from rapidocr_paddle import RapidOCR
        #         self.ocr_engine = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
        #     else:
        #         from rapidocr_onnxruntime import RapidOCR
        #         self.ocr_engine = RapidOCR()
        # else:
        #     self.ocr_model_name = "PaddleOCR"
        #     self.ocr_engine = ocr_engine

        self.ocr_model_name = "RapidOCR"
        if torch.cuda.is_available():
            from rapidocr_paddle import RapidOCR
            self.ocr_engine = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
        else:
            from rapidocr_onnxruntime import RapidOCR
            self.ocr_engine = RapidOCR()

    def predict(self, image):

        if self.ocr_model_name == "RapidOCR":
            ocr_result, _ = self.ocr_engine(np.asarray(image))
        elif self.ocr_model_name == "PaddleOCR":
            bgr_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            ocr_result = self.ocr_engine.ocr(bgr_image)[0]
            if ocr_result:
                ocr_result = [[item[0], item[1][0], item[1][1]] for item in ocr_result if
                          len(item) == 2 and isinstance(item[1], tuple)]
            else:
                ocr_result = None
        else:
            logger.error("OCR model not supported")
            ocr_result = None

        if ocr_result:
            table_results = self.table_model(np.asarray(image), ocr_result)
            html_code = table_results.pred_html
            table_cell_bboxes = table_results.cell_bboxes
            logic_points = table_results.logic_points
            elapse = table_results.elapse
            return html_code, table_cell_bboxes, logic_points, elapse
        else:
            return None, None, None, None
