import cv2
import numpy as np
import torch
from loguru import logger
from rapid_table import RapidTable


class RapidTableModel(object):
    def __init__(self, ocr_engine):
        self.table_model = RapidTable()
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
            html_code, table_cell_bboxes, elapse = self.table_model(np.asarray(image), ocr_result)
            return html_code, table_cell_bboxes, elapse
        else:
            return None, None, None
