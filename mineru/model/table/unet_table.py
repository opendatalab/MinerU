import html
import cv2
import numpy as np
from mineru.model.table.wired_table_rec.main import WiredTableRecognition
from mineru.model.table.wired_table_rec.main import WiredTableInput
from PIL import Image
from loguru import logger
from mineru.model.ocr.paddleocr2pytorch.pytorch_paddle import PytorchPaddleOCR


def escape_html(input_string):
    """Escape HTML Entities."""
    return html.escape(input_string)


class UnetTableModel:
    def __init__(self, ocr_engine):
        input_args = WiredTableInput()
        self.table_model = WiredTableRecognition(input_args)
        self.ocr_engine = ocr_engine

    def predict(self, img):
        bgr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        ocr_result = self.ocr_engine.ocr(bgr_img)[0]

        if ocr_result:
            ocr_result = [
                [item[0], escape_html(item[1][0]), item[1][1]]
                for item in ocr_result
                if len(item) == 2 and isinstance(item[1], tuple)
            ]
        else:
            ocr_result = None
        if ocr_result:
            try:
                table_results = self.table_model(np.asarray(img), ocr_result)
                html_code = table_results.pred_html
                table_cell_bboxes = table_results.cell_bboxes
                logic_points = table_results.logic_points
                elapse = table_results.elapse
                return html_code, table_cell_bboxes, logic_points, elapse
            except Exception as e:
                logger.exception(e)
        return None, None, None, None
