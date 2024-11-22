import numpy as np
from rapid_table import RapidTable
from rapidocr_paddle import RapidOCR


class RapidTableModel(object):
    def __init__(self):
        self.table_model = RapidTable()
        self.ocr_engine = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)

    def predict(self, image):
        ocr_result, _ = self.ocr_engine(np.asarray(image))
        if ocr_result is None:
            return None, None, None
        html_code, table_cell_bboxes, elapse = self.table_model(np.asarray(image), ocr_result)
        return html_code, table_cell_bboxes, elapse