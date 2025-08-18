import os
import html
import cv2
import numpy as np
from loguru import logger
from .main import RapidTable, RapidTableInput

from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


def escape_html(input_string):
    """Escape HTML Entities."""
    return html.escape(input_string)


class RapidTableModel(object):
    def __init__(self, ocr_engine):
        slanet_plus_model_path = os.path.join(auto_download_and_get_model_root_path(ModelPath.slanet_plus), ModelPath.slanet_plus)
        input_args = RapidTableInput(model_type='slanet_plus', model_path=slanet_plus_model_path)
        self.table_model = RapidTable(input_args)
        self.ocr_engine = ocr_engine


    def predict(self, image, table_cls_score):
        bgr_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        # Continue with OCR on potentially rotated image
        ocr_result = self.ocr_engine.ocr(bgr_image)[0]
        if ocr_result:
            ocr_result = [[item[0], escape_html(item[1][0]), item[1][1]] for item in ocr_result if
                      len(item) == 2 and isinstance(item[1], tuple)]
        else:
            ocr_result = None

        if ocr_result:
            try:
                table_results = self.table_model(np.asarray(image), ocr_result)
                html_code = table_results.pred_html
                table_cell_bboxes = table_results.cell_bboxes
                logic_points = table_results.logic_points
                elapse = table_results.elapse
                return html_code, table_cell_bboxes, logic_points, elapse
            except Exception as e:
                logger.exception(e)

        return None, None, None, None
