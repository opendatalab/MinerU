import os
import cv2
import numpy as np
from rapid_table import RapidTable
from rapidocr_paddle import RapidOCR

try:
    import torchtext

    if torchtext.__version__ >= '0.18.0':
        torchtext.disable_torchtext_deprecation_warning()
except ImportError:
    pass
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁止albumentations检查更新

from magic_pdf.model.sub_modules.model_init import AtomModelSingleton


class RapidTableModel(object):
    def __init__(self, lang=None):
        self.table_model = RapidTable()
        # self.ocr_engine = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)

        atom_model_manager = AtomModelSingleton()
        self.ocr_engine = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            ocr_show_log=False,
            det_db_box_thresh=0.3,
            lang=lang,
        )

    def predict(self, image):
        # ocr_result, _ = self.ocr_engine(np.asarray(image))

        bgr_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        ocr_result = self.ocr_engine.ocr(bgr_image)[0]
        ocr_result = [[item[0], item[1][0], item[1][1]] for item in ocr_result if
                      len(item) == 2 and isinstance(item[1], tuple)]

        if ocr_result:
            html_code, table_cell_bboxes, elapse = self.table_model(np.asarray(image), ocr_result)
            return html_code, table_cell_bboxes, elapse
        else:
            return None, None, None
