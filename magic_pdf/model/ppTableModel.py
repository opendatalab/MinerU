import cv2
from paddleocr.ppstructure.table.predict_table import TableSystem
from paddleocr.tools.infer.utility import (
    init_args as infer_args,
)
from paddleocr.ppstructure.utility import init_args
from magic_pdf.libs.Constants import *
from paddleocr.ppocr.utils.utility import check_and_read
import os


class ppTableModel(object):

    def __init__(self, config):
        args = self.parse_args(**config)
        self.table_sys = TableSystem(args)
    def img2html(self, image):
        pred_res, _ = self.table_sys(image)
        pred_html = pred_res["html"]
        res = '<td><table  border="1">' + pred_html.replace("<html><body><table>", "").replace("</table></body></html>",
                                                                                               "") + "</table></td>\n"
        return res

    def parse_args(self, **kwargs):
        parser = init_args()
        model_dir = kwargs.get("model_dir")
        table_model_dir = os.path.join(model_dir, TABLE_MASTER_DIR)
        table_char_dict_path = os.path.join(model_dir, TABLE_MASTER_DICT)
        det_model_dir = os.path.join(model_dir, DETECT_MODEL_DIR)
        rec_model_dir = os.path.join(model_dir, REC_MODEL_DIR)
        rec_char_dict_path = os.path.join(model_dir, REC_CHAR_DICT)
        device = kwargs.get("device", "cpu")
        use_gpu = True if device == "cuda" else False
        config = {
            "use_gpu": use_gpu,
            "table_max_len": kwargs.get("table_max_len", TABLE_MAX_LEN),
            "table_algorithm": TABLE_MASTER,
            "table_model_dir": table_model_dir,
            "table_char_dict_path": table_char_dict_path,
            "det_model_dir": det_model_dir,
            "rec_model_dir": rec_model_dir,
            "rec_char_dict_path": rec_char_dict_path,
        }
        print(config)
        parser.set_defaults(**config)
        return parser.parse_args()


if __name__ == '__main__':
    config = {"device": "cuda",
             "model_dir": "D:/models/paddle/"}
    pp_table_model = ppTableModel(config)
    # print(os.path.exists("E:/test_data/表格/test.jpg"))
    image_file = "D:/test_data/table/test.PNG"
    img, flag, _ = check_and_read(image_file)
    if not flag:
        img = cv2.imread(image_file)

    # print(img)

    res = pp_table_model.img2html(img)
    print(res)

