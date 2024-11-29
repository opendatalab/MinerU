import os

import cv2
import numpy as np
from ppstructure.table.predict_table import TableSystem
from ppstructure.utility import init_args
from PIL import Image

from magic_pdf.config.constants import *  # noqa: F403


class TableMasterPaddleModel(object):
    """This class is responsible for converting image of table into HTML format
    using a pre-trained model.

    Attributes:
    - table_sys: An instance of TableSystem initialized with parsed arguments.

    Methods:
    - __init__(config): Initializes the model with configuration parameters.
    - img2html(image): Converts a PIL Image or NumPy array to HTML string.
    - parse_args(**kwargs): Parses configuration arguments.
    """

    def __init__(self, config):
        """
        Parameters:
        - config (dict): Configuration dictionary containing model_dir and device.
        """
        args = self.parse_args(**config)
        self.table_sys = TableSystem(args)

    def img2html(self, image):
        """
        Parameters:
        - image (PIL.Image or np.ndarray): The image of the table to be converted.

        Return:
        - HTML (str): A string representing the HTML structure with content of the table.
        """
        if isinstance(image, Image.Image):
            image = np.asarray(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        pred_res, _ = self.table_sys(image)
        pred_html = pred_res['html']
        # res = '<td><table  border="1">' + pred_html.replace("<html><body><table>", "").replace(
        # "</table></body></html>","") + "</table></td>\n"
        return pred_html

    def parse_args(self, **kwargs):
        parser = init_args()
        model_dir = kwargs.get('model_dir')
        table_model_dir = os.path.join(model_dir, TABLE_MASTER_DIR)  # noqa: F405
        table_char_dict_path = os.path.join(model_dir, TABLE_MASTER_DICT)  # noqa: F405
        det_model_dir = os.path.join(model_dir, DETECT_MODEL_DIR)  # noqa: F405
        rec_model_dir = os.path.join(model_dir, REC_MODEL_DIR)  # noqa: F405
        rec_char_dict_path = os.path.join(model_dir, REC_CHAR_DICT)  # noqa: F405
        device = kwargs.get('device', 'cpu')
        use_gpu = True if device.startswith('cuda') else False
        config = {
            'use_gpu': use_gpu,
            'table_max_len': kwargs.get('table_max_len', TABLE_MAX_LEN),  # noqa: F405
            'table_algorithm': 'TableMaster',
            'table_model_dir': table_model_dir,
            'table_char_dict_path': table_char_dict_path,
            'det_model_dir': det_model_dir,
            'rec_model_dir': rec_model_dir,
            'rec_char_dict_path': rec_char_dict_path,
        }
        parser.set_defaults(**config)
        return parser.parse_args([])
