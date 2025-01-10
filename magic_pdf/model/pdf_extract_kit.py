# flake8: noqa
import os
import time

import cv2
import numpy as np
import torch
import yaml
from loguru import logger
from PIL import Image

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁止albumentations检查更新

try:
    import torchtext

    if torchtext.__version__ >= '0.18.0':
        torchtext.disable_torchtext_deprecation_warning()
except ImportError:
    pass

from magic_pdf.config.constants import *
from magic_pdf.model.model_list import AtomicModel
from magic_pdf.model.sub_modules.model_init import AtomModelSingleton
from magic_pdf.model.sub_modules.model_utils import (
    clean_vram, crop_img, get_res_list_from_layout_res)
from magic_pdf.model.sub_modules.ocr.paddleocr.ocr_utils import (
    get_adjusted_mfdetrec_res, get_ocr_result_list)


class CustomPEKModel:

    def __init__(self, ocr: bool = False, show_log: bool = False, **kwargs):
        """
        ======== model init ========
        """
        # 获取当前文件（即 pdf_extract_kit.py）的绝对路径
        current_file_path = os.path.abspath(__file__)
        # 获取当前文件所在的目录(model)
        current_dir = os.path.dirname(current_file_path)
        # 上一级目录(magic_pdf)
        root_dir = os.path.dirname(current_dir)
        # model_config目录
        model_config_dir = os.path.join(root_dir, 'resources', 'model_config')
        # 构建 model_configs.yaml 文件的完整路径
        config_path = os.path.join(model_config_dir, 'model_configs.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)
        # 初始化解析配置

        # layout config
        self.layout_config = kwargs.get('layout_config')
        self.layout_model_name = self.layout_config.get(
            'model', MODEL_NAME.DocLayout_YOLO
        )

        # formula config
        self.formula_config = kwargs.get('formula_config')
        self.mfd_model_name = self.formula_config.get(
            'mfd_model', MODEL_NAME.YOLO_V8_MFD
        )
        self.mfr_model_name = self.formula_config.get(
            'mfr_model', MODEL_NAME.UniMerNet_v2_Small
        )
        self.apply_formula = self.formula_config.get('enable', True)

        # table config
        self.table_config = kwargs.get('table_config')
        self.apply_table = self.table_config.get('enable', False)
        self.table_max_time = self.table_config.get('max_time', TABLE_MAX_TIME_VALUE)
        self.table_model_name = self.table_config.get('model', MODEL_NAME.RAPID_TABLE)

        # ocr config
        self.apply_ocr = ocr
        self.lang = kwargs.get('lang', None)

        logger.info(
            'DocAnalysis init, this may take some times, layout_model: {}, apply_formula: {}, apply_ocr: {}, '
            'apply_table: {}, table_model: {}, lang: {}'.format(
                self.layout_model_name,
                self.apply_formula,
                self.apply_ocr,
                self.apply_table,
                self.table_model_name,
                self.lang,
            )
        )
        # 初始化解析方案
        self.device = kwargs.get('device', 'cpu')

        if str(self.device).startswith("npu"):
            import torch_npu
            os.environ['FLAGS_npu_jit_compile'] = '0'
            os.environ['FLAGS_use_stride_kernel'] = '0'
        elif str(self.device).startswith("mps"):
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

        logger.info('using device: {}'.format(self.device))
        models_dir = kwargs.get(
            'models_dir', os.path.join(root_dir, 'resources', 'models')
        )
        logger.info('using models_dir: {}'.format(models_dir))

        atom_model_manager = AtomModelSingleton()

        # 初始化公式识别
        if self.apply_formula:
            # 初始化公式检测模型
            self.mfd_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.MFD,
                mfd_weights=str(
                    os.path.join(
                        models_dir, self.configs['weights'][self.mfd_model_name]
                    )
                ),
                device=self.device,
            )

            # 初始化公式解析模型
            mfr_weight_dir = str(
                os.path.join(models_dir, self.configs['weights'][self.mfr_model_name])
            )
            mfr_cfg_path = str(os.path.join(model_config_dir, 'UniMERNet', 'demo.yaml'))

            self.mfr_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.MFR,
                mfr_weight_dir=mfr_weight_dir,
                mfr_cfg_path=mfr_cfg_path,
                device='cpu' if str(self.device).startswith("mps") else self.device,
            )

        # 初始化layout模型
        if self.layout_model_name == MODEL_NAME.LAYOUTLMv3:
            self.layout_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Layout,
                layout_model_name=MODEL_NAME.LAYOUTLMv3,
                layout_weights=str(
                    os.path.join(
                        models_dir, self.configs['weights'][self.layout_model_name]
                    )
                ),
                layout_config_file=str(
                    os.path.join(
                        model_config_dir, 'layoutlmv3', 'layoutlmv3_base_inference.yaml'
                    )
                ),
                device=self.device,
            )
        elif self.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            self.layout_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Layout,
                layout_model_name=MODEL_NAME.DocLayout_YOLO,
                doclayout_yolo_weights=str(
                    os.path.join(
                        models_dir, self.configs['weights'][self.layout_model_name]
                    )
                ),
                device=self.device,
            )
        # 初始化ocr
        self.ocr_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            ocr_show_log=show_log,
            det_db_box_thresh=0.3,
            lang=self.lang
        )
        # init table model
        if self.apply_table:
            table_model_dir = self.configs['weights'][self.table_model_name]
            self.table_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Table,
                table_model_name=self.table_model_name,
                table_model_path=str(os.path.join(models_dir, table_model_dir)),
                table_max_time=self.table_max_time,
                device=self.device,
                ocr_engine=self.ocr_model,
            )

        logger.info('DocAnalysis init done!')

    def __call__(self, image):

        pil_img = Image.fromarray(image)
        width, height = pil_img.size
        # logger.info(f'width: {width}, height: {height}')

        # layout检测
        layout_start = time.time()
        layout_res = []
        if self.layout_model_name == MODEL_NAME.LAYOUTLMv3:
            # layoutlmv3
            layout_res = self.layout_model(image, ignore_catids=[])
        elif self.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            # doclayout_yolo
            if height > width:
                input_res = {"poly":[0,0,width,0,width,height,0,height]}
                new_image, useful_list = crop_img(input_res, pil_img, crop_paste_x=width//2, crop_paste_y=0)
                paste_x, paste_y, xmin, ymin, xmax, ymax, new_width, new_height = useful_list
                layout_res = self.layout_model.predict(new_image)
                for res in layout_res:
                    p1, p2, p3, p4, p5, p6, p7, p8 = res['poly']
                    p1 = p1 - paste_x + xmin
                    p2 = p2 - paste_y + ymin
                    p3 = p3 - paste_x + xmin
                    p4 = p4 - paste_y + ymin
                    p5 = p5 - paste_x + xmin
                    p6 = p6 - paste_y + ymin
                    p7 = p7 - paste_x + xmin
                    p8 = p8 - paste_y + ymin
                    res['poly'] = [p1, p2, p3, p4, p5, p6, p7, p8]
            else:
                layout_res = self.layout_model.predict(image)

        layout_cost = round(time.time() - layout_start, 2)
        logger.info(f'layout detection time: {layout_cost}')

        if self.apply_formula:
            # 公式检测
            mfd_start = time.time()
            mfd_res = self.mfd_model.predict(image)
            logger.info(f'mfd time: {round(time.time() - mfd_start, 2)}')

            # 公式识别
            mfr_start = time.time()
            formula_list = self.mfr_model.predict(mfd_res, image)
            layout_res.extend(formula_list)
            mfr_cost = round(time.time() - mfr_start, 2)
            logger.info(f'formula nums: {len(formula_list)}, mfr time: {mfr_cost}')

        # 清理显存
        clean_vram(self.device, vram_threshold=8)

        # 从layout_res中获取ocr区域、表格区域、公式区域
        ocr_res_list, table_res_list, single_page_mfdetrec_res = (
            get_res_list_from_layout_res(layout_res)
        )

        # ocr识别
        ocr_start = time.time()
        # Process each area that requires OCR processing
        for res in ocr_res_list:
            new_image, useful_list = crop_img(res, pil_img, crop_paste_x=50, crop_paste_y=50)
            adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(single_page_mfdetrec_res, useful_list)

            # OCR recognition
            new_image = cv2.cvtColor(np.asarray(new_image), cv2.COLOR_RGB2BGR)

            if self.apply_ocr:
                ocr_res = self.ocr_model.ocr(new_image, mfd_res=adjusted_mfdetrec_res)[0]
            else:
                ocr_res = self.ocr_model.ocr(new_image, mfd_res=adjusted_mfdetrec_res, rec=False)[0]

            # Integration results
            if ocr_res:
                ocr_result_list = get_ocr_result_list(ocr_res, useful_list)
                layout_res.extend(ocr_result_list)

        ocr_cost = round(time.time() - ocr_start, 2)
        if self.apply_ocr:
            logger.info(f"ocr time: {ocr_cost}")
        else:
            logger.info(f"det time: {ocr_cost}")

        # 表格识别 table recognition
        if self.apply_table:
            table_start = time.time()
            for res in table_res_list:
                new_image, _ = crop_img(res, pil_img)
                single_table_start_time = time.time()
                html_code = None
                if self.table_model_name == MODEL_NAME.STRUCT_EQTABLE:
                    with torch.no_grad():
                        table_result = self.table_model.predict(new_image, 'html')
                        if len(table_result) > 0:
                            html_code = table_result[0]
                elif self.table_model_name == MODEL_NAME.TABLE_MASTER:
                    html_code = self.table_model.img2html(new_image)
                elif self.table_model_name == MODEL_NAME.RAPID_TABLE:
                    html_code, table_cell_bboxes, elapse = self.table_model.predict(
                        new_image
                    )
                run_time = time.time() - single_table_start_time
                if run_time > self.table_max_time:
                    logger.warning(
                        f'table recognition processing exceeds max time {self.table_max_time}s'
                    )
                # 判断是否返回正常
                if html_code:
                    expected_ending = html_code.strip().endswith(
                        '</html>'
                    ) or html_code.strip().endswith('</table>')
                    if expected_ending:
                        res['html'] = html_code
                    else:
                        logger.warning(
                            'table recognition processing fails, not found expected HTML table end'
                        )
                else:
                    logger.warning(
                        'table recognition processing fails, not get html return'
                    )
            logger.info(f'table time: {round(time.time() - table_start, 2)}')

        return layout_res
