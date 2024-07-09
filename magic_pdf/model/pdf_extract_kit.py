import os
import numpy as np
import yaml
from ultralytics import YOLO
from loguru import logger
from magic_pdf.model.pek_sub_modules.layoutlmv3.model_init import Layoutlmv3_Predictor
from unimernet.common.config import Config
import unimernet.tasks as tasks
from unimernet.processors import load_processor
import argparse
from torchvision import transforms

from magic_pdf.model.pek_sub_modules.self_modify import ModifiedPaddleOCR


def layout_model_init(weight, config_file):
    model = Layoutlmv3_Predictor(weight, config_file)
    return model


def mfr_model_init(weight_dir, cfg_path, device='cpu'):
    args = argparse.Namespace(cfg_path=cfg_path, options=None)
    cfg = Config(args)
    cfg.config.model.pretrained = os.path.join(weight_dir, "pytorch_model.bin")
    cfg.config.model.model_config.model_name = weight_dir
    cfg.config.model.tokenizer_config.path = weight_dir
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    model = model.to(device)
    vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
    return model, vis_processor


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
        with open(config_path, "r") as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)
        # 初始化解析配置
        self.apply_layout = kwargs.get("apply_layout", self.configs["config"]["layout"])
        self.apply_formula = kwargs.get("apply_formula", self.configs["config"]["formula"])
        self.apply_ocr = ocr
        logger.info(
            "DocAnalysis init, this may take some times. apply_layout: {}, apply_formula: {}, apply_ocr: {}".format(
                self.apply_layout, self.apply_formula, self.apply_ocr
            )
        )
        assert self.apply_layout, "DocAnalysis must contain layout model."
        # 初始化解析方案
        self.device = self.configs["config"]["device"]
        logger.info("using device: {}".format(self.device))
        # 初始化layout模型
        self.layout_model = layout_model_init(
            os.path.join(root_dir, self.configs['weights']['layout']),
            os.path.join(model_config_dir, "layoutlmv3", "layoutlmv3_base_inference.yaml")
        )
        # 初始化公式识别
        if self.apply_formula:
            # 初始化公式检测模型
            self.mfd_model = YOLO(model=str(os.path.join(root_dir, self.configs["weights"]["mfd"])))
            # 初始化公式解析模型
            mfr_config_path = os.path.join(model_config_dir, 'UniMERNet', 'demo.yaml')
            self.mfr_model, mfr_vis_processors = mfr_model_init(
                os.path.join(root_dir, self.configs["weights"]["mfr"]), mfr_config_path,
                device=self.device)
            self.mfr_transform = transforms.Compose([mfr_vis_processors, ])
        # 初始化ocr
        if self.apply_ocr:
            self.ocr_model = ModifiedPaddleOCR(show_log=show_log)

        logger.info('DocAnalysis init done!')


    def __call__(self, image):
        pass
