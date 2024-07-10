import os
import time

import cv2
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO
from loguru import logger
from magic_pdf.model.pek_sub_modules.layoutlmv3.model_init import Layoutlmv3_Predictor
from unimernet.common.config import Config
import unimernet.tasks as tasks
from unimernet.processors import load_processor
import argparse
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from magic_pdf.model.pek_sub_modules.post_process import get_croped_image, latex_rm_whitespace
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


class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # if not pil image, then convert to pil image
        if isinstance(self.image_paths[idx], str):
            raw_image = Image.open(self.image_paths[idx])
        else:
            raw_image = self.image_paths[idx]
        if self.transform:
            image = self.transform(raw_image)
        return image


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

    def __call__(self, images):
        # layout检测 + 公式检测
        doc_layout_result = []
        latex_filling_list = []
        mf_image_list = []
        for idx, img_dict in enumerate(images):
            image = img_dict["img"]
            img_height, img_width = img_dict["height"], img_dict["width"]
            layout_res = self.layout_model(image, ignore_catids=[])
            # 公式检测
            mfd_res = self.mfd_model.predict(image, imgsz=1888, conf=0.25, iou=0.45, verbose=True)[0]
            for xyxy, conf, cla in zip(mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()):
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                new_item = {
                    'category_id': 13 + int(cla.item()),
                    'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                    'score': round(float(conf.item()), 2),
                    'latex': '',
                }
                layout_res['layout_dets'].append(new_item)
                latex_filling_list.append(new_item)
                bbox_img = get_croped_image(Image.fromarray(image), [xmin, ymin, xmax, ymax])
                mf_image_list.append(bbox_img)

            layout_res['page_info'] = dict(
                page_no=idx,
                height=img_height,
                width=img_width
            )
            doc_layout_result.append(layout_res)

        # 公式识别，因为识别速度较慢，为了提速，把单个pdf的所有公式裁剪完，一起批量做识别。
        a = time.time()
        dataset = MathDataset(mf_image_list, transform=self.mfr_transform)
        dataloader = DataLoader(dataset, batch_size=128, num_workers=0)
        mfr_res = []
        for imgs in dataloader:
            start = time.time()
            imgs = imgs.to(self.device)
            output = self.mfr_model.generate({'image': imgs})
            mfr_res.extend(output['pred_str'])
            cost = time.time() - start
            logger.info(f"batch size: {len(imgs)}, cost time: {round(cost, 2)}")
        for res, latex in zip(latex_filling_list, mfr_res):
            res['latex'] = latex_rm_whitespace(latex)
        b = time.time()
        logger.info(f"formula nums: {len(mf_image_list)}, mfr time: {round(b - a, 2)}")

        # ocr识别
        if self.apply_ocr:
            for idx, img_dict in enumerate(images):
                image = img_dict["img"]
                pil_img = Image.fromarray(image)
                single_page_res = doc_layout_result[idx]['layout_dets']
                single_page_mfdetrec_res = []
                for res in single_page_res:
                    if int(res['category_id']) in [13, 14]:
                        xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                        xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                        single_page_mfdetrec_res.append({
                            "bbox": [xmin, ymin, xmax, ymax],
                        })
                for res in single_page_res:
                    if int(res['category_id']) in [0, 1, 2, 4, 6, 7]:  # 需要进行ocr的类别
                        xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                        xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                        crop_box = [xmin, ymin, xmax, ymax]
                        cropped_img = Image.new('RGB', pil_img.size, 'white')
                        cropped_img.paste(pil_img.crop(crop_box), crop_box)
                        cropped_img = cv2.cvtColor(np.asarray(cropped_img), cv2.COLOR_RGB2BGR)
                        ocr_res = self.ocr_model.ocr(cropped_img, mfd_res=single_page_mfdetrec_res)[0]
                        if ocr_res:
                            for box_ocr_res in ocr_res:
                                p1, p2, p3, p4 = box_ocr_res[0]
                                text, score = box_ocr_res[1]
                                doc_layout_result[idx]['layout_dets'].append({
                                    'category_id': 15,
                                    'poly': p1 + p2 + p3 + p4,
                                    'score': round(score, 2),
                                    'text': text,
                                })

        return doc_layout_result
