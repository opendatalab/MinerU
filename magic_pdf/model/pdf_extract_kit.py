from loguru import logger
import os
import time


os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁止albumentations检查更新
try:
    import cv2
    import yaml
    import argparse
    import numpy as np
    import torch
    import torchtext

    if torchtext.__version__ >= "0.18.0":
        torchtext.disable_torchtext_deprecation_warning()
    from PIL import Image
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader
    from ultralytics import YOLO
    from unimernet.common.config import Config
    import unimernet.tasks as tasks
    from unimernet.processors import load_processor

except ImportError as e:
    logger.exception(e)
    logger.error(
        'Required dependency not installed, please install by \n'
        '"pip install magic-pdf[full] detectron2 --extra-index-url https://myhloli.github.io/wheels/"')
    exit(1)

from magic_pdf.model.pek_sub_modules.layoutlmv3.model_init import Layoutlmv3_Predictor
from magic_pdf.model.pek_sub_modules.post_process import get_croped_image, latex_rm_whitespace
from magic_pdf.model.pek_sub_modules.self_modify import ModifiedPaddleOCR
from magic_pdf.model.pek_sub_modules.structeqtable.StructTableModel import StructTableModel


def table_model_init(model_path, max_time=400, _device_='cpu'):
    table_model = StructTableModel(model_path, max_time=max_time, device=_device_)
    return table_model


def mfd_model_init(weight):
    mfd_model = YOLO(weight)
    return mfd_model


def mfr_model_init(weight_dir, cfg_path, _device_='cpu'):
    args = argparse.Namespace(cfg_path=cfg_path, options=None)
    cfg = Config(args)
    cfg.config.model.pretrained = os.path.join(weight_dir, "pytorch_model.bin")
    cfg.config.model.model_config.model_name = weight_dir
    cfg.config.model.tokenizer_config.path = weight_dir
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    model = model.to(_device_)
    vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
    return model, vis_processor


def layout_model_init(weight, config_file, device):
    model = Layoutlmv3_Predictor(weight, config_file, device)
    return model


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
        with open(config_path, "r", encoding='utf-8') as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)
        # 初始化解析配置
        self.apply_layout = kwargs.get("apply_layout", self.configs["config"]["layout"])
        self.apply_formula = kwargs.get("apply_formula", self.configs["config"]["formula"])
        self.table_config = kwargs.get("table_config", self.configs["config"]["table_config"])
        self.apply_table = self.table_config.get("is_table_recog_enable", False)
        self.apply_ocr = ocr
        logger.info(
            "DocAnalysis init, this may take some times. apply_layout: {}, apply_formula: {}, apply_ocr: {}, apply_table: {}".format(
                self.apply_layout, self.apply_formula, self.apply_ocr, self.apply_table
            )
        )
        assert self.apply_layout, "DocAnalysis must contain layout model."
        # 初始化解析方案
        self.device = kwargs.get("device", self.configs["config"]["device"])
        logger.info("using device: {}".format(self.device))
        models_dir = kwargs.get("models_dir", os.path.join(root_dir, "resources", "models"))
        logger.info("using models_dir: {}".format(models_dir))

        # 初始化公式识别
        if self.apply_formula:
            # 初始化公式检测模型
            self.mfd_model = mfd_model_init(str(os.path.join(models_dir, self.configs["weights"]["mfd"])))

            # 初始化公式解析模型
            mfr_weight_dir = str(os.path.join(models_dir, self.configs["weights"]["mfr"]))
            mfr_cfg_path = str(os.path.join(model_config_dir, "UniMERNet", "demo.yaml"))
            self.mfr_model, mfr_vis_processors = mfr_model_init(mfr_weight_dir, mfr_cfg_path, _device_=self.device)
            self.mfr_transform = transforms.Compose([mfr_vis_processors, ])

        # 初始化layout模型
        self.layout_model = Layoutlmv3_Predictor(
            str(os.path.join(models_dir, self.configs['weights']['layout'])),
            str(os.path.join(model_config_dir, "layoutlmv3", "layoutlmv3_base_inference.yaml")),
            device=self.device
        )
        # 初始化ocr
        if self.apply_ocr:
            self.ocr_model = ModifiedPaddleOCR(show_log=show_log)

        # init structeqtable
        if self.apply_table:
            max_time = self.table_config.get("max_time", 400)
            self.table_model = table_model_init(str(os.path.join(models_dir, self.configs["weights"]["table"])),
                                                max_time=max_time, _device_=self.device)
        logger.info('DocAnalysis init done!')

    def __call__(self, image):

        latex_filling_list = []
        mf_image_list = []

        # layout检测
        layout_start = time.time()
        layout_res = self.layout_model(image, ignore_catids=[])
        layout_cost = round(time.time() - layout_start, 2)
        logger.info(f"layout detection cost: {layout_cost}")

        if self.apply_formula:
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
                layout_res.append(new_item)
                latex_filling_list.append(new_item)
                bbox_img = get_croped_image(Image.fromarray(image), [xmin, ymin, xmax, ymax])
                mf_image_list.append(bbox_img)

            # 公式识别
            mfr_start = time.time()
            dataset = MathDataset(mf_image_list, transform=self.mfr_transform)
            dataloader = DataLoader(dataset, batch_size=64, num_workers=0)
            mfr_res = []
            for mf_img in dataloader:
                mf_img = mf_img.to(self.device)
                output = self.mfr_model.generate({'image': mf_img})
                mfr_res.extend(output['pred_str'])
            for res, latex in zip(latex_filling_list, mfr_res):
                res['latex'] = latex_rm_whitespace(latex)
            mfr_cost = round(time.time() - mfr_start, 2)
            logger.info(f"formula nums: {len(mf_image_list)}, mfr time: {mfr_cost}")

        # ocr识别
        if self.apply_ocr:
            ocr_start = time.time()
            pil_img = Image.fromarray(image)

            # 筛选出需要OCR的区域和公式区域
            ocr_res_list = []
            single_page_mfdetrec_res = []
            for res in layout_res:
                if int(res['category_id']) in [13, 14]:
                    single_page_mfdetrec_res.append({
                        "bbox": [int(res['poly'][0]), int(res['poly'][1]),
                                 int(res['poly'][4]), int(res['poly'][5])],
                    })
                elif int(res['category_id']) in [0, 1, 2, 4, 6, 7]:
                    ocr_res_list.append(res)

            # 对每一个需OCR处理的区域进行处理
            for res in ocr_res_list:
                xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                xmax, ymax = int(res['poly'][4]), int(res['poly'][5])

                paste_x = 50
                paste_y = 50
                # 创建一个宽高各多50的白色背景
                new_width = xmax - xmin + paste_x * 2
                new_height = ymax - ymin + paste_y * 2
                new_image = Image.new('RGB', (new_width, new_height), 'white')

                # 裁剪图像
                crop_box = (xmin, ymin, xmax, ymax)
                cropped_img = pil_img.crop(crop_box)
                new_image.paste(cropped_img, (paste_x, paste_y))

                # 调整公式区域坐标
                adjusted_mfdetrec_res = []
                for mf_res in single_page_mfdetrec_res:
                    mf_xmin, mf_ymin, mf_xmax, mf_ymax = mf_res["bbox"]
                    # 将公式区域坐标调整为相对于裁剪区域的坐标
                    x0 = mf_xmin - xmin + paste_x
                    y0 = mf_ymin - ymin + paste_y
                    x1 = mf_xmax - xmin + paste_x
                    y1 = mf_ymax - ymin + paste_y
                    # 过滤在图外的公式块
                    if any([x1 < 0, y1 < 0]) or any([x0 > new_width, y0 > new_height]):
                        continue
                    else:
                        adjusted_mfdetrec_res.append({
                            "bbox": [x0, y0, x1, y1],
                        })

                # OCR识别
                new_image = cv2.cvtColor(np.asarray(new_image), cv2.COLOR_RGB2BGR)
                ocr_res = self.ocr_model.ocr(new_image, mfd_res=adjusted_mfdetrec_res)[0]

                # 整合结果
                if ocr_res:
                    for box_ocr_res in ocr_res:
                        p1, p2, p3, p4 = box_ocr_res[0]
                        text, score = box_ocr_res[1]

                        # 将坐标转换回原图坐标系
                        p1 = [p1[0] - paste_x + xmin, p1[1] - paste_y + ymin]
                        p2 = [p2[0] - paste_x + xmin, p2[1] - paste_y + ymin]
                        p3 = [p3[0] - paste_x + xmin, p3[1] - paste_y + ymin]
                        p4 = [p4[0] - paste_x + xmin, p4[1] - paste_y + ymin]

                        layout_res.append({
                            'category_id': 15,
                            'poly': p1 + p2 + p3 + p4,
                            'score': round(score, 2),
                            'text': text,
                        })

            ocr_cost = round(time.time() - ocr_start, 2)
            logger.info(f"ocr cost: {ocr_cost}")

        # 表格识别 table recognition
        if self.apply_table:
            pil_img = Image.fromarray(image)
            for layout in layout_res:
                if layout.get("category_id", -1) == 5:
                    poly = layout["poly"]
                    xmin, ymin = int(poly[0]), int(poly[1])
                    xmax, ymax = int(poly[4]), int(poly[5])

                    paste_x = 50
                    paste_y = 50
                    # 创建一个宽高各多50的白色背景 create a whiteboard with 50 larger width and length
                    new_width = xmax - xmin + paste_x * 2
                    new_height = ymax - ymin + paste_y * 2
                    new_image = Image.new('RGB', (new_width, new_height), 'white')

                    # 裁剪图像 crop image
                    crop_box = (xmin, ymin, xmax, ymax)
                    cropped_img = pil_img.crop(crop_box)
                    new_image.paste(cropped_img, (paste_x, paste_y))
                    start_time = time.time()
                    logger.info("------------------table recognition processing begins-----------------")
                    latex_code = self.table_model.image2latex(new_image)[0]
                    end_time = time.time()
                    run_time = end_time - start_time
                    logger.info(f"------------table recognition processing ends within {run_time}s-----")
                    layout["latex"] = latex_code

        return layout_res
