import os
import pkgutil
import numpy as np
import yaml
import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO
from unimernet.common.config import Config
import unimernet.tasks as tasks
from unimernet.processors import load_processor
from magic_pdf.libs.config_reader import get_local_models_dir, get_device
from torchvision import transforms
from magic_pdf.pre_proc.ocr_span_list_modify import remove_overlaps_low_confidence_spans, remove_overlaps_min_spans
from PIL import Image
from common.ext import singleton_func
from common.custom_response import generate_response


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


@singleton_func
class CustomPEKModel:
    def __init__(self):
        # PDF-Extract-Kit/models
        models_dir = get_local_models_dir()
        self.device = get_device()
        loader = pkgutil.get_loader("magic_pdf")
        root_dir = Path(loader.path).parent
        # model_config目录
        model_config_dir = os.path.join(root_dir, 'resources', 'model_config')
        # 构建 model_configs.yaml 文件的完整路径
        config_path = os.path.join(model_config_dir, 'model_configs.yaml')
        with open(config_path, "r", encoding='utf-8') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        # 初始化公式检测模型
        self.mfd_model = mfd_model_init(str(os.path.join(models_dir, configs["weights"]["mfd"])))

        # 初始化公式解析模型
        mfr_weight_dir = str(os.path.join(models_dir, configs["weights"]["mfr"]))
        mfr_cfg_path = str(os.path.join(model_config_dir, "UniMERNet", "demo.yaml"))
        self.mfr_model, mfr_vis_processors = mfr_model_init(mfr_weight_dir, mfr_cfg_path, _device_=self.device)
        self.mfr_transform = transforms.Compose([mfr_vis_processors, ])


def get_all_spans(layout_dets) -> list:
    def remove_duplicate_spans(spans):
        new_spans = []
        for span in spans:
            if not any(span == existing_span for existing_span in new_spans):
                new_spans.append(span)
        return new_spans

    all_spans = []
    # allow_category_id_list = [3, 5, 13, 14, 15]
    """当成span拼接的"""
    #  3: 'image', # 图片
    #  5: 'table',       # 表格
    #  13: 'inline_equation',     # 行内公式
    #  14: 'interline_equation',      # 行间公式
    #  15: 'text',      # ocr识别文本
    for layout_det in layout_dets:
        if layout_det.get("bbox") is not None:
            # 兼容直接输出bbox的模型数据,如paddle
            x0, y0, x1, y1 = layout_det["bbox"]
        else:
            # 兼容直接输出poly的模型数据，如xxx
            x0, y0, _, _, x1, y1, _, _ = layout_det["poly"]
        bbox = [x0, y0, x1, y1]
        layout_det["bbox"] = bbox
        all_spans.append(layout_det)
    return remove_duplicate_spans(all_spans)


def formula_predict(mfd_model, image):
    """
    公式检测
    :param mfd_model:
    :param image:
    :return:
    """
    latex_filling_list = []
    # 公式检测
    mfd_res = mfd_model.predict(image, imgsz=1888, conf=0.25, iou=0.45, verbose=True)[0]
    for xyxy, conf, cla in zip(mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()):
        xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
        new_item = {
            'category_id': 13 + int(cla.item()),
            'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
            'score': round(float(conf.item()), 2),
            'latex': '',
        }
        latex_filling_list.append(new_item)
    return latex_filling_list


def formula_detection(file_path, upload_dir):
    """
    公式检测
    :param file_path:  文件路径
    :param upload_dir:  上传文件夹
    :return:
    """
    try:
        image_open = Image.open(file_path)
    except IOError:
        return generate_response(code=400, msg="params is not valid", msgZh="参数类型不是图片，无效参数")

    filename = Path(file_path).name

    # 获取图片宽高
    width, height = image_open.size
    # 转换为RGB，忽略透明度通道
    rgb_image = image_open.convert('RGB')
    # 保存转换后的图片
    rgb_image.save(file_path)

    # 初始化模型
    cpm = CustomPEKModel()
    # 初始化公式检测模型
    mfd_model = cpm.mfd_model

    image_conv = Image.open(file_path)
    image_array = np.array(image_conv)
    pdf_width = 1416
    pdf_height = 1888

    # 重置图片大小
    scale = min(pdf_width // 2 / width, pdf_height // 2 / height)  # 缩放比例
    nw = int(width * scale)
    nh = int(height * scale)
    image_resize = cv2.resize(image_array, (nw, nh), interpolation=cv2.INTER_LINEAR)
    resize_image_path = f"{upload_dir}/resize_{filename}"
    cv2.imwrite(resize_image_path, image_resize)
    # 将重置的图片贴到pdf白纸中
    x = (pdf_width - nw) // 2
    y = (pdf_height - nh) // 2
    new_img = Image.new('RGB', (pdf_width, pdf_height), 'white')
    image_scale = Image.open(resize_image_path)
    new_img.paste(image_scale, (x, y))

    # 公式检测
    latex_filling_list = formula_predict(mfd_model, new_img)

    os.remove(resize_image_path)

    # 将缩放图公式检测的坐标还原为原图公式检测的坐标
    for item in latex_filling_list:
        item_poly = item["poly"]
        item["poly"] = [
            (item_poly[0] - x) / scale,
            (item_poly[1] - y) / scale,
            (item_poly[2] - x) / scale,
            (item_poly[3] - y) / scale,
            (item_poly[4] - x) / scale,
            (item_poly[5] - y) / scale,
            (item_poly[6] - x) / scale,
            (item_poly[7] - y) / scale,
        ]

    if not latex_filling_list:
        return generate_response(code=1001, msg="detection fail", msgZh="公式检测失败，图片过小，无法检测")

    spans = get_all_spans(latex_filling_list)
    '''删除重叠spans中置信度较低的那些'''
    spans, dropped_spans_by_confidence = remove_overlaps_low_confidence_spans(spans)
    '''删除重叠spans中较小的那些'''
    spans, dropped_spans_by_span_overlap = remove_overlaps_min_spans(spans)

    return generate_response(data={
        'layout': spans,
    })


def formula_recognition(file_path, upload_dir):
    """
    公式识别
    :param file_path:  文件路径
    :param upload_dir:  上传文件夹
    :return:
    """
    try:
        image_open = Image.open(file_path)
    except IOError:
        return generate_response(code=400, msg="params is not valid", msgZh="参数类型不是图片，无效参数")

    filename = Path(file_path).name

    # 获取图片宽高
    width, height = image_open.size
    # 转换为RGB，忽略透明度通道
    rgb_image = image_open.convert('RGB')
    # 保存转换后的图片
    rgb_image.save(file_path)

    image_conv = Image.open(file_path)
    image_array = np.array(image_conv)
    pdf_width = 1416
    pdf_height = 1888

    # 重置图片大小
    scale = min(pdf_width // 2 / width, pdf_height // 2 / height)  # 缩放比例
    nw = int(width * scale)
    nh = int(height * scale)
    image_resize = cv2.resize(image_array, (nw, nh), interpolation=cv2.INTER_LINEAR)
    resize_image_path = f"{upload_dir}/resize_{filename}"
    cv2.imwrite(resize_image_path, image_resize)
    # 将重置的图片贴到pdf白纸中
    x = (pdf_width - nw) // 2
    y = (pdf_height - nh) // 2
    new_img = Image.new('RGB', (pdf_width, pdf_height), 'white')
    image_scale = Image.open(resize_image_path)
    new_img.paste(image_scale, (x, y))
    new_img_array = np.array(new_img)

    # 初始化模型
    cpm = CustomPEKModel()
    # device
    device = cpm.device
    # 初始化公式检测模型
    mfd_model = cpm.mfd_model
    # 初始化公式解析模型
    mfr_model = cpm.mfr_model
    mfr_transform = cpm.mfr_transform
    # 公式识别
    latex_filling_list, mfr_res = formula_recognition(mfd_model, new_img_array, mfr_transform, device, mfr_model,
                                                      image_open)

    os.remove(resize_image_path)

    # 将缩放图公式检测的坐标还原为原图公式检测的坐标
    for item in latex_filling_list:
        item_poly = item["poly"]
        item["poly"] = [
            (item_poly[0] - x) / scale,
            (item_poly[1] - y) / scale,
            (item_poly[2] - x) / scale,
            (item_poly[3] - y) / scale,
            (item_poly[4] - x) / scale,
            (item_poly[5] - y) / scale,
            (item_poly[6] - x) / scale,
            (item_poly[7] - y) / scale,
        ]

    spans = get_all_spans(latex_filling_list)
    '''删除重叠spans中置信度较低的那些'''
    spans, dropped_spans_by_confidence = remove_overlaps_low_confidence_spans(spans)
    '''删除重叠spans中较小的那些'''
    spans, dropped_spans_by_span_overlap = remove_overlaps_min_spans(spans)

    if not latex_filling_list:
        width, height = image_open.size
        latex_filling_list.append({
            'category_id': 14,
            'poly': [0, 0, width, 0, width, height, 0, height],
            'score': 1,
            'latex': mfr_res[0] if mfr_res else "",
        })

    return generate_response(data={
        'layout': spans if spans else latex_filling_list,
        "mfr_res": mfr_res
    })
