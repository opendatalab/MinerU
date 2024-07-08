
import os
import time

import cv2
import fitz
import numpy as np
import torch
import unimernet.tasks as tasks
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO
from unimernet.common.config import Config
from unimernet.processors import load_processor



class CustomPEKModel:
    def __init__(self, ocr: bool = False, show_log: bool = False):
        ## ======== model init ========##
        with open('configs/model_configs.yaml') as f:
            model_configs = yaml.load(f, Loader=yaml.FullLoader)
        img_size = model_configs['model_args']['img_size']
        conf_thres = model_configs['model_args']['conf_thres']
        iou_thres = model_configs['model_args']['iou_thres']
        device = model_configs['model_args']['device']
        dpi = model_configs['model_args']['pdf_dpi']
        mfd_model = mfd_model_init(model_configs['model_args']['mfd_weight'])
        mfr_model, mfr_vis_processors = mfr_model_init(model_configs['model_args']['mfr_weight'], device=device)
        mfr_transform = transforms.Compose([mfr_vis_processors, ])
        layout_model = layout_model_init(model_configs['model_args']['layout_weight'])
        ocr_model = ModifiedPaddleOCR(show_log=True)
        print(now.strftime('%Y-%m-%d %H:%M:%S'))
        print('Model init done!')
        ## ======== model init ========##

    def __call__(self, image):

        # layout检测 + 公式检测
        doc_layout_result = []
        latex_filling_list = []
        mf_image_list = []

            img_H, img_W = image.shape[0], image.shape[1]
            layout_res = layout_model(image, ignore_catids=[])
            # 公式检测
            mfd_res = mfd_model.predict(image, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=True)[0]
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
                height=img_H,
                width=img_W
            )
            doc_layout_result.append(layout_res)

        # 公式识别，因为识别速度较慢，为了提速，把单个pdf的所有公式裁剪完，一起批量做识别。
        a = time.time()
        dataset = MathDataset(mf_image_list, transform=mfr_transform)
        dataloader = DataLoader(dataset, batch_size=128, num_workers=0)
        mfr_res = []
        gpu_total_cost = 0
        for imgs in dataloader:
            imgs = imgs.to(device)
            gpu_start = time.time()
            output = mfr_model.generate({'image': imgs})
            gpu_cost = time.time() - gpu_start
            gpu_total_cost += gpu_cost
            print(f"gpu_cost: {gpu_cost}")
            mfr_res.extend(output['pred_str'])
        print(f"gpu_total_cost: {gpu_total_cost}")
        for res, latex in zip(latex_filling_list, mfr_res):
            res['latex'] = latex_rm_whitespace(latex)
        b = time.time()
        print("formula nums:", len(mf_image_list), "mfr time:", round(b - a, 2))

        # ocr识别
        for idx, image in enumerate(img_list):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
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
                    ocr_res = ocr_model.ocr(cropped_img, mfd_res=single_page_mfdetrec_res)[0]
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

        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(single_pdf)[0:-4]
        with open(os.path.join(output_dir, f'{basename}.json'), 'w') as f:
            json.dump(doc_layout_result, f)