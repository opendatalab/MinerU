import os
from typing import List, Union
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw

from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


class YOLOv8MFDModel:
    def __init__(
        self,
        weight: str,
        device: str = "cpu",
        imgsz: int = 1888,
        conf: float = 0.25,
        iou: float = 0.45,
    ):
        self.model = YOLO(weight).to(device)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def _run_predict(
        self,
        inputs: Union[np.ndarray, Image.Image, List],
        is_batch: bool = False,
        conf: float = None,
    ) -> List:
        preds = self.model.predict(
            inputs,
            imgsz=self.imgsz,
            conf=conf if conf is not None else self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device
        )
        return [pred.cpu() for pred in preds] if is_batch else preds[0].cpu()

    def predict(
            self,
            image: Union[np.ndarray, Image.Image],
            conf: float = None,
    ):
        return self._run_predict(image, is_batch=False, conf=conf)

    def batch_predict(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 4,
        conf: float = None,
    ) -> List:
        results = []
        with tqdm(total=len(images), desc="MFD Predict") as pbar:
            for idx in range(0, len(images), batch_size):
                batch = images[idx: idx + batch_size]
                batch_preds = self._run_predict(batch, is_batch=True, conf=conf)
                results.extend(batch_preds)
                pbar.update(len(batch))
        return results

    def visualize(
        self,
        image: Union[np.ndarray, Image.Image],
        results: List
    ) -> Image.Image:

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        formula_list = []
        for xyxy, conf, cla in zip(
                results.boxes.xyxy.cpu(), results.boxes.conf.cpu(), results.boxes.cls.cpu()
        ):
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
            new_item = {
                "category_id": 13 + int(cla.item()),
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(conf.item()), 2),
            }
            formula_list.append(new_item)

        draw = ImageDraw.Draw(image)
        for res in formula_list:
            poly = res['poly']
            xmin, ymin, xmax, ymax = poly[0], poly[1], poly[4], poly[5]
            print(
                f"Detected box: {xmin}, {ymin}, {xmax}, {ymax}, Category ID: {res['category_id']}, Score: {res['score']}")
            # 使用PIL在图像上画框
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
            # 在框旁边画置信度
            draw.text((xmax + 10, ymin + 10), f"{res['score']:.2f}", fill="red", font_size=22)
        return image

if __name__ == '__main__':
    image_path = r"C:\Users\zhaoxiaomeng\Downloads\screenshot-20250821-192948.png"
    yolo_v8_mfd_weights = os.path.join(auto_download_and_get_model_root_path(ModelPath.yolo_v8_mfd),
                                          ModelPath.yolo_v8_mfd)
    device = 'cuda'
    model = YOLOv8MFDModel(
        weight=yolo_v8_mfd_weights,
        device=device,
    )
    image = Image.open(image_path)
    results = model.predict(image)

    image = model.visualize(image, results)

    image.show()  # 显示图像