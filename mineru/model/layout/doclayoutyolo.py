import os
from typing import List, Dict, Union

from doclayout_yolo import YOLOv10
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw

from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


class DocLayoutYOLOModel:
    def __init__(
        self,
        weight: str,
        device: str = "cuda",
        imgsz: int = 1280,
        conf: float = 0.1,
        iou: float = 0.45,
    ):
        self.model = YOLOv10(weight).to(device)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def _parse_prediction(self, prediction) -> List[Dict]:
        layout_res = []

        # 容错处理
        if not hasattr(prediction, "boxes") or prediction.boxes is None:
            return layout_res

        for xyxy, conf, cls in zip(
            prediction.boxes.xyxy.cpu(),
            prediction.boxes.conf.cpu(),
            prediction.boxes.cls.cpu(),
        ):
            coords = list(map(int, xyxy.tolist()))
            xmin, ymin, xmax, ymax = coords
            layout_res.append({
                "category_id": int(cls.item()),
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(conf.item()), 3),
            })
        return layout_res

    def predict(self, image: Union[np.ndarray, Image.Image]) -> List[Dict]:
        prediction = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )[0]
        return self._parse_prediction(prediction)

    def batch_predict(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 4
    ) -> List[List[Dict]]:
        results = []
        with tqdm(total=len(images), desc="Layout Predict") as pbar:
            for idx in range(0, len(images), batch_size):
                batch = images[idx: idx + batch_size]
                if batch_size == 1:
                    conf = 0.9 * self.conf
                else:
                    conf = self.conf
                predictions = self.model.predict(
                    batch,
                    imgsz=self.imgsz,
                    conf=conf,
                    iou=self.iou,
                    verbose=False,
                )
                for pred in predictions:
                    results.append(self._parse_prediction(pred))
                pbar.update(len(batch))
        return results

    def visualize(
            self,
            image: Union[np.ndarray, Image.Image],
            results: List
    ) -> Image.Image:

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        draw = ImageDraw.Draw(image)
        for res in results:
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
    image_path = r"C:\Users\zhaoxiaomeng\Downloads\下载1.jpg"
    doclayout_yolo_weights = os.path.join(auto_download_and_get_model_root_path(ModelPath.doclayout_yolo), ModelPath.doclayout_yolo)
    device = 'cuda'
    model = DocLayoutYOLOModel(
        weight=doclayout_yolo_weights,
        device=device,
    )
    image = Image.open(image_path)
    results = model.predict(image)

    image = model.visualize(image, results)

    image.show()  # 显示图像