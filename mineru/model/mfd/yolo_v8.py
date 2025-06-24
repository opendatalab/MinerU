from typing import List, Union
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
from PIL import Image


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
        is_batch: bool = False
    ) -> List:
        preds = self.model.predict(
            inputs,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device
        )
        return [pred.cpu() for pred in preds] if is_batch else preds[0].cpu()

    def predict(self, image: Union[np.ndarray, Image.Image]):
        return self._run_predict(image)

    def batch_predict(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 4
    ) -> List:
        results = []
        with tqdm(total=len(images), desc="MFD Predict") as pbar:
            for idx in range(0, len(images), batch_size):
                batch = images[idx: idx + batch_size]
                batch_preds = self._run_predict(batch, is_batch=True)
                results.extend(batch_preds)
                pbar.update(len(batch))
        return results