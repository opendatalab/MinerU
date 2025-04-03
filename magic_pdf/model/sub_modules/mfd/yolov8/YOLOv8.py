from tqdm import tqdm
from ultralytics import YOLO


class YOLOv8MFDModel(object):
    def __init__(self, weight, device="cpu"):
        self.mfd_model = YOLO(weight)
        self.device = device

    def predict(self, image):
        mfd_res = self.mfd_model.predict(
            image, imgsz=1888, conf=0.25, iou=0.45, verbose=False, device=self.device
        )[0]
        return mfd_res

    def batch_predict(self, images: list, batch_size: int) -> list:
        images_mfd_res = []
        # for index in range(0, len(images), batch_size):
        for index in tqdm(range(0, len(images), batch_size), desc="MFD Predict"):
            mfd_res = [
                image_res.cpu()
                for image_res in self.mfd_model.predict(
                    images[index : index + batch_size],
                    imgsz=1888,
                    conf=0.25,
                    iou=0.45,
                    verbose=False,
                    device=self.device,
                )
            ]
            for image_res in mfd_res:
                images_mfd_res.append(image_res)
        return images_mfd_res
