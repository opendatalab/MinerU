from ultralytics import YOLO


class YOLOv8MFDModel(object):
    def __init__(self, weight, device='cpu'):
        self.mfd_model = YOLO(weight)
        self.device = device

    def predict(self, image):
        mfd_res = self.mfd_model.predict(image, imgsz=1888, conf=0.25, iou=0.45, verbose=True, device=self.device)[0]
        return mfd_res

