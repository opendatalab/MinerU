from doclayout_yolo import YOLOv10


class DocLayoutYOLOModel(object):
    def __init__(self, weight, device):
        self.model = YOLOv10(weight)
        self.device = device

    def predict(self, image):
        layout_res = []
        doclayout_yolo_res = self.model.predict(image, imgsz=1024, conf=0.25, iou=0.45, verbose=True, device=self.device)[0]
        for xyxy, conf, cla in zip(doclayout_yolo_res.boxes.xyxy.cpu(), doclayout_yolo_res.boxes.conf.cpu(),
                                   doclayout_yolo_res.boxes.cls.cpu()):
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
            new_item = {
                'category_id': int(cla.item()),
                'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                'score': round(float(conf.item()), 3),
            }
            layout_res.append(new_item)
        return layout_res