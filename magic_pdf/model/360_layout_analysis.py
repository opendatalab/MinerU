from ultralytics import YOLO

image_path = ''  # 待预测图片路径
model_path = ''  # 权重路径
model = YOLO(model_path)

result = model(image_path, save=True, conf=0.5, save_crop=False, line_width=2)
print(result)