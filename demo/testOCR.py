from magic_pdf.model.pp_structure_v2 import CustomPaddleModel
from magic_pdf.model.pek_sub_modules.self_modify import ModifiedPaddleOCR
from PIL import Image
import numpy as np
import os

def testCusPaddleModel (image_path, output_path):
    img = Image.open(image_path)
    img_np = np.array(img)
    # img_np = img_np[:, :, :3]
    # os.makedirs(output_path, exist_ok=True)
    model = ModifiedPaddleOCR()
    results = model.ocr(img_np)
    # for i, item in enumerate(results):
    #     if 'bbox' in item:
    #         bbox = item['bbox']
    #         x_min, y_min, x_max, y_max = bbox
    #         cropped_img = img.crop((x_min, y_min, x_max, y_max))
    #         if cropped_img.mode == 'RGBA':
    #             cropped_img = cropped_img.convert('RGB')
    #         typee = 'ocr'
    #         if 'type' in item:
    #             typee = item['type']
    #         cropped_img.save(os.path.join(output_path, f'{typee}_{i}.jpg'))
    return results

if __name__ == "__main__":
    img_path = r'D:\codePJ\Mely\MeLy-MinerU\demo\testocr.png'
    output_path = 'testOCR'
    res = testCusPaddleModel(img_path, output_path)
    print("RES")
    print(len(res[0]))
    for each in res[0]:
        print(each, sep='\n')