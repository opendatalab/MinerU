import random

from loguru import logger

try:
    from paddleocr import PPStructure
except ImportError:
    logger.error('paddleocr not installed, please install by "pip install magic-pdf[lite]"')
    exit(1)


def region_to_bbox(region):
    x0 = region[0][0]
    y0 = region[0][1]
    x1 = region[2][0]
    y1 = region[2][1]
    return [x0, y0, x1, y1]


class CustomPaddleModel:
    def __init__(self, ocr: bool = False, show_log: bool = False):
        self.model = PPStructure(table=False, ocr=ocr, show_log=show_log)

    def __call__(self, img):
        try:
            import cv2
        except ImportError:
            logger.error("opencv-python not installed, please install by pip.")
            exit(1)
        # 将RGB图片转换为BGR格式适配paddle
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = self.model(img)
        spans = []
        for line in result:
            line.pop("img")
            """
            为paddle输出适配type no.    
            title: 0 # 标题
            text: 1 # 文本
            header: 2 # abandon
            footer: 2 # abandon
            reference: 1 # 文本 or abandon
            equation: 8 # 行间公式 block
            equation: 14 # 行间公式 text
            figure: 3 # 图片
            figure_caption: 4 # 图片描述
            table: 5 # 表格
            table_caption: 6 # 表格描述
            """
            if line["type"] == "title":
                line["category_id"] = 0
            elif line["type"] in ["text", "reference"]:
                line["category_id"] = 1
            elif line["type"] == "figure":
                line["category_id"] = 3
            elif line["type"] == "figure_caption":
                line["category_id"] = 4
            elif line["type"] == "table":
                line["category_id"] = 5
            elif line["type"] == "table_caption":
                line["category_id"] = 6
            elif line["type"] == "equation":
                line["category_id"] = 8
            elif line["type"] in ["header", "footer"]:
                line["category_id"] = 2
            else:
                logger.warning(f"unknown type: {line['type']}")

            # 兼容不输出score的paddleocr版本
            if line.get("score") is None:
                line["score"] = 0.5 + random.random() * 0.5

            res = line.pop("res", None)
            if res is not None and len(res) > 0:
                for span in res:
                    new_span = {
                        "category_id": 15,
                        "bbox": region_to_bbox(span["text_region"]),
                        "score": span["confidence"],
                        "text": span["text"],
                    }
                    spans.append(new_span)

        if len(spans) > 0:
            result.extend(spans)

        return result
