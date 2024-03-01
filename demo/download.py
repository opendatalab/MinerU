import json
import os
from tqdm import tqdm

from pdf_tools.libs import join_path

with open('/mnt/petrelfs/share_data/ouyanglinke/OCR/OCR_validation_dataset.json', 'r') as f:
    samples = json.load(f)

pdf_model_dir = 's3://llm-pdf-text/eval_1k/layout_res/'

labels = []
det_res = []
edit_distance_list = []
for sample in tqdm(samples):
    pdf_name = sample['pdf_name']
    page_num = sample['page']
    pdf_model_path = join_path(pdf_model_dir, pdf_name)
    model_output_json = join_path(pdf_model_path, f"page_{page_num}.json") # 模型输出的页面编号从1开始的
    save_root_path = '/mnt/petrelfs/share_data/ouyanglinke/OCR/OCR_val_docxchain/'
    save_path = join_path(save_root_path, pdf_name)
    os.makedirs(save_path, exist_ok=True)
    # print("s3c cp {} {}".format(model_output_json, save_path))
    os.system("aws --profile langchao --endpoint-url=http://10.140.85.161:80 s3 cp {} {}".format(model_output_json, save_path))
