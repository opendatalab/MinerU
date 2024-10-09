import os
import requests
import json
from huggingface_hub import snapshot_download


def download_and_modify_json(url, local_filename, modifications):
    if os.path.exists(local_filename):
        data = json.load(open(local_filename))
    else:
        # 下载JSON文件
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功

        # 解析JSON内容
        data = response.json()

    # 修改内容
    for key, value in modifications.items():
        data[key] = value

    # 保存修改后的内容
    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    model_dir = snapshot_download('opendatalab/PDF-Extract-Kit')
    layoutreader_model_dir = snapshot_download('hantian/layoutreader')
    model_dir = model_dir + "/models"
    print(f"model_dir is: {model_dir}")
    print(f"layoutreader_model_dir is: {layoutreader_model_dir}")

    json_url = 'https://github.com/opendatalab/MinerU/raw/master/magic-pdf.template.json'
    config_file_name = "magic-pdf.json"
    home_dir = os.path.expanduser("~")
    config_file = os.path.join(home_dir, config_file_name)

    json_mods = {
        'models-dir': model_dir,
        'layoutreader-model-dir': layoutreader_model_dir,
    }

    download_and_modify_json(json_url, config_file, json_mods)
    print(f"The configuration file has been configured successfully, the path is: {config_file}")