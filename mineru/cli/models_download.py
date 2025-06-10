import json
import os
import sys
import click
import requests

from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


def download_json(url):
    """下载JSON文件"""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def download_and_modify_json(url, local_filename, modifications):
    """下载JSON并修改内容"""
    if os.path.exists(local_filename):
        data = json.load(open(local_filename))
        config_version = data.get('config_version', '0.0.0')
        if config_version < '1.3.0':
            data = download_json(url)
    else:
        data = download_json(url)

    # 修改内容
    for key, value in modifications.items():
        if key in data:
            if isinstance(data[key], dict):
                # 如果是字典，合并新值
                data[key].update(value)
            else:
                # 否则直接替换
                data[key] = value

    # 保存修改后的内容
    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def configure_model(model_dir, model_type):
    """配置模型"""
    json_url = 'https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/mineru.template.json'
    config_file_name = 'mineru.json'
    home_dir = os.path.expanduser('~')
    config_file = os.path.join(home_dir, config_file_name)

    json_mods = {
        'models-dir': {
            f'{model_type}': model_dir
        }
    }

    download_and_modify_json(json_url, config_file, json_mods)
    print(f'The configuration file has been successfully configured, the path is: {config_file}')


@click.command()
def download_models():
    """下载MinerU模型文件。

    支持从ModelScope或HuggingFace下载pipeline或VLM模型。
    """
    # 交互式输入下载来源
    source = click.prompt(
        "Please select the model download source: ",
        type=click.Choice(['huggingface', 'modelscope']),
        default='huggingface'
    )

    os.environ['MINERU_MODEL_SOURCE'] = source

    # 交互式输入模型类型
    model_type = click.prompt(
        "Please select the model type to download: ",
        type=click.Choice(['pipeline', 'vlm']),
        default='pipeline'
    )

    click.echo(f"Downloading {model_type} model from {source}...")

    try:
        download_finish_path = ""
        if model_type == 'pipeline':
            for model_path in [ModelPath.doclayout_yolo, ModelPath.yolo_v8_mfd, ModelPath.unimernet_small, ModelPath.pytorch_paddle, ModelPath.layout_reader, ModelPath.slanet_plus]:
                click.echo(f"Downloading model: {model_path}")
                download_finish_path = auto_download_and_get_model_root_path(model_path, repo_mode=model_type)
        elif model_type == 'vlm':
            download_finish_path = auto_download_and_get_model_root_path("/", repo_mode=model_type)
        click.echo(f"Models downloaded successfully to: {download_finish_path}")
        configure_model(download_finish_path, model_type)
    except Exception as e:
        click.echo(f"Download failed: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    download_models()
