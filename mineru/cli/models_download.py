# Copyright (c) Opendatalab. All rights reserved.
from contextlib import contextmanager
import json
import os
import sys
import click
import requests
from loguru import logger

from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path

MODEL_SOURCE_ENV_VAR = 'MINERU_MODEL_SOURCE'
REMOTE_MODEL_SOURCES = ('huggingface', 'modelscope')


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
        if config_version < '1.3.1':
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
    config_file_name = os.getenv('MINERU_TOOLS_CONFIG_JSON', 'mineru.json')
    home_dir = os.path.expanduser('~')
    config_file = os.path.join(home_dir, config_file_name)

    json_mods = {
        'models-dir': {
            f'{model_type}': model_dir
        }
    }

    download_and_modify_json(json_url, config_file, json_mods)
    logger.info(f'Il file di configurazione è stato configurato con successo, il percorso è: {config_file}')


def download_pipeline_models():
    """下载Pipeline模型"""
    model_paths = [
        ModelPath.pp_doclayout_v2,
        ModelPath.unimernet_small,
        ModelPath.pytorch_paddle,
        ModelPath.slanet_plus,
        ModelPath.unet_structure,
        ModelPath.paddle_table_cls,
        ModelPath.paddle_orientation_classification,
        ModelPath.pp_formulanet_plus_m,
    ]
    download_finish_path = ""
    for model_path in model_paths:
        logger.info(f"Download del modello: {model_path}")
        download_finish_path = auto_download_and_get_model_root_path(model_path, repo_mode='pipeline')
    logger.info(f"Modelli Pipeline scaricati con successo in: {download_finish_path}")
    configure_model(download_finish_path, "pipeline")


def download_vlm_models():
    """下载VLM模型"""
    download_finish_path = auto_download_and_get_model_root_path("/", repo_mode='vlm')
    logger.info(f"Modelli VLM scaricati con successo in: {download_finish_path}")
    configure_model(download_finish_path, "vlm")


def get_effective_download_model_source(requested_model_source):
    """获取本次下载命令实际使用的模型源。"""
    current_model_source = os.getenv(MODEL_SOURCE_ENV_VAR)
    if current_model_source == 'local':
        logger.warning(
            f"{MODEL_SOURCE_ENV_VAR}=local significa utilizzare modelli locali scaricati in precedenza. "
            f"`mineru-models-download` utilizzerà temporaneamente '{requested_model_source}' "
            f"per eseguire un download reale."
        )
        return requested_model_source

    if current_model_source is None:
        return requested_model_source

    return current_model_source


@contextmanager
def temporary_model_source(model_source):
    """在命令执行期间临时设置模型源，并在结束后恢复。"""
    original_model_source = os.getenv(MODEL_SOURCE_ENV_VAR)
    os.environ[MODEL_SOURCE_ENV_VAR] = model_source
    try:
        yield
    finally:
        if original_model_source is None:
            os.environ.pop(MODEL_SOURCE_ENV_VAR, None)
        else:
            os.environ[MODEL_SOURCE_ENV_VAR] = original_model_source


@click.command()
@click.option(
    '-s',
    '--source',
    'model_source',
    type=click.Choice(REMOTE_MODEL_SOURCES),
    help="""
        The source of the model repository. 
        """,
    default=None,
)
@click.option(
    '-m',
    '--model_type',
    'model_type',
    type=click.Choice(['pipeline', 'vlm', 'all']),
    help="""
    help="Il tipo di modello da scaricare.",
)
def download_models(model_source, model_type):
    """Scarica i file dei modelli MinerU.

    Supporta il download di modelli pipeline o VLM da ModelScope o HuggingFace.
    """
    # Se non specificato esplicitamente, richiede l'origine del download in modo interattivo
    if model_source is None:
        model_source = click.prompt(
            "Seleziona l'origine del download del modello: ",
            type=click.Choice(REMOTE_MODEL_SOURCES),
            default='huggingface'
        )

    effective_model_source = get_effective_download_model_source(model_source)

    # Se non specificato esplicitamente, richiede il tipo di modello in modo interattivo
    if model_type is None:
        model_type = click.prompt(
            "Seleziona il tipo di modello da scaricare: ",
            type=click.Choice(['pipeline', 'vlm', 'all']),
            default='all'
        )

    logger.info(f"Download del modello {model_type} da {effective_model_source}...")

    try:
        with temporary_model_source(effective_model_source):
            if model_type == 'pipeline':
                download_pipeline_models()
            elif model_type == 'vlm':
                download_vlm_models()
            elif model_type == 'all':
                download_pipeline_models()
                download_vlm_models()
            else:
                click.echo(f"Tipo di modello non supportato: {model_type}", err=True)
                sys.exit(1)

    except Exception as e:
        logger.exception(f"Si è verificato un errore durante il download dei modelli: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    download_models()
