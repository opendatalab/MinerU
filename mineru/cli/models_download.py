# Copyright (c) Opendatalab. All rights reserved.
from contextlib import contextmanager
import os
import sys
import click
from loguru import logger

from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import (
    CONFIG_TEMPLATE_URL,
    auto_download_and_get_model_root_path,
    download_and_modify_json,
    get_tools_config_file_path,
    resolve_model_source,
)

MODEL_SOURCE_ENV_VAR = 'MINERU_MODEL_SOURCE'
REMOTE_MODEL_SOURCES = ('auto', 'huggingface', 'modelscope')


def configure_model(model_dir, model_type, model_source):
    """配置模型"""
    config_file = get_tools_config_file_path()

    json_mods = {
        'models-dir': {
            f'{model_type}': model_dir
        },
        'model-source': model_source,
    }

    download_and_modify_json(CONFIG_TEMPLATE_URL, config_file, json_mods)
    logger.info(f'The configuration file has been successfully configured, the path is: {config_file}')


def download_pipeline_models(model_source):
    """下载Pipeline模型"""
    model_paths = [
        ModelPath.pp_doclayout_v2,
        ModelPath.unimernet_small,
        ModelPath.pytorch_paddle,
        ModelPath.slanet_plus,
        ModelPath.unet_structure,
        ModelPath.paddle_table_cls,
        ModelPath.pp_formulanet_plus_m,
    ]
    download_finish_path = ""
    for model_path in model_paths:
        logger.info(f"Downloading model: {model_path}")
        download_finish_path = auto_download_and_get_model_root_path(model_path, repo_mode='pipeline')
    logger.info(f"Pipeline models downloaded successfully to: {download_finish_path}")
    configure_model(download_finish_path, "pipeline", model_source)


def download_vlm_models(model_source):
    """下载VLM模型"""
    download_finish_path = auto_download_and_get_model_root_path("/", repo_mode='vlm')
    logger.info(f"VLM models downloaded successfully to: {download_finish_path}")
    configure_model(download_finish_path, "vlm", model_source)


def get_effective_download_model_source(requested_model_source):
    """获取本次下载命令实际使用的模型源。"""
    current_model_source = os.getenv(MODEL_SOURCE_ENV_VAR)
    if current_model_source == 'local':
        logger.warning(
            f"{MODEL_SOURCE_ENV_VAR}=local means using pre-downloaded local models. "
            f"`mineru-models-download` will temporarily use '{requested_model_source}' "
            f"to perform a real download."
        )
        return resolve_model_source(requested_model_source, allow_auto=True)

    if current_model_source is None:
        return resolve_model_source(requested_model_source, allow_auto=True)

    return resolve_model_source(current_model_source)


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
        The type of the model to download.
        """,
    default=None,
)
def download_models(model_source, model_type):
    """Download MinerU model files.

    Supports downloading pipeline or VLM models from ModelScope or HuggingFace.
    """
    # 如果未显式指定则交互式输入下载来源
    if model_source is None:
        model_source = click.prompt(
            "Please select the model download source: ",
            type=click.Choice(REMOTE_MODEL_SOURCES),
            default='auto'
        )

    effective_model_source = get_effective_download_model_source(model_source)

    # 如果未显式指定则交互式输入模型类型
    if model_type is None:
        model_type = click.prompt(
            "Please select the model type to download: ",
            type=click.Choice(['pipeline', 'vlm', 'all']),
            default='all'
        )

    logger.info(f"Downloading {model_type} model from {effective_model_source}...")

    try:
        with temporary_model_source(effective_model_source):
            if model_type == 'pipeline':
                download_pipeline_models(effective_model_source)
            elif model_type == 'vlm':
                download_vlm_models(effective_model_source)
            elif model_type == 'all':
                download_pipeline_models(effective_model_source)
                download_vlm_models(effective_model_source)
            else:
                click.echo(f"Unsupported model type: {model_type}", err=True)
                sys.exit(1)

    except Exception as e:
        logger.exception(f"An error occurred while downloading models: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    download_models()
