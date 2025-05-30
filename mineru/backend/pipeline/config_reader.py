# Copyright (c) Opendatalab. All rights reserved.
import json
import os

from loguru import logger

# 定义配置文件名常量
CONFIG_FILE_NAME = os.getenv('MINERU_TOOLS_CONFIG_JSON', 'magic-pdf.json')


def read_config():
    if os.path.isabs(CONFIG_FILE_NAME):
        config_file = CONFIG_FILE_NAME
    else:
        home_dir = os.path.expanduser('~')
        config_file = os.path.join(home_dir, CONFIG_FILE_NAME)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f'{config_file} not found')

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def get_s3_config(bucket_name: str):
    """~/magic-pdf.json 读出来."""
    config = read_config()

    bucket_info = config.get('bucket_info')
    if bucket_name not in bucket_info:
        access_key, secret_key, storage_endpoint = bucket_info['[default]']
    else:
        access_key, secret_key, storage_endpoint = bucket_info[bucket_name]

    if access_key is None or secret_key is None or storage_endpoint is None:
        raise Exception(f'ak, sk or endpoint not found in {CONFIG_FILE_NAME}')

    # logger.info(f"get_s3_config: ak={access_key}, sk={secret_key}, endpoint={storage_endpoint}")

    return access_key, secret_key, storage_endpoint


def get_s3_config_dict(path: str):
    access_key, secret_key, storage_endpoint = get_s3_config(get_bucket_name(path))
    return {'ak': access_key, 'sk': secret_key, 'endpoint': storage_endpoint}


def get_bucket_name(path):
    bucket, key = parse_bucket_key(path)
    return bucket


def parse_bucket_key(s3_full_path: str):
    """
    输入 s3://bucket/path/to/my/file.txt
    输出 bucket, path/to/my/file.txt
    """
    s3_full_path = s3_full_path.strip()
    if s3_full_path.startswith("s3://"):
        s3_full_path = s3_full_path[5:]
    if s3_full_path.startswith("/"):
        s3_full_path = s3_full_path[1:]
    bucket, key = s3_full_path.split("/", 1)
    return bucket, key


def get_local_models_dir():
    config = read_config()
    models_dir = config.get('models-dir')
    if models_dir is None:
        logger.warning(f"'models-dir' not found in {CONFIG_FILE_NAME}, use '/tmp/models' as default")
        return '/tmp/models'
    else:
        return models_dir


def get_local_layoutreader_model_dir():
    config = read_config()
    layoutreader_model_dir = config.get('layoutreader-model-dir')
    if layoutreader_model_dir is None or not os.path.exists(layoutreader_model_dir):
        home_dir = os.path.expanduser('~')
        layoutreader_at_modelscope_dir_path = os.path.join(home_dir, '.cache/modelscope/hub/ppaanngggg/layoutreader')
        logger.warning(f"'layoutreader-model-dir' not exists, use {layoutreader_at_modelscope_dir_path} as default")
        return layoutreader_at_modelscope_dir_path
    else:
        return layoutreader_model_dir


def get_device():
    config = read_config()
    device = config.get('device-mode')
    if device is None:
        logger.warning(f"'device-mode' not found in {CONFIG_FILE_NAME}, use 'cpu' as default")
        return 'cpu'
    else:
        return device


def get_table_recog_config():
    config = read_config()
    table_config = config.get('table-config')
    if table_config is None:
        logger.warning(f"'table-config' not found in {CONFIG_FILE_NAME}, use 'False' as default")
        return json.loads(f'{{"enable": true}}')
    else:
        return table_config


def get_formula_config():
    config = read_config()
    formula_config = config.get('formula-config')
    if formula_config is None:
        logger.warning(f"'formula-config' not found in {CONFIG_FILE_NAME}, use 'True' as default")
        return json.loads(f'{{"enable": true}}')
    else:
        return formula_config