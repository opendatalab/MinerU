"""
根据bucket的名字返回对应的s3 AK， SK，endpoint三元组

"""

import json
import os

from loguru import logger

from magic_pdf.libs.commons import parse_bucket_key

# 定义配置文件名常量
CONFIG_FILE_NAME = "magic-pdf.json"


def read_config():
    home_dir = os.path.expanduser("~")

    config_file = os.path.join(home_dir, CONFIG_FILE_NAME)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def get_s3_config(bucket_name: str):
    """
    ~/magic-pdf.json 读出来
    """
    config = read_config()

    bucket_info = config.get("bucket_info")
    if bucket_name not in bucket_info:
        access_key, secret_key, storage_endpoint = bucket_info["[default]"]
    else:
        access_key, secret_key, storage_endpoint = bucket_info[bucket_name]

    if access_key is None or secret_key is None or storage_endpoint is None:
        raise Exception(f"ak, sk or endpoint not found in {CONFIG_FILE_NAME}")

    # logger.info(f"get_s3_config: ak={access_key}, sk={secret_key}, endpoint={storage_endpoint}")

    return access_key, secret_key, storage_endpoint


def get_s3_config_dict(path: str):
    access_key, secret_key, storage_endpoint = get_s3_config(get_bucket_name(path))
    return {"ak": access_key, "sk": secret_key, "endpoint": storage_endpoint}


def get_bucket_name(path):
    bucket, key = parse_bucket_key(path)
    return bucket


def get_local_models_dir():
    config = read_config()
    models_dir = config.get("models-dir")
    if models_dir is None:
        logger.warning(f"'models-dir' not found in {CONFIG_FILE_NAME}, use '/tmp/models' as default")
        return "/tmp/models"
    else:
        return models_dir


def get_device():
    config = read_config()
    device = config.get("device-mode")
    if device is None:
        logger.warning(f"'device-mode' not found in {CONFIG_FILE_NAME}, use 'cpu' as default")
        return "cpu"
    else:
        return device


def get_table_recog_config():
    config = read_config()
    table_config = config.get("table-config")
    if table_config is None:
        logger.warning(f"'table-config' not found in {CONFIG_FILE_NAME}, use 'False' as default")
        return json.loads('{"is_table_recog_enable": false, "max_time": 400}')
    else:
        return table_config


if __name__ == "__main__":
    ak, sk, endpoint = get_s3_config("llm-raw")
