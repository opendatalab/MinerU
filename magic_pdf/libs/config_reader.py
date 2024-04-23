"""
根据bucket的名字返回对应的s3 AK， SK，endpoint三元组

"""

import json
import os

from loguru import logger

from magic_pdf.libs.commons import parse_bucket_key


def read_config():
    home_dir = os.path.expanduser("~")

    config_file = os.path.join(home_dir, "magic-pdf.json")

    if not os.path.exists(config_file):
        raise Exception(f"{config_file} not found")

    with open(config_file, "r") as f:
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
        raise Exception("ak, sk or endpoint not found in magic-pdf.json")

    # logger.info(f"get_s3_config: ak={access_key}, sk={secret_key}, endpoint={storage_endpoint}")

    return access_key, secret_key, storage_endpoint


def get_s3_config_dict(path: str):
    access_key, secret_key, storage_endpoint = get_s3_config(get_bucket_name(path))
    return {"ak": access_key, "sk": secret_key, "endpoint": storage_endpoint}


def get_bucket_name(path):
    bucket, key = parse_bucket_key(path)
    return bucket


def get_local_dir():
    config = read_config()
    return config.get("temp-output-dir", "/tmp")


if __name__ == "__main__":
    ak, sk, endpoint = get_s3_config("llm-raw")
