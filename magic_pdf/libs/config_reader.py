"""
根据bucket的名字返回对应的s3 AK， SK，endpoint三元组

"""
import json
import os

from loguru import logger


def get_s3_config(bucket_name: str):
    """
    ~/magic-pdf.json 读出来
    """

    home_dir = os.path.expanduser("~")

    config_file = os.path.join(home_dir, "magic-pdf.json")

    if not os.path.exists(config_file):
        raise Exception("magic-pdf.json not found")

    with open(config_file, "r") as f:
        config = json.load(f)

    bucket_info = config.get("bucket_info")
    if bucket_name not in bucket_info:
        raise Exception("bucket_name not found in magic-pdf.json")

    access_key, secret_key, storage_endpoint = bucket_info[bucket_name]

    if access_key is None or secret_key is None or storage_endpoint is None:
        raise Exception("ak, sk or endpoint not found in magic-pdf.json")

    # logger.info(f"get_s3_config: ak={access_key}, sk={secret_key}, endpoint={storage_endpoint}")

    return access_key, secret_key, storage_endpoint


if __name__ == '__main__':
    ak, sk, endpoint = get_s3_config("llm-raw")
