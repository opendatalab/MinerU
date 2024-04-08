"""
根据bucket的名字返回对应的s3 AK， SK，endpoint三元组

"""
import json
import os

from loguru import logger


def get_s3_config(bucket_name: str):
    """
    ~/magic_pdf_config.json 读出来
    """
    if os.name == "posix":  # Linux or macOS
        home_dir = os.path.expanduser("~")
    elif os.name == "nt":  # Windows
        home_dir = os.path.expandvars("%USERPROFILE%")
    else:
        raise Exception("Unsupported operating system")

    config_file = os.path.join(home_dir, "magic_pdf_config.json")

    if not os.path.exists(config_file):
        raise Exception("magic_pdf_config.json not found")

    with open(config_file, "r") as f:
        config = json.load(f)

    if bucket_name not in config:
        raise Exception("bucket_name not found in magic_pdf_config.json")

    access_key = config[bucket_name].get("ak")
    secret_key = config[bucket_name].get("sk")
    storage_endpoint = config[bucket_name].get("endpoint")

    if access_key is None or secret_key is None or storage_endpoint is None:
        raise Exception("ak, sk or endpoint not found in magic_pdf_config.json")

    # logger.info(f"get_s3_config: ak={access_key}, sk={secret_key}, endpoint={storage_endpoint}")

    return access_key, secret_key, storage_endpoint


if __name__ == '__main__':
    ak, sk, endpoint = get_s3_config("llm-raw")
