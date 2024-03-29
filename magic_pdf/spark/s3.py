# from app.common import s3
import boto3
from botocore.client import Config

import re
import random
from typing import List, Union
try:
    from app.config import s3_buckets, s3_clusters, s3_users
    from app.common.runtime import get_cluster_name
except ImportError:
    from magic_pdf.config import s3_buckets, s3_clusters, get_cluster_name, s3_users

__re_s3_path = re.compile("^s3a?://([^/]+)(?:/(.*))?$")
def get_s3_config(path: Union[str, List[str]], outside=False):
    paths = [path] if type(path) == str else path
    bucket_config = None
    for p in paths:
        bc = __get_s3_bucket_config(p)
        if bucket_config in [bc, None]:
            bucket_config = bc
            continue
        raise Exception(f"{paths} have different s3 config, cannot read together.")
    if not bucket_config:
        raise Exception("path is empty.")
    return __get_s3_config(bucket_config, outside, prefer_ip=True)

def __get_s3_config(
    bucket_config: tuple,
    outside: bool,
    prefer_ip=False,
    prefer_auto=False,
):
    cluster, user = bucket_config
    cluster_config = s3_clusters[cluster]

    if outside:
        endpoint_key = "outside"
    elif prefer_auto and "auto" in cluster_config:
        endpoint_key = "auto"
    elif cluster_config.get("cluster") == get_cluster_name():
        endpoint_key = "inside"
    else:
        endpoint_key = "outside"

    if prefer_ip and f"{endpoint_key}_ips" in cluster_config:
        endpoint_key = f"{endpoint_key}_ips"

    endpoints = cluster_config[endpoint_key]
    endpoint = random.choice(endpoints)
    return {"endpoint": endpoint, **s3_users[user]}

def split_s3_path(path: str):
    "split bucket and key from path"
    m = __re_s3_path.match(path)
    if m is None:
        return "", ""
    return m.group(1), (m.group(2) or "")

def __get_s3_bucket_config(path: str):
    bucket = split_s3_path(path)[0] if path else ""
    bucket_config = s3_buckets.get(bucket)
    if not bucket_config:
        bucket_config = s3_buckets.get("[default]")
        assert bucket_config is not None
    return bucket_config

def get_s3_client(path: Union[str, List[str]], outside=False):
    s3_config = get_s3_config(path, outside)
    try:
        return boto3.client(
            "s3",
            aws_access_key_id=s3_config["ak"],
            aws_secret_access_key=s3_config["sk"],
            endpoint_url=s3_config["endpoint"],
            config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 8, "mode": "standard"}),
        )
    except:
        # older boto3 do not support retries.mode param.
        return boto3.client(
            "s3",
            aws_access_key_id=s3_config["ak"],
            aws_secret_access_key=s3_config["sk"],
            endpoint_url=s3_config["endpoint"],
            config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 8}),
        )