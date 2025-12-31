import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from loguru import logger

from mineru.cli.common import read_fn
from mineru.data.utils.path_utils import parse_s3path

# ------- optional s3 backends (lazy import) -------
try:
    from mineru.data.io.s3 import S3Reader
    from mineru.data.data_reader_writer.s3 import S3DataWriter
except Exception:
    S3Reader = None
    S3DataWriter = None


def _require_s3_backend():
    if S3Reader is None or S3DataWriter is None:
        raise ImportError(
            "S3 backend not installed. Please run: pip install mineru[s3]"
        )


def get_s3_env_config() -> Dict[str, str]:
    
    ak = os.getenv("AWS_ACCESS_KEY_ID")
    sk = os.getenv("AWS_SECRET_ACCESS_KEY")
    endpoint = os.getenv("S3_ENDPOINT_URL")
    addressing_style = os.getenv("S3_ADDRESSING_STYLE", "auto")

    if not all([ak, sk, endpoint]):
        raise ValueError(
            "S3 credentials not configured. Please set: "
            "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT_URL"
        )
    return {"ak": ak, "sk": sk, "endpoint": endpoint, "addressing_style": addressing_style}


def read_bytes_from_uri(input_uri: str) -> bytes:
    """
    读取输入资源。当前版本只支持本地路径和 s3:// / s3a://。
    其他带 scheme 的 URI（如 http/https/file）如需支持可在此处拓展。
    """
    if input_uri.startswith(("s3://", "s3a://")):
        _require_s3_backend()
        cfg = get_s3_env_config()
        bucket, key = parse_s3path(input_uri)
        logger.info(f"Reading from S3: s3://{bucket}/{key}")
        reader = S3Reader(
            bucket=bucket,
            ak=cfg["ak"],
            sk=cfg["sk"],
            endpoint_url=cfg["endpoint"],
            addressing_style=cfg["addressing_style"],
        )
        return reader.read(key)

    if "://" in input_uri:
        # 非 s3 的 scheme，当前版本不支持（http/https/file 等可在后续扩展）
        scheme = input_uri.split("://", 1)[0]
        raise ValueError(
            f"Unsupported URI scheme: {scheme}. Only local paths and s3:// are supported."
        )

    # local path
    return read_fn(Path(input_uri))


def prepare_output_dir(output_uri: Optional[str], fallback_local_dir: str) -> Tuple[str, bool, Optional[str]]:
    """
    根据 output_uri 决定实际写入目录：
    - s3://...  => 写 temp dir，之后上传
    - local/None => 写本地 fallback_local_dir
    返回 (actual_output_dir, is_s3_output, normalized_output_uri)
    """
    if output_uri and output_uri.startswith(("s3://", "s3a://")):
        tmp = tempfile.mkdtemp(prefix="mineru_")
        return tmp, True, output_uri

    # local
    os.makedirs(fallback_local_dir, exist_ok=True)
    return fallback_local_dir, False, output_uri or fallback_local_dir


def upload_parse_dir_to_s3(local_parse_dir: str, output_uri: str) -> str:
    """
    把 local_parse_dir 整体上传到 output_uri 指定的 s3 prefix.
    返回最终 s3 parse_dir（供 API 返回给客户端）
    """
    _require_s3_backend()
    cfg = get_s3_env_config()
    bucket, prefix = parse_s3path(output_uri)
    prefix = prefix.rstrip("/")

    writer = S3DataWriter(
        default_prefix_without_bucket=prefix,
        bucket=bucket,
        ak=cfg["ak"],
        sk=cfg["sk"],
        endpoint_url=cfg["endpoint"],
        addressing_style=cfg["addressing_style"],
    )

    count = 0
    for root, _dirs, files in os.walk(local_parse_dir):
        for f in files:
            lp = os.path.join(root, f)
            rel = os.path.relpath(lp, local_parse_dir).replace("\\", "/")
            with open(lp, "rb") as fp:
                writer.write(rel, fp.read())
                count += 1

    logger.info(f"Uploaded {count} files to {output_uri}")
    # 返回 parse_dir（s3 语义）
    return f"s3://{bucket}/{prefix}/"
    

def cleanup_temp_dir(tmp_dir: str):
    try:
        shutil.rmtree(tmp_dir)
    except Exception as e:
        logger.warning(f"cleanup temp dir failed: {e}")
