

from s3pathlib import S3Path

def remove_non_official_s3_args(s3path):
    """
    example: s3://abc/xxxx.json?bytes=0,81350 ==> s3://abc/xxxx.json
    """
    arr = s3path.split("?")
    return arr[0]

def parse_s3path(s3path: str):
    p = S3Path(remove_non_official_s3_args(s3path))
    return p.bucket, p.key

def parse_s3_range_params(s3path: str):
    """
    example: s3://abc/xxxx.json?bytes=0,81350 ==> [0, 81350]
    """
    arr = s3path.split("?bytes=")
    if len(arr) == 1:
        return None
    return arr[1].split(",")
