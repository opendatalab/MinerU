

def remove_non_official_s3_args(s3path):
    """
    example: s3://abc/xxxx.json?bytes=0,81350 ==> s3://abc/xxxx.json
    """
    arr = s3path.split("?")
    return arr[0]

def parse_s3path(s3path: str):
    # from s3pathlib import S3Path
    # p = S3Path(remove_non_official_s3_args(s3path))
    # return p.bucket, p.key
    s3path = remove_non_official_s3_args(s3path).strip()
    if s3path.startswith(('s3://', 's3a://')):
        prefix, path = s3path.split('://', 1)
        bucket_name, key = path.split('/', 1)
        return bucket_name, key
    elif s3path.startswith('/'):
        raise ValueError("The provided path starts with '/'. This does not conform to a valid S3 path format.")
    else:
        raise ValueError("Invalid S3 path format. Expected 's3://bucket-name/key' or 's3a://bucket-name/key'.")


def parse_s3_range_params(s3path: str):
    """
    example: s3://abc/xxxx.json?bytes=0,81350 ==> [0, 81350]
    """
    arr = s3path.split("?bytes=")
    if len(arr) == 1:
        return None
    return arr[1].split(",")
