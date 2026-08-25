# Copyright (c) Opendatalab. All rights reserved.
import hashlib


def str_sha256(input_string: str) -> str:
    hasher = hashlib.sha256()
    # 在Python3中，需要将字符串转化为字节对象才能被哈希函数处理
    input_bytes = input_string.encode("utf-8")
    hasher.update(input_bytes)
    return hasher.hexdigest()
