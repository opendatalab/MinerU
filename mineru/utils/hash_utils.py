# Copyright (c) Opendatalab. All rights reserved.
import hashlib
import json


def bytes_md5(file_bytes):
    hasher = hashlib.md5()
    hasher.update(file_bytes)
    return hasher.hexdigest().upper()


def str_md5(input_string):
    hasher = hashlib.md5()
    # In Python 3, strings need to be converted to byte objects for hash functions to process
    input_bytes = input_string.encode('utf-8')
    hasher.update(input_bytes)
    return hasher.hexdigest()


def str_sha256(input_string):
    hasher = hashlib.sha256()
    # In Python 3, strings need to be converted to byte objects for hash functions to process
    input_bytes = input_string.encode('utf-8')
    hasher.update(input_bytes)
    return hasher.hexdigest()


def dict_md5(d):
    json_str = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(json_str.encode('utf-8')).hexdigest()