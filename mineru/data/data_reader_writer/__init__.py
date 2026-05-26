# Copyright (c) Opendatalab. All rights reserved.
from .base import DataReader, DataWriter
from .dummy import DummyDataWriter
from .filebase import FileBasedDataReader, FileBasedDataWriter

__all__ = [
    "DataReader",
    "DataWriter",
    "FileBasedDataReader",
    "FileBasedDataWriter",
    "S3DataReader",
    "S3DataWriter",
    "MultiBucketS3DataReader",
    "MultiBucketS3DataWriter",
    "DummyDataWriter",
]


def __getattr__(name):
    """按需加载 S3 DataReader/DataWriter，避免默认安装场景强制依赖 boto3。"""
    if name in {"MultiBucketS3DataReader", "MultiBucketS3DataWriter"}:
        from .multi_bucket_s3 import MultiBucketS3DataReader, MultiBucketS3DataWriter

        s3_exports = {
            "MultiBucketS3DataReader": MultiBucketS3DataReader,
            "MultiBucketS3DataWriter": MultiBucketS3DataWriter,
        }
        globals().update(s3_exports)
        return s3_exports[name]

    if name in {"S3DataReader", "S3DataWriter"}:
        from .s3 import S3DataReader, S3DataWriter

        s3_exports = {"S3DataReader": S3DataReader, "S3DataWriter": S3DataWriter}
        globals().update(s3_exports)
        return s3_exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
