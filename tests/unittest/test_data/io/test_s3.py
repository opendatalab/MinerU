import json
import os

import pytest

from magic_pdf.data.io.s3 import S3Reader, S3Writer


@pytest.mark.skipif(
    os.getenv('S3_ACCESS_KEY', None) is None, reason='s3 config not found'
)
def test_s3_reader():
    """test s3 reader.

    must config s3 config in the environment export S3_BUCKET=xxx export S3_ACCESS_KEY=xxx export S3_SECRET_KEY=xxx
    export S3_ENDPOINT=xxx
    """

    bucket = os.getenv('S3_BUCKET', '')
    ak = os.getenv('S3_ACCESS_KEY', '')
    sk = os.getenv('S3_SECRET_KEY', '')
    endpoint_url = os.getenv('S3_ENDPOINT', '')
    reader = S3Reader(bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint_url)
    bits = reader.read(
        'meta-index/scihub/v001/scihub/part-66210c190659-000026.jsonl'
    )
    assert len(bits) > 0

    bits = reader.read_at(
        'meta-index/scihub/v001/scihub/part-66210c190659-000026.jsonl',
        566,
        713,
    )
    assert len(json.loads(bits)) > 0


@pytest.mark.skipif(
    os.getenv('S3_ACCESS_KEY', None) is None, reason='s3 config not found'
)
def test_s3_writer():
    """test s3 reader.

    must config s3 config in the environment export S3_BUCKET=xxx export S3_ACCESS_KEY=xxx export S3_SECRET_KEY=xxx
    export S3_ENDPOINT=xxx
    """
    bucket = os.getenv('S3_BUCKET', '')
    ak = os.getenv('S3_ACCESS_KEY', '')
    sk = os.getenv('S3_SECRET_KEY', '')
    endpoint_url = os.getenv('S3_ENDPOINT', '')
    writer = S3Writer(bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint_url)
    test_fn = 'unittest/io/test.jsonl'
    writer.write(test_fn, '123'.encode())
    reader = S3Reader(bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint_url)
    bits = reader.read(test_fn)
    assert bits.decode() == '123'
