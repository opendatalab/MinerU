import json
import os

import fitz
import pytest

from magic_pdf.data.data_reader_writer import (MultiBucketS3DataReader,
                                               MultiBucketS3DataWriter)
from magic_pdf.data.schemas import S3Config


@pytest.mark.skipif(
    os.getenv('S3_ACCESS_KEY_2', None) is None, reason='need s3 config!'
)
def test_multi_bucket_s3_reader_writer():
    """test multi bucket s3 reader writer must config s3 config in the
    environment export S3_BUCKET=xxx export S3_ACCESS_KEY=xxx export
    S3_SECRET_KEY=xxx export S3_ENDPOINT=xxx.

    export S3_BUCKET_2=xxx export S3_ACCESS_KEY_2=xxx export S3_SECRET_KEY_2=xxx export S3_ENDPOINT_2=xxx
    """
    bucket = os.getenv('S3_BUCKET', '')
    ak = os.getenv('S3_ACCESS_KEY', '')
    sk = os.getenv('S3_SECRET_KEY', '')
    endpoint_url = os.getenv('S3_ENDPOINT', '')

    bucket_2 = os.getenv('S3_BUCKET_2', '')
    ak_2 = os.getenv('S3_ACCESS_KEY_2', '')
    sk_2 = os.getenv('S3_SECRET_KEY_2', '')
    endpoint_url_2 = os.getenv('S3_ENDPOINT_2', '')

    s3configs = [
        S3Config(
            bucket_name=bucket, access_key=ak, secret_key=sk, endpoint_url=endpoint_url
        ),
        S3Config(
            bucket_name=bucket_2,
            access_key=ak_2,
            secret_key=sk_2,
            endpoint_url=endpoint_url_2,
        ),
    ]

    reader = MultiBucketS3DataReader(bucket, s3configs)
    writer = MultiBucketS3DataWriter(bucket, s3configs)

    bits = reader.read('meta-index/scihub/v001/scihub/part-66210c190659-000026.jsonl')

    assert bits == reader.read(
        f's3://{bucket}/meta-index/scihub/v001/scihub/part-66210c190659-000026.jsonl'
    )

    bits = reader.read(
        f's3://{bucket_2}/enbook-scimag/78800000/libgen.scimag78872000-78872999/10.1017/cbo9780511770425.012.pdf'
    )
    docs = fitz.open('pdf', bits)
    assert len(docs) == 10

    bits = reader.read(
        'meta-index/scihub/v001/scihub/part-66210c190659-000026.jsonl?bytes=566,713'
    )
    assert bits == reader.read_at(
        'meta-index/scihub/v001/scihub/part-66210c190659-000026.jsonl', 566, 713
    )
    assert len(json.loads(bits)) > 0

    writer.write_string(
        'unittest/data/data_reader_writer/multi_bucket_s3_data/test01.txt', 'abc'
    )

    assert 'abc'.encode() == reader.read(
        'unittest/data/data_reader_writer/multi_bucket_s3_data/test01.txt'
    )

    writer.write(
        'unittest/data/data_reader_writer/multi_bucket_s3_data/test02.txt',
        '123'.encode(),
    )

    assert '123'.encode() == reader.read(
        'unittest/data/data_reader_writer/multi_bucket_s3_data/test02.txt'
    )


@pytest.mark.skipif(
    os.getenv('S3_ACCESS_KEY_2', None) is None, reason='need s3 config!'
)
def test_multi_bucket_s3_reader_writer_with_prefix():
    """test multi bucket s3 reader writer must config s3 config in the
    environment export S3_BUCKET=xxx export S3_ACCESS_KEY=xxx export
    S3_SECRET_KEY=xxx export S3_ENDPOINT=xxx.

    export S3_BUCKET_2=xxx export S3_ACCESS_KEY_2=xxx export S3_SECRET_KEY_2=xxx export S3_ENDPOINT_2=xxx
    """
    bucket = os.getenv('S3_BUCKET', '')
    ak = os.getenv('S3_ACCESS_KEY', '')
    sk = os.getenv('S3_SECRET_KEY', '')
    endpoint_url = os.getenv('S3_ENDPOINT', '')

    bucket_2 = os.getenv('S3_BUCKET_2', '')
    ak_2 = os.getenv('S3_ACCESS_KEY_2', '')
    sk_2 = os.getenv('S3_SECRET_KEY_2', '')
    endpoint_url_2 = os.getenv('S3_ENDPOINT_2', '')

    s3configs = [
        S3Config(
            bucket_name=bucket, access_key=ak, secret_key=sk, endpoint_url=endpoint_url
        ),
        S3Config(
            bucket_name=bucket_2,
            access_key=ak_2,
            secret_key=sk_2,
            endpoint_url=endpoint_url_2,
        ),
    ]

    prefix = 'meta-index'
    reader = MultiBucketS3DataReader(f'{bucket}/{prefix}', s3configs)
    writer = MultiBucketS3DataWriter(f'{bucket}/{prefix}', s3configs)

    bits = reader.read('scihub/v001/scihub/part-66210c190659-000026.jsonl')

    assert bits == reader.read(
        f's3://{bucket}/{prefix}/scihub/v001/scihub/part-66210c190659-000026.jsonl'
    )

    bits = reader.read(
        f's3://{bucket_2}/enbook-scimag/78800000/libgen.scimag78872000-78872999/10.1017/cbo9780511770425.012.pdf'
    )
    docs = fitz.open('pdf', bits)
    assert len(docs) == 10

    bits = reader.read(
        'scihub/v001/scihub/part-66210c190659-000026.jsonl?bytes=566,713'
    )
    assert bits == reader.read_at(
        'scihub/v001/scihub/part-66210c190659-000026.jsonl', 566, 713
    )
    assert len(json.loads(bits)) > 0

    writer.write_string(
        'unittest/data/data_reader_writer/multi_bucket_s3_data/test01.txt', 'abc'
    )

    assert 'abc'.encode() == reader.read(
        'unittest/data/data_reader_writer/multi_bucket_s3_data/test01.txt'
    )

    assert 'abc'.encode() == reader.read(
        f's3://{bucket}/{prefix}/unittest/data/data_reader_writer/multi_bucket_s3_data/test01.txt'
    )

    writer.write(
        'unittest/data/data_reader_writer/multi_bucket_s3_data/test02.txt',
        '123'.encode(),
    )

    assert '123'.encode() == reader.read(
        'unittest/data/data_reader_writer/multi_bucket_s3_data/test02.txt'
    )
