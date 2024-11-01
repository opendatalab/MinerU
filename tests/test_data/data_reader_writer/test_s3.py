import json
import os

import pytest

from magic_pdf.data.data_reader_writer import S3DataReader, S3DataWriter


@pytest.mark.skipif(
    os.getenv('S3_ACCESS_KEY', None) is None, reason='need s3 config!'
)
def test_s3_reader_writer():
    """test multi bucket s3 reader writer must config s3 config in the
    environment export S3_BUCKET=xxx export S3_ACCESS_KEY=xxx export
    S3_SECRET_KEY=xxx export S3_ENDPOINT=xxx."""
    bucket = os.getenv('S3_BUCKET', '')
    ak = os.getenv('S3_ACCESS_KEY', '')
    sk = os.getenv('S3_SECRET_KEY', '')
    endpoint_url = os.getenv('S3_ENDPOINT', '')

    reader = S3DataReader('', bucket, ak, sk, endpoint_url)
    writer = S3DataWriter('', bucket, ak, sk, endpoint_url)

    bits = reader.read('meta-index/scihub/v001/scihub/part-66210c190659-000026.jsonl')

    assert bits == reader.read(
        f's3://{bucket}/meta-index/scihub/v001/scihub/part-66210c190659-000026.jsonl'
    )

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
        f'{bucket}/unittest/data/data_reader_writer/multi_bucket_s3_data/test02.txt',
        '123'.encode(),
    )

    assert '123'.encode() == reader.read(
        'unittest/data/data_reader_writer/multi_bucket_s3_data/test02.txt'
    )


@pytest.mark.skipif(
    os.getenv('S3_ACCESS_KEY', None) is None, reason='need s3 config!'
)
def test_s3_reader_writer_with_prefix():
    """test multi bucket s3 reader writer must config s3 config in the
    environment export S3_BUCKET=xxx export S3_ACCESS_KEY=xxx export
    S3_SECRET_KEY=xxx export S3_ENDPOINT=xxx."""
    bucket = os.getenv('S3_BUCKET', '')
    ak = os.getenv('S3_ACCESS_KEY', '')
    sk = os.getenv('S3_SECRET_KEY', '')
    endpoint_url = os.getenv('S3_ENDPOINT', '')

    prefix = 'meta-index'

    reader = S3DataReader(prefix, bucket, ak, sk, endpoint_url)
    writer = S3DataWriter(prefix, bucket, ak, sk, endpoint_url)

    bits = reader.read('scihub/v001/scihub/part-66210c190659-000026.jsonl')

    assert bits == reader.read(
        f's3://{bucket}/{prefix}/scihub/v001/scihub/part-66210c190659-000026.jsonl'
    )

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
        f'{bucket}/{prefix}/unittest/data/data_reader_writer/multi_bucket_s3_data/test02.txt',
        '123'.encode(),
    )

    assert '123'.encode() == reader.read(
        'unittest/data/data_reader_writer/multi_bucket_s3_data/test02.txt'
    )
