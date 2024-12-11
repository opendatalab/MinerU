import os

import pytest

from magic_pdf.data.data_reader_writer import MultiBucketS3DataReader
from magic_pdf.data.read_api import (read_jsonl, read_local_images,
                                     read_local_pdfs)
from magic_pdf.data.schemas import S3Config


def test_read_local_pdfs():
    datasets = read_local_pdfs('tests/unittest/test_data/assets/pdfs')
    assert len(datasets) == 2
    assert len(datasets[0]) > 0
    assert len(datasets[1]) > 0

    assert datasets[0].get_page(0).get_page_info().w > 0
    assert datasets[0].get_page(0).get_page_info().h > 0


def test_read_local_images():
    datasets = read_local_images('tests/unittest/test_data/assets/pngs', suffixes=['.png'])
    assert len(datasets) == 2
    assert len(datasets[0]) == 1
    assert len(datasets[1]) == 1

    assert datasets[0].get_page(0).get_page_info().w > 0
    assert datasets[0].get_page(0).get_page_info().h > 0


@pytest.mark.skipif(
    os.getenv('S3_ACCESS_KEY_2', None) is None, reason='need s3 config!'
)
def test_read_json():
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

    datasets = read_jsonl(
        f's3://{bucket}/meta-index/scihub/v001/scihub/part-66210c190659-000026.jsonl',
        reader,
    )
    assert len(datasets) > 0
    assert len(datasets[0]) == 10

    datasets = read_jsonl('tests/unittest/test_data/assets/jsonl/test_01.jsonl', reader)
    assert len(datasets) == 1
    assert len(datasets[0]) == 10

    datasets = read_jsonl('tests/unittest/test_data/assets/jsonl/test_02.jsonl')
    assert len(datasets) == 1
    assert len(datasets[0]) == 1
