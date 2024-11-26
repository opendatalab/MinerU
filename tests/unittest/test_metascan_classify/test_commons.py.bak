import io
import json
import os

import boto3
from botocore.config import Config

from magic_pdf.libs.commons import fitz
from magic_pdf.libs.config_reader import get_s3_config_dict

from magic_pdf.libs.commons import join_path, json_dump_path, read_file, parse_bucket_key
from loguru import logger

test_pdf_dir_path = "s3://llm-pdf-text/unittest/pdf/"


def get_test_pdf_json(book_name):
    json_path = join_path(json_dump_path, book_name + ".json")
    s3_config = get_s3_config_dict(json_path)
    file_content = read_file(json_path, s3_config)
    json_str = file_content.decode('utf-8')
    json_object = json.loads(json_str)
    return json_object


def read_test_file(book_name):
    test_pdf_path = join_path(test_pdf_dir_path, book_name + ".pdf")
    s3_config = get_s3_config_dict(test_pdf_path)
    try:
        file_content = read_file(test_pdf_path, s3_config)
        return file_content
    except Exception as e:
        if "NoSuchKey" in str(e):
            logger.warning("File not found in test_pdf_path. Downloading from orig_s3_pdf_path.")
            try:
                json_object = get_test_pdf_json(book_name)
                orig_s3_pdf_path = json_object.get('file_location')
                s3_config = get_s3_config_dict(orig_s3_pdf_path)
                file_content = read_file(orig_s3_pdf_path, s3_config)
                s3_client = get_s3_client(test_pdf_path)
                bucket_name, bucket_key = parse_bucket_key(test_pdf_path)
                file_obj = io.BytesIO(file_content)
                s3_client.upload_fileobj(file_obj, bucket_name, bucket_key)
                return file_content
            except Exception as e:
                logger.exception(e)
        else:
            logger.exception(e)


def get_docs_from_test_pdf(book_name):
    file_content = read_test_file(book_name)
    return fitz.open("pdf", file_content)


def get_test_json_data(directory_path, json_file_name):
    with open(os.path.join(directory_path, json_file_name), "r", encoding='utf-8') as f:
        test_data = json.load(f)
    return test_data


def get_s3_client(path):
    s3_config = get_s3_config_dict(path)
    try:
        return boto3.client(
            "s3",
            aws_access_key_id=s3_config["ak"],
            aws_secret_access_key=s3_config["sk"],
            endpoint_url=s3_config["endpoint"],
            config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 8, "mode": "standard"}),
        )
    except:
        # older boto3 do not support retries.mode param.
        return boto3.client(
            "s3",
            aws_access_key_id=s3_config["ak"],
            aws_secret_access_key=s3_config["sk"],
            endpoint_url=s3_config["endpoint"],
            config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 8}),
        )
