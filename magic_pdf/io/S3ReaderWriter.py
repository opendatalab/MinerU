

from magic_pdf.io.AbsReaderWriter import AbsReaderWriter
from magic_pdf.libs.commons import parse_aws_param, parse_bucket_key
import boto3
from loguru import logger
from boto3.s3.transfer import TransferConfig
from botocore.config import Config


class S3ReaderWriter(AbsReaderWriter):
    def __init__(self, ak: str, sk: str, endpoint_url: str, addressing_style: str):
        self.client = self._get_client(ak, sk, endpoint_url, addressing_style)

    def _get_client(self, ak: str, sk: str, endpoint_url: str, addressing_style: str):
        s3_client = boto3.client(
            service_name="s3",
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            endpoint_url=endpoint_url,
            config=Config(s3={"addressing_style": addressing_style},
                          retries={'max_attempts': 5, 'mode': 'standard'}),
        )
        return s3_client
    def read(self, s3_path, mode="text", encoding="utf-8"):
        bucket_name, bucket_key = parse_bucket_key(s3_path)
        res = self.client.get_object(Bucket=bucket_name, Key=bucket_key)
        body = res["Body"].read()
        if mode == 'text':
            data = body.decode(encoding)  # Decode bytes to text
        elif mode == 'binary':
            data = body
        else:
            raise ValueError("Invalid mode. Use 'text' or 'binary'.")
        return data

    def write(self, data, s3_path, mode="text", encoding="utf-8"):
        if mode == 'text':
            body = data.encode(encoding)  # Encode text data as bytes
        elif mode == 'binary':
            body = data
        else:
            raise ValueError("Invalid mode. Use 'text' or 'binary'.")
        bucket_name, bucket_key = parse_bucket_key(s3_path)
        self.client.put_object(Body=body, Bucket=bucket_name, Key=bucket_key)
        logger.info(f"内容已写入 {s3_path} ")


if __name__ == "__main__":
    # Config the connection info
    ak = ""
    sk = ""
    endpoint_url = ""
    addressing_style = ""

    # Create an S3ReaderWriter object
    s3_reader_writer = S3ReaderWriter(ak, sk, endpoint_url, addressing_style)

    # Write text data to S3
    text_data = "This is some text data"
    s3_reader_writer.write(data=text_data, s3_path = "s3://bucket_name/ebook/test/test.json", mode='text')

    # Read text data from S3
    text_data_read = s3_reader_writer.read(s3_path = "s3://bucket_name/ebook/test/test.json", mode='text')
    logger.info(f"Read text data from S3: {text_data_read}")
    # Write binary data to S3
    binary_data = b"This is some binary data"
    s3_reader_writer.write(data=text_data, s3_path = "s3://bucket_name/ebook/test/test2.json", mode='binary')

    # Read binary data from S3
    binary_data_read = s3_reader_writer.read(s3_path = "s3://bucket_name/ebook/test/test2.json", mode='binary')
    logger.info(f"Read binary data from S3: {binary_data_read}")