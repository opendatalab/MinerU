from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.libs.commons import parse_aws_param, parse_bucket_key, join_path
import boto3
from loguru import logger
from botocore.config import Config


class S3ReaderWriter(AbsReaderWriter):
    def __init__(
        self,
        ak: str,
        sk: str,
        endpoint_url: str,
        addressing_style: str = "auto",
        parent_path: str = "",
    ):
        self.client = self._get_client(ak, sk, endpoint_url, addressing_style)
        self.path = parent_path

    def _get_client(self, ak: str, sk: str, endpoint_url: str, addressing_style: str):
        s3_client = boto3.client(
            service_name="s3",
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            endpoint_url=endpoint_url,
            config=Config(
                s3={"addressing_style": addressing_style},
                retries={"max_attempts": 5, "mode": "standard"},
            ),
        )
        return s3_client

    def read(self, s3_relative_path, mode=AbsReaderWriter.MODE_TXT, encoding="utf-8"):
        if s3_relative_path.startswith("s3://"):
            s3_path = s3_relative_path
        else:
            s3_path = join_path(self.path, s3_relative_path)
        bucket_name, key = parse_bucket_key(s3_path)
        res = self.client.get_object(Bucket=bucket_name, Key=key)
        body = res["Body"].read()
        if mode == AbsReaderWriter.MODE_TXT:
            data = body.decode(encoding)  # Decode bytes to text
        elif mode == AbsReaderWriter.MODE_BIN:
            data = body
        else:
            raise ValueError("Invalid mode. Use 'text' or 'binary'.")
        return data

    def write(self, content, s3_relative_path, mode=AbsReaderWriter.MODE_TXT, encoding="utf-8"):
        if s3_relative_path.startswith("s3://"):
            s3_path = s3_relative_path
        else:
            s3_path = join_path(self.path, s3_relative_path)
        if mode == AbsReaderWriter.MODE_TXT:
            body = content.encode(encoding)  # Encode text data as bytes
        elif mode == AbsReaderWriter.MODE_BIN:
            body = content
        else:
            raise ValueError("Invalid mode. Use 'text' or 'binary'.")
        bucket_name, key = parse_bucket_key(s3_path)
        self.client.put_object(Body=body, Bucket=bucket_name, Key=key)
        logger.info(f"内容已写入 {s3_path} ")

    def read_offset(self, path: str, offset=0, limit=None) -> bytes:
        if path.startswith("s3://"):
            s3_path = path
        else:
            s3_path = join_path(self.path, path)
        bucket_name, key = parse_bucket_key(s3_path)

        range_header = (
            f"bytes={offset}-{offset+limit-1}" if limit else f"bytes={offset}-"
        )
        res = self.client.get_object(Bucket=bucket_name, Key=key, Range=range_header)
        return res["Body"].read()


if __name__ == "__main__":
    if 0:
        # Config the connection info
        ak = ""
        sk = ""
        endpoint_url = ""
        addressing_style = "auto"
        bucket_name = ""
        # Create an S3ReaderWriter object
        s3_reader_writer = S3ReaderWriter(
            ak, sk, endpoint_url, addressing_style, "s3://bucket_name/"
        )

        # Write text data to S3
        text_data = "This is some text data"
        s3_reader_writer.write(
            text_data,
            s3_relative_path=f"s3://{bucket_name}/ebook/test/test.json",
            mode=AbsReaderWriter.MODE_TXT,
        )

        # Read text data from S3
        text_data_read = s3_reader_writer.read(
            s3_relative_path=f"s3://{bucket_name}/ebook/test/test.json", mode=AbsReaderWriter.MODE_TXT
        )
        logger.info(f"Read text data from S3: {text_data_read}")
        # Write binary data to S3
        binary_data = b"This is some binary data"
        s3_reader_writer.write(
            text_data,
            s3_relative_path=f"s3://{bucket_name}/ebook/test/test.json",
            mode=AbsReaderWriter.MODE_BIN,
        )

        # Read binary data from S3
        binary_data_read = s3_reader_writer.read(
            s3_relative_path=f"s3://{bucket_name}/ebook/test/test.json", mode=AbsReaderWriter.MODE_BIN
        )
        logger.info(f"Read binary data from S3: {binary_data_read}")

        # Range Read text data from S3
        binary_data_read = s3_reader_writer.read_offset(
            path=f"s3://{bucket_name}/ebook/test/test.json", offset=0, limit=10
        )
        logger.info(f"Read binary data from S3: {binary_data_read}")
    if 1:
        import os
        import json

        ak = os.getenv("AK", "")
        sk = os.getenv("SK", "")
        endpoint_url = os.getenv("ENDPOINT", "")
        bucket = os.getenv("S3_BUCKET", "")
        prefix = os.getenv("S3_PREFIX", "")
        key_basename = os.getenv("S3_KEY_BASENAME", "")
        s3_reader_writer = S3ReaderWriter(
            ak, sk, endpoint_url, "auto", f"s3://{bucket}/{prefix}"
        )
        content_bin = s3_reader_writer.read_offset(key_basename)
        assert content_bin[:10] == b'{"track_id'
        assert content_bin[-10:] == b'r":null}}\n'

        content_bin = s3_reader_writer.read_offset(key_basename, offset=424, limit=426)
        jso = json.dumps(content_bin.decode("utf-8"))
        print(jso)
