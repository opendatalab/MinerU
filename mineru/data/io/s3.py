import boto3
from botocore.config import Config

from ..io.base import IOReader, IOWriter


class S3Reader(IOReader):
    def __init__(
        self,
        bucket: str,
        ak: str,
        sk: str,
        endpoint_url: str,
        addressing_style: str = 'auto',
    ):
        """s3 reader client.

        Args:
            bucket (str): bucket name
            ak (str): access key
            sk (str): secret key
            endpoint_url (str): endpoint url of s3
            addressing_style (str, optional): Defaults to 'auto'. Other valid options here are 'path' and 'virtual'
            refer to https://boto3.amazonaws.com/v1/documentation/api/1.9.42/guide/s3.html
        """
        self._bucket = bucket
        self._ak = ak
        self._sk = sk
        self._s3_client = boto3.client(
            service_name='s3',
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            endpoint_url=endpoint_url,
            config=Config(
                s3={'addressing_style': addressing_style},
                retries={'max_attempts': 5, 'mode': 'standard'},
            ),
        )

    def read(self, key: str) -> bytes:
        """Read the file.

        Args:
            path (str): file path to read

        Returns:
            bytes: the content of the file
        """
        return self.read_at(key)

    def read_at(self, key: str, offset: int = 0, limit: int = -1) -> bytes:
        """Read at offset and limit.

        Args:
            path (str): the path of file, if the path is relative path, it will be joined with parent_dir.
            offset (int, optional): the number of bytes skipped. Defaults to 0.
            limit (int, optional): the length of bytes want to read. Defaults to -1.

        Returns:
            bytes: the content of file
        """
        if limit > -1:
            range_header = f'bytes={offset}-{offset+limit-1}'
            res = self._s3_client.get_object(
                Bucket=self._bucket, Key=key, Range=range_header
            )
        else:
            res = self._s3_client.get_object(
                Bucket=self._bucket, Key=key, Range=f'bytes={offset}-'
            )
        return res['Body'].read()


class S3Writer(IOWriter):
    def __init__(
        self,
        bucket: str,
        ak: str,
        sk: str,
        endpoint_url: str,
        addressing_style: str = 'auto',
    ):
        """s3 reader client.

        Args:
            bucket (str): bucket name
            ak (str): access key
            sk (str): secret key
            endpoint_url (str): endpoint url of s3
            addressing_style (str, optional): Defaults to 'auto'. Other valid options here are 'path' and 'virtual'
            refer to https://boto3.amazonaws.com/v1/documentation/api/1.9.42/guide/s3.html
        """
        self._bucket = bucket
        self._ak = ak
        self._sk = sk
        self._s3_client = boto3.client(
            service_name='s3',
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            endpoint_url=endpoint_url,
            config=Config(
                s3={'addressing_style': addressing_style},
                retries={'max_attempts': 5, 'mode': 'standard'},
            ),
        )

    def write(self, key: str, data: bytes):
        """Write file with data.

        Args:
            path (str): the path of file, if the path is relative path, it will be joined with parent_dir.
            data (bytes): the data want to write
        """
        self._s3_client.put_object(Bucket=self._bucket, Key=key, Body=data)
