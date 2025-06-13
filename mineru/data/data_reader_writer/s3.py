from .multi_bucket_s3 import MultiBucketS3DataReader, MultiBucketS3DataWriter
from ..utils.schemas import S3Config


class S3DataReader(MultiBucketS3DataReader):
    def __init__(
        self,
        default_prefix_without_bucket: str,
        bucket: str,
        ak: str,
        sk: str,
        endpoint_url: str,
        addressing_style: str = 'auto',
    ):
        """s3 reader client.

        Args:
            default_prefix_without_bucket: prefix that not contains bucket
            bucket (str): bucket name
            ak (str): access key
            sk (str): secret key
            endpoint_url (str): endpoint url of s3
            addressing_style (str, optional): Defaults to 'auto'. Other valid options here are 'path' and 'virtual'
            refer to https://boto3.amazonaws.com/v1/documentation/api/1.9.42/guide/s3.html
        """
        super().__init__(
            f'{bucket}/{default_prefix_without_bucket}',
            [
                S3Config(
                    bucket_name=bucket,
                    access_key=ak,
                    secret_key=sk,
                    endpoint_url=endpoint_url,
                    addressing_style=addressing_style,
                )
            ],
        )


class S3DataWriter(MultiBucketS3DataWriter):
    def __init__(
        self,
        default_prefix_without_bucket: str,
        bucket: str,
        ak: str,
        sk: str,
        endpoint_url: str,
        addressing_style: str = 'auto',
    ):
        """s3 writer client.

        Args:
            default_prefix_without_bucket: prefix that not contains bucket
            bucket (str): bucket name
            ak (str): access key
            sk (str): secret key
            endpoint_url (str): endpoint url of s3
            addressing_style (str, optional): Defaults to 'auto'. Other valid options here are 'path' and 'virtual'
            refer to https://boto3.amazonaws.com/v1/documentation/api/1.9.42/guide/s3.html
        """
        super().__init__(
            f'{bucket}/{default_prefix_without_bucket}',
            [
                S3Config(
                    bucket_name=bucket,
                    access_key=ak,
                    secret_key=sk,
                    endpoint_url=endpoint_url,
                    addressing_style=addressing_style,
                )
            ],
        )
