# Copyright (c) Opendatalab. All rights reserved.

from pydantic import BaseModel, Field


class S3Config(BaseModel):
    """S3 config
    """
    bucket_name: str = Field(description='s3 bucket name', min_length=1)
    access_key: str = Field(description='s3 access key', min_length=1)
    secret_key: str = Field(description='s3 secret key', min_length=1)
    endpoint_url: str = Field(description='s3 endpoint url', min_length=1)
    addressing_style: str = Field(description='s3 addressing style', default='auto', min_length=1)


class PageInfo(BaseModel):
    """The width and height of page
    """
    w: float = Field(description='the width of page')
    h: float = Field(description='the height of page')
