
数据读取和写入类 
=================

旨在从不同的媒介读取或写入字节。如果 MinerU 没有提供合适的类，你可以实现新的类以满足个人场景的需求。实现新的类非常容易，唯一的要求是继承自 DataReader 或 DataWriter。

.. code:: python

    class SomeReader(DataReader):
        def read(self, path: str) -> bytes:
            pass

        def read_at(self, path: str, offset: int = 0, limit: int = -1) -> bytes:
            pass


    class SomeWriter(DataWriter):
        def write(self, path: str, data: bytes) -> None:
            pass

        def write_string(self, path: str, data: str) -> None:
            pass

读者可能会对 io 和本节的区别感到好奇。乍一看，这两部分非常相似。io 提供基本功能，而本节则更注重应用层面。用户可以构建自己的类以满足特定应用需求，这些类可能共享相同的基本 IO 功能。这就是为什么我们有 io。

重要类
------------
.. code:: python

    class FileBasedDataReader(DataReader):
        def __init__(self, parent_dir: str = ''):
            pass


    class FileBasedDataWriter(DataWriter):
        def __init__(self, parent_dir: str = '') -> None:
            pass

类 FileBasedDataReader 使用单个参数 parent_dir 初始化。这意味着 FileBasedDataReader 提供的每个方法将具有以下特性：

#. 从绝对路径文件读取内容，parent_dir 将被忽略。
#. 从相对路径读取文件，首先将路径与 parent_dir 连接，然后从合并后的路径读取内容。

.. note::

    `FileBasedDataWriter` 与 `FileBasedDataReader` 具有相同的行为。

.. code:: python

    class MultiS3Mixin:
        def __init__(self, default_prefix: str, s3_configs: list[S3Config]):
            pass

    class MultiBucketS3DataReader(DataReader, MultiS3Mixin):
        pass

MultiBucketS3DataReader 提供的所有读取相关方法将具有以下特性：

#. 从完整的 S3 格式路径读取对象，例如 s3://test_bucket/test_object，default_prefix 将被忽略。
#. 从相对路径读取对象，首先将路径与 default_prefix 连接并去掉 bucket_name，然后读取内容。bucket_name 是将 default_prefix 用分隔符 \ 分割后的第一个元素。

.. note::
    MultiBucketS3DataWriter 与 MultiBucketS3DataReader 具有类似的行为。

.. code:: python

    class S3DataReader(MultiBucketS3DataReader):
        pass

S3DataReader 基于 MultiBucketS3DataReader 构建，但仅支持单个桶。S3DataWriter 也是类似的情况。

读取示例
---------
.. code:: python

    from magic_pdf.data.data_reader_writer import * 

    # 文件相关的
    file_based_reader1 = FileBasedDataReader('')

    ## 将读取文件 abc 
    file_based_reader1.read('abc') 

    file_based_reader2 = FileBasedDataReader('/tmp')

    ## 将读取 /tmp/abc
    file_based_reader2.read('abc')

    ## 将读取 /var/logs/message.txt
    file_based_reader2.read('/var/logs/message.txt')

    # 多桶 S3 相关的
    multi_bucket_s3_reader1 = MultiBucketS3DataReader("test_bucket1/test_prefix", list[S3Config(
            bucket_name=test_bucket1, access_key=ak, secret_key=sk, endpoint_url=endpoint_url
        ),
        S3Config(
            bucket_name=test_bucket_2,
            access_key=ak_2,
            secret_key=sk_2,
            endpoint_url=endpoint_url_2,
        )])

    ## 将读取 s3://test_bucket1/test_prefix/abc
    multi_bucket_s3_reader1.read('abc')

    ## 将读取 s3://test_bucket1/efg
    multi_bucket_s3_reader1.read('s3://test_bucket1/efg')

    ## 将读取 s3://test_bucket2/abc
    multi_bucket_s3_reader1.read('s3://test_bucket2/abc')

    # S3 相关的
    s3_reader1 = S3DataReader(
        default_prefix_without_bucket = "test_prefix",
        bucket: "test_bucket",
        ak: "ak",
        sk: "sk",
        endpoint_url: "localhost"
    )

    ## 将读取 s3://test_bucket/test_prefix/abc 
    s3_reader1.read('abc')

    ## 将读取 s3://test_bucket/efg
    s3_reader1.read('s3://test_bucket/efg')

写入示例
----------
.. code:: python

    from magic_pdf.data.data_reader_writer import *

    # 文件相关的
    file_based_writer1 = FileBasedDataWriter('')

    ## 将写入 123 到 abc
    file_based_writer1.write('abc', '123'.encode()) 

    ## 将写入 123 到 abc
    file_based_writer1.write_string('abc', '123') 

    file_based_writer2 = FileBasedDataWriter('/tmp')

    ## 将写入 123 到 /tmp/abc
    file_based_writer2.write_string('abc', '123')

    ## 将写入 123 到 /var/logs/message.txt
    file_based_writer2.write_string('/var/logs/message.txt', '123')

    # 多桶 S3 相关的
    multi_bucket_s3_writer1 = MultiBucketS3DataWriter("test_bucket1/test_prefix", list[S3Config(
            bucket_name=test_bucket1, access_key=ak, secret_key=sk, endpoint_url=endpoint_url
        ),
        S3Config(
            bucket_name=test_bucket_2,
            access_key=ak_2,
            secret_key=sk_2,
            endpoint_url=endpoint_url_2,
        )])

    ## 将写入 123 到 s3://test_bucket1/test_prefix/abc
    multi_bucket_s3_writer1.write_string('abc', '123')

    ## 将写入 123 到 s3://test_bucket1/test_prefix/abc
    multi_bucket_s3_writer1.write('abc', '123'.encode())

    ## 将写入 123 到 s3://test_bucket1/efg
    multi_bucket_s3_writer1.write('s3://test_bucket1/efg', '123'.encode())

    ## 将写入 123 到 s3://test_bucket2/abc
    multi_bucket_s3_writer1.write('s3://test_bucket2/abc', '123'.encode())

    # S3 相关的
    s3_writer1 = S3DataWriter(
        default_prefix_without_bucket = "test_prefix",
        bucket: "test_bucket",
        ak: "ak",
        sk: "sk",
        endpoint_url: "localhost"
    )

    ## 将写入 123 到 s3://test_bucket/test_prefix/abc 
    s3_writer1.write('abc', '123'.encode())

    ## 将写入 123 到 s3://test_bucket/test_prefix/abc 
    s3_writer1.write_string('abc', '123')

    ## 将写入 123 到 s3://test_bucket/efg
    s3_writer1.write('s3://test_bucket/efg', '123'.encode())

