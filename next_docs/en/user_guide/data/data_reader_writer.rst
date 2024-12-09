
Data Reader Writer 
====================

Aims for read or write bytes from different media, You can implement new classes to meet the needs of your personal scenarios 
if MinerU have not provide the suitable classes. It is easy to implement new classes, the only one requirement is to inherit from
``DataReader`` or ``DataWriter``

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


Reader may curious about the difference between :doc:`io` and this section. Those two sections look very similarity at first glance.
:doc:`io` provides fundamental functions, while This section thinks more at application level. Customer can build they own classes to meet 
their own applications need which may share same IO function. That is why we have :doc:`io`.


Important Classes
-----------------

.. code:: python

    class FileBasedDataReader(DataReader):
        def __init__(self, parent_dir: str = ''):
            pass


    class FileBasedDataWriter(DataWriter):
        def __init__(self, parent_dir: str = '') -> None:
            pass

Class ``FileBasedDataReader`` initialized with unary param ``parent_dir``, That means that every method ``FileBasedDataReader`` provided will have features as follow.

Features:
    #. read content from the absolute path file, ``parent_dir`` will be ignored.
    #. read the relative path, file will first join with ``parent_dir``, then read content from the merged path


.. note::

    ``FileBasedDataWriter`` shares the same behavior with ``FileBaseDataReader``


.. code:: python 

    class MultiS3Mixin:
        def __init__(self, default_prefix: str, s3_configs: list[S3Config]):
            pass

    class MultiBucketS3DataReader(DataReader, MultiS3Mixin):
        pass

All read-related method that class ``MultiBucketS3DataReader`` provided will have features as follow.

Features:
    #. read object with full s3-format path, for example ``s3://test_bucket/test_object``, ``default_prefix`` will be ignored.
    #. read object with relative path, file will join ``default_prefix`` and trim the ``bucket_name`` firstly, then read the content. ``bucket_name`` is the first element of the result after split ``default_prefix`` with delimiter ``\`` 

.. note::
    ``MultiBucketS3DataWriter`` shares the same behavior with ``MultiBucketS3DataReader``


.. code:: python

    class S3DataReader(MultiBucketS3DataReader):
        pass

``S3DataReader`` is build on top of MultiBucketS3DataReader which only support for bucket. So is ``S3DataWriter``. 


Read Examples
------------

.. code:: python

    import os 
    from magic_pdf.data.data_reader_writer import *
    from magic_pdf.data.data_reader_writer import MultiBucketS3DataReader
    from magic_pdf.data.schemas import S3Config

    # file based related
    file_based_reader1 = FileBasedDataReader('')

    ## will read file abc
    file_based_reader1.read('abc')

    file_based_reader2 = FileBasedDataReader('/tmp')

    ## will read /tmp/abc
    file_based_reader2.read('abc')

    ## will read /tmp/logs/message.txt
    file_based_reader2.read('/tmp/logs/message.txt')

    # multi bucket s3 releated
    bucket = "bucket"               # replace with real bucket
    ak = "ak"                       # replace with real access key
    sk = "sk"                       # replace with real secret key
    endpoint_url = "endpoint_url"   # replace with real endpoint_url

    bucket_2 = "bucket_2"               # replace with real bucket
    ak_2 = "ak_2"                       # replace with real access key
    sk_2 = "sk_2"                       # replace with real secret key 
    endpoint_url_2 = "endpoint_url_2"   # replace with real endpoint_url

    test_prefix = 'test/unittest'
    multi_bucket_s3_reader1 = MultiBucketS3DataReader(f"{bucket}/{test_prefix}", [S3Config(
            bucket_name=bucket, access_key=ak, secret_key=sk, endpoint_url=endpoint_url
        ),
        S3Config(
            bucket_name=bucket_2,
            access_key=ak_2,
            secret_key=sk_2,
            endpoint_url=endpoint_url_2,
        )])

    ## will read s3://{bucket}/{test_prefix}/abc
    multi_bucket_s3_reader1.read('abc')

    ## will read s3://{bucket}/{test_prefix}/efg
    multi_bucket_s3_reader1.read(f's3://{bucket}/{test_prefix}/efg')

    ## will read s3://{bucket2}/{test_prefix}/abc
    multi_bucket_s3_reader1.read(f's3://{bucket_2}/{test_prefix}/abc')

    # s3 related
    s3_reader1 = S3DataReader(
        test_prefix,
        bucket,
        ak,
        sk,
        endpoint_url
    )

    ## will read s3://{bucket}/{test_prefix}/abc
    s3_reader1.read('abc')

    ## will read s3://{bucket}/efg
    s3_reader1.read(f's3://{bucket}/efg')


Write Examples
---------------

.. code:: python

    import os
    from magic_pdf.data.data_reader_writer import *
    from magic_pdf.data.data_reader_writer import MultiBucketS3DataWriter
    from magic_pdf.data.schemas import S3Config

    # file based related
    file_based_writer1 = FileBasedDataWriter("")

    ## will write 123 to abc
    file_based_writer1.write("abc", "123".encode())

    ## will write 123 to abc
    file_based_writer1.write_string("abc", "123")

    file_based_writer2 = FileBasedDataWriter("/tmp")

    ## will write 123 to /tmp/abc
    file_based_writer2.write_string("abc", "123")

    ## will write 123 to /tmp/logs/message.txt
    file_based_writer2.write_string("/tmp/logs/message.txt", "123")

    # multi bucket s3 releated
    bucket = "bucket"               # replace with real bucket
    ak = "ak"                       # replace with real access key
    sk = "sk"                       # replace with real secret key
    endpoint_url = "endpoint_url"   # replace with real endpoint_url

    bucket_2 = "bucket_2"               # replace with real bucket
    ak_2 = "ak_2"                       # replace with real access key
    sk_2 = "sk_2"                       # replace with real secret key 
    endpoint_url_2 = "endpoint_url_2"   # replace with real endpoint_url

    test_prefix = "test/unittest"
    multi_bucket_s3_writer1 = MultiBucketS3DataWriter(
        f"{bucket}/{test_prefix}",
        [
            S3Config(
                bucket_name=bucket, access_key=ak, secret_key=sk, endpoint_url=endpoint_url
            ),
            S3Config(
                bucket_name=bucket_2,
                access_key=ak_2,
                secret_key=sk_2,
                endpoint_url=endpoint_url_2,
            ),
        ],
    )

    ## will write 123 to s3://{bucket}/{test_prefix}/abc
    multi_bucket_s3_writer1.write_string("abc", "123")

    ## will write 123 to s3://{bucket}/{test_prefix}/abc
    multi_bucket_s3_writer1.write("abc", "123".encode())

    ## will write 123 to s3://{bucket}/{test_prefix}/efg
    multi_bucket_s3_writer1.write(f"s3://{bucket}/{test_prefix}/efg", "123".encode())

    ## will write 123 to s3://{bucket_2}/{test_prefix}/abc
    multi_bucket_s3_writer1.write(f's3://{bucket_2}/{test_prefix}/abc', '123'.encode())

    # s3 related
    s3_writer1 = S3DataWriter(test_prefix, bucket, ak, sk, endpoint_url)

    ## will write 123 to s3://{bucket}/{test_prefix}/abc
    s3_writer1.write("abc", "123".encode())

    ## will write 123 to s3://{bucket}/{test_prefix}/abc
    s3_writer1.write_string("abc", "123")

    ## will write 123 to s3://{bucket}/efg
    s3_writer1.write(f"s3://{bucket}/efg", "123".encode())



Check :doc:`../../api/data_reader_writer` for more details
