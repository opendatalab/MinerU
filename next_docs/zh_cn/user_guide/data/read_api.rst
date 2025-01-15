

read_api
=========

从文件或目录读取内容以创建 Dataset。目前，我们提供了几个覆盖某些场景的函数。如果你有新的、大多数用户都会遇到的场景，可以在官方 GitHub 问题页面上发布详细描述。同时，实现你自己的读取相关函数也非常容易。

重要函数
---------

read_jsonl
^^^^^^^^^^^^^^^^

从本地机器或远程 S3 上的 JSONL 文件读取内容。如果你想了解更多关于 JSONL 的信息，请参阅 :doc:`../../additional_notes/glossary`。

.. code:: python

    from magic_pdf.data.read_api import *
    from magic_pdf.data.data_reader_writer import MultiBucketS3DataReader
    from magic_pdf.data.schemas import S3Config

    # 读取本地 jsonl 文件
    datasets = read_jsonl("tt.jsonl", None)   # 替换为有效的文件

    # 读取 s3 jsonl 文件

    bucket = "bucket_1"                     # 替换为有效的 s3 bucket
    ak = "access_key_1"                     # 替换为有效的 s3 access key
    sk = "secret_key_1"                     # 替换为有效的 s3 secret key
    endpoint_url = "endpoint_url_1"         # 替换为有效的 s3 endpoint url

    bucket_2 = "bucket_2"                   # 替换为有效的 s3 bucket
    ak_2 = "access_key_2"                   # 替换为有效的 s3 access key
    sk_2 = "secret_key_2"                   # 替换为有效的 s3 secret key
    endpoint_url_2 = "endpoint_url_2"       # 替换为有效的 s3 endpoint url

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

    s3_reader = MultiBucketS3DataReader(bucket, s3configs)

    datasets = read_jsonl(f"s3://bucket_1/tt.jsonl", s3_reader)  # 替换为有效的 s3 jsonl file


read_local_pdfs
^^^^^^^^^^^^^^^^

从路径或目录读取 PDF 文件。

.. code:: python

    from magic_pdf.data.read_api import *

    # 读取 PDF 路径
    datasets = read_local_pdfs("tt.pdf")  # 替换为有效的文件

    # 读取目录下的 PDF 文件
    datasets = read_local_pdfs("pdfs/")   # 替换为有效的文件目录

read_local_images
^^^^^^^^^^^^^^^^^^^

从路径或目录读取图像。

.. code:: python

    from magic_pdf.data.read_api import *

    # 从图像路径读取
    datasets = read_local_images("tt.png")  # 替换为有效的文件

    # 从目录读取以 suffixes 数组中指定后缀结尾的文件
    datasets = read_local_images("images/", suffixes=["png", "jpg"])  # 替换为有效的文件目录
