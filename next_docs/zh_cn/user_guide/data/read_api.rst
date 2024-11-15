

read_api
=========

从文件或目录读取内容以创建 Dataset。目前，我们提供了几个覆盖某些场景的函数。如果你有新的、大多数用户都会遇到的场景，可以在官方 GitHub 问题页面上发布详细描述。同时，实现你自己的读取相关函数也非常容易。

重要函数
---------

read_jsonl
^^^^^^^^^^^^^^^^

从本地机器或远程 S3 上的 JSONL 文件读取内容。如果你想了解更多关于 JSONL 的信息，请参阅 :doc:`../../additional_notes/glossary`。

.. code:: python

    from magic_pdf.data.io.read_api import *

    # 从本地机器读取 JSONL
    datasets = read_jsonl("tt.jsonl", None)

    # 从远程 S3 读取 JSONL
    datasets = read_jsonl("s3://bucket_1/tt.jsonl", s3_reader)

read_local_pdfs
^^^^^^^^^^^^^^^^

从路径或目录读取 PDF 文件。

.. code:: python

    from magic_pdf.data.io.read_api import *

    # 读取 PDF 路径
    datasets = read_local_pdfs("tt.pdf")

    # 读取目录下的 PDF 文件
    datasets = read_local_pdfs("pdfs/")

read_local_images
^^^^^^^^^^^^^^^^^^^

从路径或目录读取图像。

.. code:: python

    from magic_pdf.data.io.read_api import *

    # 从图像路径读取
    datasets = read_local_images("tt.png")

    # 从目录读取以 suffixes 数组中指定后缀结尾的文件
    datasets = read_local_images("images/", suffixes=["png", "jpg"])
