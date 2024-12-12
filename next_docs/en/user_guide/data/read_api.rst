
read_api 
==========

Read the content from file or directory to create ``Dataset``, Currently we provided serval functions that cover some scenarios.
if you have new scenarios that is common to most of the users, you can post it on the offical github issues with detail descriptions.
Also it is easy to implement your own read-related funtions.


Important Functions
-------------------


read_jsonl
^^^^^^^^^^^^^^^^

Read the contet from jsonl which may located on local machine or remote s3. if you want to know more about jsonl, please goto :doc:`../../additional_notes/glossary`

.. code:: python

    from magic_pdf.data.read_api import *
    from magic_pdf.data.data_reader_writer import MultiBucketS3DataReader
    from magic_pdf.data.schemas import S3Config

    # read jsonl from local machine
    datasets = read_jsonl("tt.jsonl", None)   # replace with real jsonl file

    # read jsonl from remote s3

    bucket = "bucket_1"                     # replace with real s3 bucket
    ak = "access_key_1"                     # replace with real s3 access key
    sk = "secret_key_1"                     # replace with real s3 secret key
    endpoint_url = "endpoint_url_1"         # replace with real s3 endpoint url

    bucket_2 = "bucket_2"                   # replace with real s3 bucket
    ak_2 = "access_key_2"                   # replace with real s3 access key
    sk_2 = "secret_key_2"                   # replace with real s3 secret key
    endpoint_url_2 = "endpoint_url_2"       # replace with real s3 endpoint url

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

    datasets = read_jsonl(f"s3://bucket_1/tt.jsonl", s3_reader)  # replace with real s3 jsonl file

read_local_pdfs
^^^^^^^^^^^^^^^^^

Read pdf from path or directory.


.. code:: python

    from magic_pdf.data.read_api import *

    # read pdf path
    datasets = read_local_pdfs("tt.pdf")

    # read pdfs under directory
    datasets = read_local_pdfs("pdfs/")


read_local_images
^^^^^^^^^^^^^^^^^^^

Read images from path or directory

.. code:: python 

    from magic_pdf.data.read_api import *

    # read from image path 
    datasets = read_local_images("tt.png")  # replace with real file path

    # read files from directory that endswith suffix in suffixes array 
    datasets = read_local_images("images/", suffixes=[".png", ".jpg"])  # replace with real directory 


read_local_office
^^^^^^^^^^^^^^^^^^^^
Read MS-Office files from path or directory

.. code:: python 

    from magic_pdf.data.read_api import *

    # read from image path 
    datasets = read_local_office("tt.doc")  # replace with real file path

    # read files from directory that endswith suffix in suffixes array 
    datasets = read_local_office("docs/")  # replace with real directory 




Check :doc:`../../api/read_api` for more details