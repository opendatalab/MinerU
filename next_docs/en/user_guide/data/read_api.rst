
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

    # read jsonl from local machine 
    datasets = read_jsonl("tt.jsonl", None)

    # read jsonl from remote s3
    datasets = read_jsonl("s3://bucket_1/tt.jsonl", s3_reader)


read_local_pdfs
^^^^^^^^^^^^^^^^

Read pdf from path or directory.


.. code:: python

    # read pdf path
    datasets = read_local_pdfs("tt.pdf")

    # read pdfs under directory
    datasets = read_local_pdfs("pdfs/")


read_local_images
^^^^^^^^^^^^^^^^^^^

Read images from path or directory

.. code:: python 

    # read from image path 
    datasets = read_local_images("tt.png")


    # read files from directory that endswith suffix in suffixes array 
    datasets = read_local_images("images/", suffixes=["png", "jpg"])


Check :doc:`../../api/read_api` for more details