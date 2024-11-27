

Convert To Markdown
========================


Local File Example
^^^^^^^^^^^^^^^^^^

.. code:: python

    import os

    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.config.make_content_config import DropMode, MakeMode
    from magic_pdf.pipe.OCRPipe import OCRPipe


    ## args
    model_list = []
    pdf_file_name = "abc.pdf"  # replace with the real pdf path


    ## prepare env
    local_image_dir, local_md_dir = "output/images", "output"
    os.makedirs(local_image_dir, exist_ok=True)

    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )
    image_dir = str(os.path.basename(local_image_dir))

    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_file_name)   # read the pdf content


    pipe = OCRPipe(pdf_bytes, model_list, image_writer)

    pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()

    pdf_info = pipe.pdf_mid_data["pdf_info"]


    md_content = pipe.pipe_mk_markdown(
        image_dir, drop_mode=DropMode.NONE, md_make_mode=MakeMode.MM_MD
    )

    if isinstance(md_content, list):
        md_writer.write_string(f"{pdf_file_name}.md", "\n".join(md_content))
    else:
        md_writer.write_string(f"{pdf_file_name}.md", md_content)


S3 File Example
^^^^^^^^^^^^^^^^

.. code:: python

    import os

    from magic_pdf.data.data_reader_writer import S3DataReader, S3DataWriter
    from magic_pdf.config.make_content_config import DropMode, MakeMode
    from magic_pdf.pipe.OCRPipe import OCRPipe

    bucket_name = "{Your S3 Bucket Name}"  # replace with real bucket name
    ak = "{Your S3 access key}"  # replace with real s3 access key
    sk = "{Your S3 secret key}"  # replace with real s3 secret key
    endpoint_url = "{Your S3 endpoint_url}"  # replace with real s3 endpoint_url


    reader = S3DataReader('unittest/tmp/', bucket_name, ak, sk, endpoint_url)  # replace `unittest/tmp` with the real s3 prefix
    writer = S3DataWriter('unittest/tmp', bucket_name, ak, sk, endpoint_url)
    image_writer = S3DataWriter('unittest/tmp/images', bucket_name, ak, sk, endpoint_url)

    ## args
    model_list = []
    pdf_file_name = f"s3://{bucket_name}/{fake pdf path}"  # replace with the real s3 path

    pdf_bytes = reader.read(pdf_file_name)  # read the pdf content


    pipe = OCRPipe(pdf_bytes, model_list, image_writer)

    pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()

    pdf_info = pipe.pdf_mid_data["pdf_info"]

    md_content = pipe.pipe_mk_markdown(
        "unittest/tmp/images", drop_mode=DropMode.NONE, md_make_mode=MakeMode.MM_MD
    )

    if isinstance(md_content, list):
        writer.write_string(f"{pdf_file_name}.md", "\n".join(md_content))
    else:
        writer.write_string(f"{pdf_file_name}.md", md_content)


Check :doc:`../data/data_reader_writer` for more [reader | writer] examples
