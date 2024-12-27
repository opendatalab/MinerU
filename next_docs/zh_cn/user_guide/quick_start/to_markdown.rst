
转换为 Markdown 文件
========================

本地文件示例
^^^^^^^^^^^^^^^^^^

.. code:: python

    import os

    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.config.enums import SupportedPdfParseMethod

    # args
    pdf_file_name = "abc.pdf"  # replace with the real pdf path
    name_without_suff = pdf_file_name.split(".")[0]

    # prepare env
    local_image_dir, local_md_dir = "output/images", "output"
    image_dir = str(os.path.basename(local_image_dir))

    os.makedirs(local_image_dir, exist_ok=True)

    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )
    image_dir = str(os.path.basename(local_image_dir))

    # read bytes
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content

    # proc
    ## Create Dataset Instance
    ds = PymuDocDataset(pdf_bytes)

    ## inference
    if ds.classify() == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True)

        ## pipeline
        pipe_result = infer_result.pipe_ocr_mode(image_writer)

    else:
        infer_result = ds.apply(doc_analyze, ocr=False)

        ## pipeline
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    ### draw model result on each page
    infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))

    ### draw layout result on each page
    pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

    ### draw spans result on each page
    pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

    ### dump markdown
    pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

    ### dump content list
    pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)


对象存储文件示例
^^^^^^^^^^^^^^^^

.. code:: python

    import os

    from magic_pdf.data.data_reader_writer import S3DataReader, S3DataWriter
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

    bucket_name = "{Your S3 Bucket Name}"  # replace with real bucket name
    ak = "{Your S3 access key}"  # replace with real s3 access key
    sk = "{Your S3 secret key}"  # replace with real s3 secret key
    endpoint_url = "{Your S3 endpoint_url}"  # replace with real s3 endpoint_url


    reader = S3DataReader('unittest/tmp/', bucket_name, ak, sk, endpoint_url)  # replace `unittest/tmp` with the real s3 prefix
    writer = S3DataWriter('unittest/tmp', bucket_name, ak, sk, endpoint_url)
    image_writer = S3DataWriter('unittest/tmp/images', bucket_name, ak, sk, endpoint_url)

    # args
    pdf_file_name = (
        "s3://llm-pdf-text-1/unittest/tmp/bug5-11.pdf"  # replace with the real s3 path
    )

    # prepare env
    local_dir = "output"
    name_without_suff = os.path.basename(pdf_file_name).split(".")[0]

    # read bytes
    pdf_bytes = reader.read(pdf_file_name)  # read the pdf content

    # proc
    ## Create Dataset Instance
    ds = PymuDocDataset(pdf_bytes)

    ## inference
    if ds.classify() == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True)

        ## pipeline
        pipe_result = infer_result.pipe_ocr_mode(image_writer)

    else:
        infer_result = ds.apply(doc_analyze, ocr=False)

        ## pipeline
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    ### draw model result on each page
    infer_result.draw_model(os.path.join(local_dir, f'{name_without_suff}_model.pdf'))  # dump to local

    ### draw layout result on each page
    pipe_result.draw_layout(os.path.join(local_dir, f'{name_without_suff}_layout.pdf'))  # dump to local

    ### draw spans result on each page
    pipe_result.draw_span(os.path.join(local_dir, f'{name_without_suff}_spans.pdf'))  # dump to local

    ### dump markdown
    pipe_result.dump_md(writer, f'{name_without_suff}.md', "unittest/tmp/images")  # dump to remote s3

    ### dump content list
    pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)

前去 :doc:`../data/data_reader_writer` 获取更多有关 **读写** 示例
