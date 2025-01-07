
Api Usage
===========


PDF
----

Local File Example
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

    ### get model inference result
    model_inference_result = infer_result.get_infer_res()

    ### draw layout result on each page
    pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

    ### draw spans result on each page
    pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

    ### get markdown content
    md_content = pipe_result.get_markdown(image_dir)

    ### dump markdown
    pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

    ### get content list content
    content_list_content = pipe_result.get_content_list(image_dir)

    ### dump content list
    pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)

    ### get middle json
    middle_json_content = pipe_result.get_middle_json()

    ### dump middle json
    pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')



S3 File Example
^^^^^^^^^^^^^^^^

.. code:: python

    import os

    from magic_pdf.data.data_reader_writer import S3DataReader, S3DataWriter
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.config.enums import SupportedPdfParseMethod

    bucket_name = "{Your S3 Bucket Name}"  # replace with real bucket name
    ak = "{Your S3 access key}"  # replace with real s3 access key
    sk = "{Your S3 secret key}"  # replace with real s3 secret key
    endpoint_url = "{Your S3 endpoint_url}"  # replace with real s3 endpoint_url

    reader = S3DataReader('unittest/tmp/', bucket_name, ak, sk, endpoint_url)  # replace `unittest/tmp` with the real s3 prefix
    writer = S3DataWriter('unittest/tmp', bucket_name, ak, sk, endpoint_url)
    image_writer = S3DataWriter('unittest/tmp/images', bucket_name, ak, sk, endpoint_url)
    md_writer = S3DataWriter('unittest/tmp', bucket_name, ak, sk, endpoint_url)

    local_image_dir, local_md_dir = "output/images", "output"
    image_dir = str(os.path.basename(local_image_dir))

    # args
    pdf_file_name = (
        f"s3://{bucket_name}/unittest/tmp/bug5-11.pdf"  # replace with the real s3 path
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
    infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))

    ### get model inference result
    model_inference_result = infer_result.get_infer_res()

    ### draw layout result on each page
    pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

    ### draw spans result on each page
    pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

    ### dump markdown
    pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

    ### dump content list
    pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)

    ### get markdown content
    md_content = pipe_result.get_markdown(image_dir)

    ### get content list content
    content_list_content = pipe_result.get_content_list(image_dir)

    ### get middle json
    middle_json_content = pipe_result.get_middle_json()

    ### dump middle json
    pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')

MS-Office
----------

.. code:: python

    import os

    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.data.read_api import read_local_office

    # prepare env
    local_image_dir, local_md_dir = "output/images", "output"
    image_dir = str(os.path.basename(local_image_dir))

    os.makedirs(local_image_dir, exist_ok=True)

    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )

    # proc
    ## Create Dataset Instance
    input_file = "some_ppt.ppt"     # replace with real ms-office file

    input_file_name = input_file.split(".")[0]
    ds = read_local_office(input_file)[0]

    ds.apply(doc_analyze, ocr=True).pipe_txt_mode(image_writer).dump_md(
        md_writer, f"{input_file_name}.md", image_dir
    )

This code snippet can be used to manipulate **ppt**, **pptx**, **doc**, **docx** file


Image
---------

Single Image File
^^^^^^^^^^^^^^^^^^^

.. code:: python

    import os

    from magic_pdf.data.data_reader_writer import FileBasedDataWriter
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.data.read_api import read_local_images

    # prepare env
    local_image_dir, local_md_dir = "output/images", "output"
    image_dir = str(os.path.basename(local_image_dir))

    os.makedirs(local_image_dir, exist_ok=True)

    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )

    # proc
    ## Create Dataset Instance
    input_file = "some_image.jpg"       # replace with real image file

    input_file_name = input_file.split(".")[0]
    ds = read_local_images(input_file)[0]

    ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer).dump_md(
        md_writer, f"{input_file_name}.md", image_dir
    )


Directory That Contains Images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    import os

    from magic_pdf.data.data_reader_writer import FileBasedDataWriter
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.data.read_api import read_local_images

    # prepare env
    local_image_dir, local_md_dir = "output/images", "output"
    image_dir = str(os.path.basename(local_image_dir))

    os.makedirs(local_image_dir, exist_ok=True)

    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )

    # proc
    ## Create Dataset Instance
    input_directory = "some_image_dir/"       # replace with real directory that contains images


    dss = read_local_images(input_directory, suffixes=['.png', '.jpg'])

    count = 0
    for ds in dss:
        ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer).dump_md(
            md_writer, f"{count}.md", image_dir
        )
        count += 1


Check :doc:`../data/data_reader_writer` for more [reader | writer] examples and check :doc:`../../api/pipe_operators` or :doc:`../../api/model_operators` for api details
