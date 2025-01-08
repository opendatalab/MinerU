

Convert Doc
=============

.. admonition:: Warning
    :class: tip

    When processing MS-Office files, we first use third-party software to convert the MS-Office files to PDF.

    For certain MS-Office files, the quality of the converted PDF files may not be very high, which can affect the quality of the final output.



Command Line
^^^^^^^^^^^^^

.. code:: python

    # replace with real ms-office file, we support MS-DOC, MS-DOCX, MS-PPT, MS-PPTX now
    magic-pdf -p a.doc -o output -m auto


API
^^^^^^^^
.. code:: python

    import os

    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.data.read_api import read_local_office
    from magic_pdf.config.enums import SupportedPdfParseMethod


    # prepare env
    local_image_dir, local_md_dir = "output/images", "output"
    image_dir = str(os.path.basename(local_image_dir))

    os.makedirs(local_image_dir, exist_ok=True)

    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )

    # proc
    ## Create Dataset Instance
    input_file = "some_doc.doc"     # replace with real ms-office file, we support MS-DOC, MS-DOCX, MS-PPT, MS-PPTX now

    input_file_name = input_file.split(".")[0]
    ds = read_local_office(input_file)[0]


    ## inference
    if ds.classify() == SupportedPdfParseMethod.OCR:
        ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer).dump_md(
        md_writer, f"{input_file_name}.md", image_dir)
    else:
        ds.apply(doc_analyze, ocr=False).pipe_txt_mode(image_writer).dump_md(
        md_writer, f"{input_file_name}.md", image_dir)
