

Convert Image
===============


Command Line
^^^^^^^^^^^^^

.. code:: python

    # make sure the file have correct suffix
    magic-pdf -p a.png -o output -m auto


API
^^^^^^

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

    # ocr mode
    ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer).dump_md(
        md_writer, f"{input_file_name}.md", image_dir
    )
