

Convert To Markdown
========================

.. code:: python

    import os

    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.libs.MakeContentConfig import DropMode, MakeMode
    from magic_pdf.pipe.OCRPipe import OCRPipe


    ## args
    model_list = []
    pdf_file_name = "abc.pdf"  # replace with the real pdf path


    ## prepare env
    local_image_dir, local_md_dir = "output/images", "output"
    os.makedirs(local_image_dir, exist_ok=True)

    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    ) # create 00
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


Check :doc:`../data/data_reader_writer` for more [reader | writer] examples 
