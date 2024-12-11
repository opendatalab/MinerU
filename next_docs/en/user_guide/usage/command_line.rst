

Command Line
===================

.. code:: bash

   magic-pdf --help
   Usage: magic-pdf [OPTIONS]

   Options:
     -v, --version                display the version and exit
     -p, --path PATH              local filepath or directory. support PDF, PPT,
                                  PPTX, DOC, DOCX, PNG, JPG files  [required]
     -o, --output-dir PATH        output local directory  [required]
     -m, --method [ocr|txt|auto]  the method for parsing pdf. ocr: using ocr
                                  technique to extract information from pdf. txt:
                                  suitable for the text-based pdf only and
                                  outperform ocr. auto: automatically choose the
                                  best method for parsing pdf from ocr and txt.
                                  without method specified, auto will be used by
                                  default.
     -l, --lang TEXT              Input the languages in the pdf (if known) to
                                  improve OCR accuracy.  Optional. You should
                                  input "Abbreviation" with language form url: ht
                                  tps://paddlepaddle.github.io/PaddleOCR/en/ppocr
                                  /blog/multi_languages.html#5-support-languages-
                                  and-abbreviations
     -d, --debug BOOLEAN          Enables detailed debugging information during
                                  the execution of the CLI commands.
     -s, --start INTEGER          The starting page for PDF parsing, beginning
                                  from 0.
     -e, --end INTEGER            The ending page for PDF parsing, beginning from
                                  0.
     --help                       Show this message and exit.


   ## show version
   magic-pdf -v

   ## command line example
   magic-pdf -p {some_pdf} -o {some_output_dir} -m auto


.. admonition:: Important
    :class: tip

    The file must endswith with the following suffix.
       .pdf 
       .png
       .jpg
       .ppt
       .pptx
       .doc
       .docx


``{some_pdf}`` can be a single PDF file or a directory containing
multiple PDFs. The results will be saved in the ``{some_output_dir}``
directory. The output file list is as follows:

.. code:: text

   ├── some_pdf.md                          # markdown file
   ├── images                               # directory for storing images
   ├── some_pdf_layout.pdf                  # layout diagram
   ├── some_pdf_middle.json                 # MinerU intermediate processing result
   ├── some_pdf_model.json                  # model inference result
   ├── some_pdf_origin.pdf                  # original PDF file
   ├── some_pdf_spans.pdf                   # smallest granularity bbox position information diagram
   └── some_pdf_content_list.json           # Rich text JSON arranged in reading order

.. admonition:: Tip
   :class: tip
   

   For more information about the output files, please refer to the :doc:`../inference_result` or :doc:`../pipe_result`
