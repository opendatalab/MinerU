

命令行
========

.. code:: bash

   magic-pdf --help
   Usage: magic-pdf [OPTIONS]

   Options:
     -v, --version                display the version and exit
     -p, --path PATH              local pdf filepath or directory  [required]
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

``{some_pdf}`` 可以是单个 PDF 文件或者一个包含多个 PDF 文件的目录。 解析的结果文件存放在目录 ``{some_output_dir}`` 下。 生成的结果文件列表如下所示：

.. code:: text

   ├── some_pdf.md                          # markdown 文件
   ├── images                               # 存放图片目录
   ├── some_pdf_layout.pdf                  # layout 绘图 （包含layout阅读顺序）
   ├── some_pdf_middle.json                 # minerU 中间处理结果
   ├── some_pdf_model.json                  # 模型推理结果
   ├── some_pdf_origin.pdf                  # 原 pdf 文件
   ├── some_pdf_spans.pdf                   # 最小粒度的bbox位置信息绘图
   └── some_pdf_content_list.json           # 按阅读顺序排列的富文本json


.. admonition:: Tip
   :class: tip

   欲知更多有关结果文件的信息，请参考 :doc:`../tutorial/output_file_description`

