<div id="top"></div>
<div align="center">

[![stars](https://img.shields.io/github/stars/magicpdf/Magic-PDF.svg)](https://github.com/magicpdf/Magic-PDF)
[![forks](https://img.shields.io/github/forks/magicpdf/Magic-PDF.svg)](https://github.com/magicpdf/Magic-PDF)
[![license](https://img.shields.io/github/license/magicpdf/Magic-PDF.svg)](https://github.com/magicpdf/Magic-PDF/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/magicpdf/Magic-PDF)](https://github.com/magicpdf/Magic-PDF/issues)
[![open issues](https://img.shields.io/github/issues-raw/magicpdf/Magic-PDF)](https://github.com/magicpdf/Magic-PDF/issues)

[English](README.md) | [简体中文](README_zh-CN.md)

</div>

<div align="center">

</div>

# Magic-PDF

## Introduction

Magic-PDF is a tool designed to convert PDF documents into Markdown format, capable of processing files stored locally or on object storage supporting S3 protocol.

Key features include:

- Support for multiple front-end model inputs
- Removal of headers, footers, footnotes, and page numbers
- Human-readable layout formatting
- Retains the original document's structure and formatting, including headings, paragraphs, lists, and more
- Extraction and display of images and tables within markdown
- Conversion of equations into LaTeX format
- Automatic detection and conversion of garbled PDFs
- Compatibility with CPU and GPU environments
- Available for Windows, Linux, and macOS platforms

## Project Panorama

![Project Panorama](docs/images/project_panorama_en.png)

## Getting Started

### Requirements

- Python 3.9 or newer

### Usage Instructions

#### 1. Install Magic-PDF
```bash
pip install magic-pdf
```

#### 2. Usage via Command Line

###### simple
```bash
cp magic-pdf.template.json to ~/magic-pdf.json
magic-pdf pdf-command --pdf "pdf_path" --model "model_json_path"
```
###### more 
```bash
magic-pdf --help
```

#### 3. Usage via Api

###### Local
```python
image_writer = DiskReaderWriter(local_image_dir)
image_dir = str(os.path.basename(local_image_dir))
jso_useful_key = {"_pdf_type": "", "model_list": model_json}
pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
pipe.pipe_classify()
pipe.pipe_parse()
md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
```

###### Object Storage
```python
s3pdf_cli = S3ReaderWriter(pdf_ak, pdf_sk, pdf_endpoint)
image_dir = "s3://img_bucket/"
s3image_cli = S3ReaderWriter(img_ak, img_sk, img_endpoint, parent_path=image_dir)
pdf_bytes = s3pdf_cli.read(s3_pdf_path, mode=s3pdf_cli.MODE_BIN)
jso_useful_key = {"_pdf_type": "", "model_list": model_json}
pipe = UNIPipe(pdf_bytes, jso_useful_key, s3image_cli)
pipe.pipe_classify()
pipe.pipe_parse()
md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
```

Demo can be referred to [demo.py](https://github.com/magicpdf/Magic-PDF/blob/master/demo/demo.py)

## All Thanks To Our Contributors

<a href="https://github.com/magicpdf/Magic-PDF/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=magicpdf/Magic-PDF" />
</a>

## License Information

See [LICENSE.md](https://github.com/magicpdf/Magic-PDF/blob/master/LICENSE.md) for details.

## Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
