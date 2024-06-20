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

Magic-PDF is a tool designed to convert PDF documents into markdown format, capable of processing files stored locally or on object storage supporting S3 protocol.

Key features include:

- Support for multiple front-end model inputs
- Removal of headers, footers, footnotes, and page numbers
- Human-readable layout formatting
- Extraction and display of images and tables within markdown
- Conversion of equations into LaTeX format
- Automatic detection and conversion of garbled PDFs
- Compatibility with CPU and GPU environments
- Available for Windows, Linux, and macOS platforms

## Getting Started

### Requirements

- Python 3.9 or newer

### Usage Instructions

1. **Install Magic-PDF**

```bash
pip install magic-pdf[cpu] # Install the CPU version 
or
pip install magic-pdf[gpu] # Install the GPU version
```

2. **Usage via Command Line**

```bash
magic-pdf --help
```

## License Information

See [LICENSE.md](https://github.com/magicpdf/Magic-PDF/blob/master/LICENSE.md) for details.

## Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
