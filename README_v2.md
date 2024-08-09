<div align="center" xmlns="http://www.w3.org/1999/html">
<!-- logo -->
<p align="center">
  <img src="docs/images/MinerU-logo.png" width="300px" style="vertical-align:middle;">
</p>

<!-- icon -->
[![stars](https://img.shields.io/github/stars/opendatalab/MinerU.svg)](https://github.com/opendatalab/MinerU)
[![forks](https://img.shields.io/github/forks/opendatalab/MinerU.svg)](https://github.com/opendatalab/MinerU)
[![open issues](https://img.shields.io/github/issues-raw/opendatalab/MinerU)](https://github.com/opendatalab/MinerU/issues)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/opendatalab/MinerU)](https://github.com/opendatalab/MinerU/issues)
[![PyPI version](https://badge.fury.io/py/magic-pdf.svg)](https://badge.fury.io/py/magic-pdf)
[![Downloads](https://static.pepy.tech/badge/magic-pdf)](https://pepy.tech/project/magic-pdf)
[![Downloads](https://static.pepy.tech/badge/magic-pdf/month)](https://pepy.tech/project/magic-pdf)
<a href="https://trendshift.io/repositories/11174" target="_blank"><img src="https://trendshift.io/api/badge/repositories/11174" alt="opendatalab%2FMinerU | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<!-- language -->
[English](README.md) | [简体中文](README_zh-CN.md)

<!-- hot link -->
<p align="center">
<a href="https://github.com/opendatalab/PDF-Extract-Kit">PDF-Extract-Kit: High-Quality PDF Extraction Toolkit</a>🔥🔥🔥
</p>

<!-- join us -->
<p align="center">
    👋 join us on <a href="https://discord.gg/gPxmVeGC" target="_blank">Discord</a> and <a href="https://cdn.vansin.top/internlm/mineru.jpg" target="_blank">WeChat</a>
</p>

</div>

# Changelog
- 2024/08/09: Version 0.7.0b1 released, simplified installation process, added table recognition functionality
- 2024/08/01: Version 0.6.2b1 released, optimized dependency conflict issues and installation documentation
- 2024/07/05: Initial open-source release

<!-- TABLE OF CONTENT -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#mineru">MinerU</a>
      <ul>
        <li><a href="#project-introduction">Project Introduction</a></li>
        <li><a href="#key-features">Key Features</a></li>
        <li><a href="#quick-start">Quick Start</a>
            <ul>
            <li><a href="#online-demo">Online Demo</a></li>
            <li><a href="#quick-cpu-demo">Quick CPU Demo</a></li>
            <li><a href="#gpu-usage">GPU Usage</a></li>
            </ul>
        </li>
        <li><a href="#usage">Usage</a>
            <ul>
            <li><a href="#command-line">Command Line</a></li>
            <li><a href="#api">API</a></li>
            <li><a href="#advanced-development">Advanced Development</a></li>
            </ul>
        </li>
      </ul>
    </li>
    <li><a href="#todo">TODO List</a></li>
    <li><a href="#known-issue">Known Issue</a></li>
    <li><a href="#faq">FAQ</a></li>
    <li><a href="#all-thanks-to-our-contributors">Contributors</a></li>
    <li><a href="#license-information">License Information</a></li>
    <li><a href="#acknowledgments">Acknowledgements</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#star-history">Star History</a></li>
    <li><a href="#magic-doc">Magic-doc: Fast Extraction from PPT/DOC/PDF</a></li>
    <li><a href="#magic-html">Magic-html: Mixed Web Page Extraction</a></li>
    <li><a href="#links">Links</a></li>
  </ol>
</details>

# MinerU
## Project Introduction
MinerU is a tool that converts PDFs into machine-readable formats (e.g., markdown, JSON), allowing for easy extraction into any format.
MinerU was born during the pre-training process of [InternLM](https://github.com/InternLM/InternLM). We focus on solving symbol conversion issues in scientific literature and hope to contribute to technological development in the era of large models.
Compared to well-known commercial products, MinerU is still young. If you encounter any issues or if the results are not as expected, please submit an issue on [issue](https://github.com/opendatalab/MinerU/issues) and **attach the relevant PDF**.

https://github.com/user-attachments/assets/4bea02c9-6d54-4cd6-97ed-dff14340982c

## Key Features

- Removes elements such as headers, footers, footnotes, and page numbers while maintaining semantic continuity
- Outputs text in a human-readable order from multi-column documents
- Retains the original structure of the document, including titles, paragraphs, and lists
- Extracts images, image captions, tables, and table captions
- Automatically recognizes formulas in the document and converts them to LaTeX
- Automatically recognizes tables in the document and converts them to LaTeX
- Automatically detects and enables OCR for corrupted PDFs
- Supports both CPU and GPU environments
- Supports Windows, Linux, and Mac platforms

## Quick Start

If you encounter any installation issues, please first consult the <a href="#faq">FAQ</a>. </br>
If the parsing results are not as expected, refer to the <a href="#known-issue">Known Issues</a>. </br>
There are three different ways to experience MinerU:
- [Online Demo (No Installation Required)](#online-demo)
- [Quick CPU Demo (Windows, Linux, Mac)](#quick-cpu-demo)
- [Linux/Windows + CUDA](#gpu-usage)

**⚠️ Pre-installation Notice—Hardware and Software Environment Support**

To ensure the stability and reliability of the project, we only optimize and test for specific hardware and software environments during development. This ensures that users deploying and running the project on recommended system configurations will get the best performance with the fewest compatibility issues.

By focusing resources on the mainline environment, our team can more efficiently resolve potential bugs and develop new features.

In non-mainline environments, due to the diversity of hardware and software configurations, as well as third-party dependency compatibility issues, we cannot guarantee 100% project availability. Therefore, for users who wish to use this project in non-recommended environments, we suggest carefully reading the documentation and FAQ first. Most issues already have corresponding solutions in the FAQ. We also encourage community feedback to help us gradually expand support.

<table>
    <tr>
        <td colspan="3" rowspan="2">Operating System</td>
    </tr>
    <tr>
        <td>Ubuntu 22.04 LTS</td>
        <td>Windows 10 / 11</td>
        <td>macOS 11+</td>
    </tr>
    <tr>
        <td colspan="3">CPU</td>
        <td>x86_64</td>
        <td>x86_64</td>
        <td>x86_64 / arm64</td>
    </tr>
    <tr>
        <td colspan="3">Memory</td>
        <td colspan="3">16GB or more, recommended 32GB+</td>
    </tr>
    <tr>
        <td colspan="3">Python Version</td>
        <td colspan="3">3.10</td>
    </tr>
    <tr>
        <td colspan="3">Nvidia Driver Version</td>
        <td>latest (Proprietary Driver)</td>
        <td>latest</td>
        <td>None</td>
    </tr>
    <tr>
        <td colspan="3">CUDA Environment</td>
        <td>Automatic installation [12.1 (pytorch) + 11.8 (paddle)]</td>
        <td>11.8 (manual installation) + cuDNN v8.7.0 (manual installation)</td>
        <td>None</td>
    </tr>
    <tr>
        <td rowspan="2">GPU Hardware Support List</td>
        <td colspan="2">Minimum Requirement 8G+ VRAM</td>
        <td colspan="2">3060ti/3070/3080/3080ti/4060/4070/4070ti<br>
        8G VRAM only enables layout and formula recognition acceleration</td>
        <td rowspan="2">None</td>
    </tr>
    <tr>
        <td colspan="2">Recommended Configuration 16G+ VRAM</td>
        <td colspan="2">3090/3090ti/4070ti super/4080/4090<br>
        16G or more can enable layout, formula recognition, and OCR acceleration simultaneously</td>
    </tr>
</table>

### Online Demo

[Click here for the online demo](https://opendatalab.com/OpenSourceTools/Extractor/PDF)

### Quick CPU Demo

#### 1. Install magic-pdf
```bash
conda create -n MinerU python=3.10
conda activate MinerU
pip install magic-pdf[full]==0.7.0b1 detectron2 --extra-index-url https://wheels.myhloli.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```
#### 2. Download model weight files

Refer to [How to Download Model Files](docs/how_to_download_models_zh_cn.md) for detailed instructions.
> ❗️After downloading the models, please make sure to verify the completeness of the model files.
> 
> Check if the model file sizes match the description on the webpage. If possible, use sha256 to verify the integrity of the files.

#### 3. Copy and configure the template file
You can find the `magic-pdf.template.json` template configuration file in the root directory of the repository.
> ❗️Make sure to execute the following command to copy the configuration file to your **user directory**; otherwise, the program will not run.
> 
> The user directory for Windows is `C:\Users\YourUsername`, for Linux it is `/home/YourUsername`, and for macOS it is `/Users/YourUsername`.
```bash
cp magic-pdf.template.json ~/magic-pdf.json
```

Find the `magic-pdf.json` file in your user directory and configure the "models-dir" path to point to the directory where the model weight files were downloaded in [Step 2](#2-download-model-weight-files).
> ❗️Make sure to correctly configure the **absolute path** to the model weight files directory, otherwise the program will not run because it can't find the model files.
>
> On Windows, this path should include the drive letter and all backslashes (`\`) in the path should be replaced with forward slashes (`/`) to avoid syntax errors in the JSON file due to escape sequences.
> 
> For example: If the models are stored in the "models" directory at the root of the D drive, the "model-dir" value should be `D:/models`.
```json
{
  // other config
  "models-dir": "D:/models",
  "table-config": {
        "is_table_recog_enable": false, // Table recognition is disabled by default, modify this value to enable it
        "max_time": 400
    }
}
```


### Using GPU
If your device supports CUDA and meets the GPU requirements of the mainline environment, you can use GPU acceleration. Please select the appropriate guide based on your system:

- [Ubuntu 22.04 LTS + GPU](docs/README_Ubuntu_CUDA_Acceleration_en_US.md)
- [Windows 10/11 + GPU](docs/README_Windows_CUDA_Acceleration_en_US.md)


## Usage

### Command Line

```bash
magic-pdf --help
Usage: magic-pdf [OPTIONS]

Options:
  -v, --version                display the version and exit
  -p, --path PATH              local pdf filepath or directory  [required]
  -o, --output-dir TEXT        output local directory
  -m, --method [ocr|txt|auto]  the method for parsing pdf.  
                               ocr: using ocr technique to extract information from pdf,
                               txt: suitable for the text-based pdf only and outperform ocr,
                               auto: automatically choose the best method for parsing pdf
                                  from ocr and txt.
                               without method specified, auto will be used by default. 
  --help                       Show this message and exit.


## show version
magic-pdf -v

## command line example
magic-pdf -p {some_pdf} -o {some_output_dir} -m auto
```

`{some_pdf}` can be a single PDF file or a directory containing multiple PDFs.
The results will be saved in the `{some_output_dir}` directory. The output file list is as follows:

```text
├── some_pdf.md                 # markdown file
├── images                      # directory for storing images
├── layout.pdf                  # layout diagram
├── middle.json                 # MinerU intermediate processing result
├── model.json                  # model inference result
├── origin.pdf                  # original PDF file
└── spans.pdf                   # smallest granularity bbox position information diagram
```

For more information about the output files, please refer to the [Output File Description](docs/output_file_zh_cn.md).

### API

Processing files from local disk
```python
image_writer = DiskReaderWriter(local_image_dir)
image_dir = str(os.path.basename(local_image_dir))
jso_useful_key = {"_pdf_type": "", "model_list": []}
pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
pipe.pipe_classify()
pipe.pipe_analyze()
pipe.pipe_parse()
md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
```

Processing files from object storage
```python
s3pdf_cli = S3ReaderWriter(pdf_ak, pdf_sk, pdf_endpoint)
image_dir = "s3://img_bucket/"
s3image_cli = S3ReaderWriter(img_ak, img_sk, img_endpoint, parent_path=image_dir)
pdf_bytes = s3pdf_cli.read(s3_pdf_path, mode=s3pdf_cli.MODE_BIN)
jso_useful_key = {"_pdf_type": "", "model_list": []}
pipe = UNIPipe(pdf_bytes, jso_useful_key, s3image_cli)
pipe.pipe_classify()
pipe.pipe_analyze()
pipe.pipe_parse()
md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
```

For detailed implementation, refer to:
- [demo.py Simplest Processing Method](demo/demo.py)
- [magic_pdf_parse_main.py More Detailed Processing Workflow](demo/magic_pdf_parse_main.py)


### Development Guide

TODO

# TODO

- [ ] Semantic-based reading order
- [ ] List recognition within the text
- [ ] Code block recognition within the text
- [ ] Table of contents recognition
- [x] Table recognition
- [ ] Chemical formula recognition
- [ ] Geometric shape recognition

# Known Issues
- Reading order is segmented based on rules, which can cause disordered sequences in some cases
- Vertical text is not supported
- Lists, code blocks, and table of contents are not yet supported in the layout model
- Comic books, art books, elementary school textbooks, and exercise books are not well-parsed yet
- Enabling OCR may produce better results in PDFs with a high density of formulas
- If you are processing PDFs with a large number of formulas, it is strongly recommended to enable the OCR function. When using PyMuPDF to extract text, overlapping text lines can occur, leading to inaccurate formula insertion positions.
- **Table Recognition** is currently in the testing phase; recognition speed is slow, and accuracy needs improvement. Below are some performance test results in an Ubuntu 22.04 LTS + Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz + NVIDIA GeForce RTX 4090 environment for reference.

| Table Size     | Parsing Time        | 
|---------------|----------------------------| 
| 6\*5 55kb     | 37s                   | 
| 16\*12 284kb  | 3m18s                 | 
| 44\*7 559kb   | 4m12s                 | 

# FAQ
[FAQ in Chinese](docs/FAQ_zh_cn.md)
[FAQ in English](docs/FAQ.md)


# All Thanks To Our Contributors

<a href="https://github.com/opendatalab/MinerU/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=opendatalab/MinerU" />
</a>

# License Information

[LICENSE.md](LICENSE.md)

This project currently uses PyMuPDF to achieve advanced functionality. However, since it adheres to the AGPL license, it may impose restrictions on certain usage scenarios. In future iterations, we plan to explore and replace it with a more permissive PDF processing library to enhance user-friendliness and flexibility.


# Acknowledgments
- [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit)
- [StructEqTable](https://github.com/UniModal4Reasoning/StructEqTable-Deploy)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
- [fast-langdetect](https://github.com/LlmKira/fast-langdetect)
- [pdfminer.six](https://github.com/pdfminer/pdfminer.six)

# Citation

```bibtex
@article{he2024opendatalab,
  title={Opendatalab: Empowering general artificial intelligence with open datasets},
  author={He, Conghui and Li, Wei and Jin, Zhenjiang and Xu, Chao and Wang, Bin and Lin, Dahua},
  journal={arXiv preprint arXiv:2407.13773},
  year={2024}
}

@misc{2024mineru,
    title={MinerU: A One-stop, Open-source, High-quality Data Extraction Tool},
    author={MinerU Contributors},
    howpublished = {\url{https://github.com/opendatalab/MinerU}},
    year={2024}
}
```

# Star History

<a>
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=opendatalab/MinerU&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=opendatalab/MinerU&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=opendatalab/MinerU&type=Date" />
 </picture>
</a>

# Magic-doc
[Magic-Doc](https://github.com/InternLM/magic-doc) Fast speed ppt/pptx/doc/docx/pdf extraction tool

# Magic-html
[Magic-HTML](https://github.com/opendatalab/magic-html) Mixed web page extraction tool

# Links

- [LabelU (A Lightweight Multi-modal Data Annotation Tool)](https://github.com/opendatalab/labelU)
- [LabelLLM (An Open-source LLM Dialogue Annotation Platform)](https://github.com/opendatalab/LabelLLM)
- [PDF-Extract-Kit (A Comprehensive Toolkit for High-Quality PDF Content Extraction)](https://github.com/opendatalab/PDF-Extract-Kit)
