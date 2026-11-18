# MinerU

![MinerU](../images/MinerU-logo.png){ width="300" }

MinerU converts PDFs, images, and office documents into Markdown and structured data. Review results alongside the source in a browser, convert a folder locally, or call the parser from your application.

## Get started

| What do you want to do? | Where to start |
| --- | --- |
| Try a document online | Open the [Hugging Face demo](https://huggingface.co/spaces/opendatalab/MinerU) and follow the [WebUI guide](usage/webui.md) to upload, review, and download the result. |
| Convert a local file to Markdown | Follow the [quick start](quick_start/index.md) to install the base package, convert a file, and open the resulting `.md`. |
| Parse documents from an application | Use the [Python SDK](usage/sdk_api.md) in your process, or send your [first HTTP request](usage/http_api.md) to a service. |
| Search and revisit parsed content | Use the [document library](usage/quick_usage.md) to keep results and read passages by locator. |

## Overview

PDFs and images support four parsing tiers. Choose Flash, Basic, Standard, or Advanced according to the layout, available hardware, and speed you need. [Tiers and runtimes](usage/tiers.md)

MinerU uses DocVortex for local native Flash parsing of DOC/DOCX, PPT/PPTX, XLS/XLSX, RTF, ODT/ODS/ODP, EPUB, OFD, HTML/MHTML, and CSV/TSV. The parsing service, document library, and WebUI each expose their own set of exports. [Output formats](reference/output_files.md)

## Releases and project resources

The [changelog](reference/changelog.md) lists features and compatibility changes by release. For an older deployment, see the [3.x → 4.0](reference/migration_4.md) and [4.0.x → 4.1](reference/migration_4_1.md){#mineru-41-automatic-acceleration-and-updated-parsing-apis} migration guides.

[GitHub](https://github.com/opendatalab/MinerU) · [Official application](https://mineru.net/) · [Colab example](https://colab.research.google.com/gist/myhloli/a3cb16570ab3cfeadf9d8f0ac91b4fca/mineru_demo.ipynb) · [License](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md)

Technical reports include [MinerU](https://arxiv.org/abs/2409.18839), [MinerU2.5](https://arxiv.org/abs/2509.22186), and [MinerU2.5 Pro](https://arxiv.org/abs/2604.04771).

[![PyPI version](https://img.shields.io/pypi/v/mineru)](https://pypi.org/project/mineru/) [![Python versions](https://img.shields.io/pypi/pyversions/mineru)](https://pypi.org/project/mineru/)
