# MinerU

![MinerU](../images/MinerU-logo.png){ width="300" }

## MinerU 4.1.0: Automatic Acceleration and Updated Parsing APIs

- **Automatic acceleration across major platforms and GPUs:** Windows and Linux gain automatic ONNX acceleration for compatible Intel, AMD, and NVIDIA GPUs, alongside existing MPS/Metal support on macOS. MinerU detects devices, prioritizes discrete GPUs, and selects GPU or CPU execution for integrated graphics based on performance and compatibility. Representative core inference workloads run at approximately **2–3× CPU speed**.
- **A unified asynchronous Tasks API:** `/v1/tasks` replaces the previous `/v1/uploads` and `/v1/parse/jobs` workflow, with file manifests, URL inputs, cached-upload reuse, per-file execution and downloads, status queries, and cancellation. This is a breaking API update; review the [4.1 migration guide](reference/migration_4_1.md) and [Tasks API reference](usage/tasks.md) before upgrading.
- **Recoverable synchronous file parsing:** The standalone `file-parse-server` provides `/v1/file_parse` for single-file uploads and attachment responses. Requests remain queryable after a wait timeout, and request IDs support recovery of known tasks. See the [file-parse API reference](usage/file_parse.md).
- **Hardware diagnostics and tier recommendations:** New `mineru-kit hardware detect` and `hardware info` commands help select a suitable service tier and inspect devices, runtime versions, and calibration information.

**Upgrade note:** Supported Python versions are now **3.11–3.14**. Deployments using the previous HTTP APIs should upgrade servers, Routers, SDKs, and clients together. See the [changelog](reference/changelog.md) for additional fixes.

## Overview

MinerU 4.0 brings document parsing, a local document library, and service tools into one workflow for document conversion, application integration, and agent reading.

- **Four parsing tiers**: Flash for fast previews and indexing, Basic for OCR and model-based parsing, and Standard / Advanced for more demanding layouts and quality requirements.
- **Multiple input formats**: PDF, images, DOC/DOCX, PPT/PPTX, XLS/XLSX, RTF, ODT/ODS/ODP, EPUB, OFD, HTML/MHTML, and CSV/TSV. [DocVortex](https://github.com/myhloli/docvortex) provides native document parsing.
- **Document library and agent reading**: discover files, cache results, search content, continue by page or block, and preserve stable citation locators.
- **Independent model configuration**: ONNX or Torch for small models; llama.cpp, vLLM, or LMDeploy for the VLM.
- **Unified tools**: Python SDK, V1 API, stateless batch conversion, multi-service Router, and a Gradio-based WebUI.
- **Structured results and rendering**: one document model supports nine rendering targets: Markdown, HTML, LaTeX, DOCX, EPUB, PDF, Structured Content, and Content List V1/V2. Each CLI/API exposes its own subset of exports; see [Output Formats and Result Contract](reference/output_files.md).

PDF and images support all four tiers. Office, OpenDocument, EPUB, OFD, HTML/MHTML, and CSV/TSV use local Flash native parsing. Plain text is read directly rather than parsed. Documents are not automatically uploaded to the official service; remote parsing requires explicit configuration.

## Get started

- [Install and get started](quick_start/index.md)
- [Tiers and runtimes](usage/tiers.md)
- [Python SDK](usage/sdk_api.md)
- [V1 HTTP API](usage/http_api.md)
- [3.x → 4.0 migration](reference/migration_4.md)
- [Legacy platforms (MinerU <4)](usage/compatibility.md)
- [Release history](reference/changelog.md)

[GitHub](https://github.com/opendatalab/MinerU) · [MinerU](https://mineru.net/) · [License](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md)
