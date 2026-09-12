# MinerU 4.0

![MinerU](../images/MinerU-logo.png){ width="300" }

MinerU 4.0 brings document parsing, a local document library, and service tools into one workflow for document conversion, application integration, and agent reading.

- **Four parsing tiers**: Flash for fast previews and indexing, Basic for OCR and model-based parsing, and Standard / Advanced for more demanding layouts and quality requirements.
- **Multiple input formats**: PDF, images, DOC/DOCX, PPT/PPTX, XLS/XLSX, RTF, ODT/ODS/ODP, EPUB, OFD, HTML, and CSV/TSV. DocVortex provides native document parsing.
- **Document library and agent reading**: discover files, cache results, search content, continue by page or block, and preserve stable citation locators.
- **Independent model configuration**: ONNX or Torch for small models; llama.cpp, vLLM, LMDeploy, or manually installed and explicitly configured MLX for the VLM.
- **Unified tools**: Python SDK, V1 API, stateless batch conversion, multi-service Router, and a Gradio-based WebUI.
- **Structured results and rendering**: one document model supports nine rendering targets: Markdown, HTML, LaTeX, DOCX, EPUB, PDF, Structured Content, and Content List V1/V2. Each CLI/API exposes its own subset of exports.

PDF and images support all four tiers. Office, OpenDocument, EPUB, OFD, HTML, and CSV/TSV use local Flash native parsing. Plain text is read directly rather than parsed. Documents are not automatically uploaded to the official service; remote parsing requires explicit configuration.

## Get started

- [Install and get started](quick_start/index.md)
- [Tiers and runtimes](usage/tiers.md)
- [Python SDK and V1 API](usage/sdk_api.md)
- [3.x → 4.0 migration](reference/migration_4.md)
- [Legacy platforms (MinerU <4)](usage/compatibility.md)
- [Release history](reference/changelog.md)

[GitHub](https://github.com/opendatalab/MinerU) · [MinerU](https://mineru.net/) · [License](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md)
