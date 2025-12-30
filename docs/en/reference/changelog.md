# Changelog

This document records the update history of MinerU project for version 2.6.7 and earlier. For the latest version updates, please check the project [README](https://github.com/opendatalab/MinerU/blob/master/README.md).

---

## 2.6 Series Versions

### 2.6.7 (2025/12/12)

- Bug fix: #4168

### 2.6.6 (2025/12/02)

**`mineru-api` tool optimizations**

- Added descriptive text to `mineru-api` interface parameters to improve API documentation readability.
- You can use the environment variable `MINERU_API_ENABLE_FASTAPI_DOCS` to control whether the auto-generated interface documentation page is enabled (enabled by default).
- Added concurrency configuration options for the `vlm-vllm-async-engine`, `vlm-lmdeploy-engine`, and `vlm-http-client` backends. Users can use the environment variable `MINERU_API_MAX_CONCURRENT_REQUESTS` to set the maximum number of concurrent API requests (unlimited by default).

### 2.6.5 (2025/11/26)

- Added support for a new backend vlm-lmdeploy-engine. Its usage is similar to vlm-vllm-(async)engine, but it uses lmdeploy as the inference engine and additionally supports native inference acceleration on Windows platforms compared to vllm.

### 2.6.4 (2025/11/04)

- Added timeout configuration for PDF image rendering, default is 300 seconds, can be configured via environment variable `MINERU_PDF_RENDER_TIMEOUT` to prevent long blocking of the rendering process caused by some abnormal PDF files.
- Added CPU thread count configuration options for ONNX models, default is the system CPU core count, can be configured via environment variables `MINERU_INTRA_OP_NUM_THREADS` and `MINERU_INTER_OP_NUM_THREADS` to reduce CPU resource contention conflicts in high concurrency scenarios.

### 2.6.3 (2025/10/31)

- Added support for a new backend `vlm-mlx-engine`, enabling MLX-accelerated inference for the MinerU2.5 model on Apple Silicon devices. Compared to the `vlm-transformers` backend, `vlm-mlx-engine` delivers a 100%–200% speed improvement.
- Bug fixes: #3849, #3859

### 2.6.2 (2025/10/24)

**`pipeline` backend optimizations**

- Added experimental support for Chinese formulas, which can be enabled by setting the environment variable `export MINERU_FORMULA_CH_SUPPORT=1`. This feature may cause a slight decrease in MFR speed and failures in recognizing some long formulas. It is recommended to enable it only when parsing Chinese formulas is needed. To disable this feature, set the environment variable to `0`.
- `OCR` speed significantly improved by 200%~300%, thanks to the optimization solution provided by [@cjsdurj](https://github.com/cjsdurj)
- `OCR` models optimized for improved accuracy and coverage of Latin script recognition, and updated Cyrillic, Arabic, Devanagari, Telugu (te), and Tamil (ta) language systems to `ppocr-v5` version, with accuracy improved by over 40% compared to previous models 

**`vlm` backend optimizations**

- `table_caption` and `table_footnote` matching logic optimized to improve the accuracy of table caption and footnote matching and reading order rationality in scenarios with multiple consecutive tables on a page
- Optimized CPU resource usage during high concurrency when using `vllm` backend, reducing server pressure
- Adapted to `vllm` version 0.11.0

**General optimizations**

- Cross-page table merging effect optimized, added support for cross-page continuation table merging, improving table merging effectiveness in multi-column merge scenarios
- Added environment variable configuration option `MINERU_TABLE_MERGE_ENABLE` for table merging feature. Table merging is enabled by default and can be disabled by setting this variable to `0`

---

## 2.5 Series Versions

### 2.5.4 (2025/09/26)

- 🎉🎉 The MinerU2.5 [Technical Report](https://arxiv.org/abs/2509.22186) is now available! We welcome you to read it for a comprehensive overview of its model architecture, training strategy, data engineering and evaluation results.
- Fixed an issue where some `PDF` files were mistakenly identified as `AI` files, causing parsing failures

### 2.5.3 (2025/09/20)

- Dependency version range adjustment to enable Turing and earlier architecture GPUs to use vLLM acceleration for MinerU2.5 model inference.
- `pipeline` backend compatibility fixes for torch 2.8.0.
- Reduced default concurrency for vLLM async backend to lower server pressure and avoid connection closure issues caused by high load.
- More compatibility-related details can be found in the [announcement](https://github.com/opendatalab/MinerU/discussions/3548)

### 2.5.2 (2025/09/19)

We are officially releasing MinerU2.5, currently the most powerful multimodal large model for document parsing.

With only 1.2B parameters, MinerU2.5's accuracy on the OmniDocBench benchmark comprehensively surpasses top-tier multimodal models like Gemini 2.5 Pro, GPT-4o, and Qwen2.5-VL-72B. It also significantly outperforms leading specialized models such as dots.ocr, MonkeyOCR, and PP-StructureV3.

The model has been released on [HuggingFace](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) and [ModelScope](https://modelscope.cn/models/opendatalab/MinerU2.5-2509-1.2B) platforms. Welcome to download and use!

**Core Highlights**

- SOTA Performance with Extreme Efficiency: As a 1.2B model, it achieves State-of-the-Art (SOTA) results that exceed models in the 10B and 100B+ classes, redefining the performance-per-parameter standard in document AI.
- Advanced Architecture for Across-the-Board Leadership: By combining a two-stage inference pipeline (decoupling layout analysis from content recognition) with a native high-resolution architecture, it achieves SOTA performance across five key areas: layout analysis, text recognition, formula recognition, table recognition, and reading order.

**Key Capability Enhancements**

- Layout Detection: Delivers more complete results by accurately covering non-body content like headers, footers, and page numbers. It also provides more precise element localization and natural format reconstruction for lists and references.
- Table Parsing: Drastically improves parsing for challenging cases, including rotated tables, borderless/semi-structured tables, and long/complex tables.
- Formula Recognition: Significantly boosts accuracy for complex, long-form, and hybrid Chinese-English formulas, greatly enhancing the parsing capability for mathematical documents.

**Repository Adjustments**

Additionally, with the release of vlm 2.5, we have made some adjustments to the repository:

- The vlm backend has been upgraded to version 2.5, supporting the MinerU2.5 model and no longer compatible with the MinerU2.0-2505-0.9B model. The last version supporting the 2.0 model is mineru-2.2.2.
- VLM inference-related code has been moved to [mineru_vl_utils](https://github.com/opendatalab/mineru-vl-utils), reducing coupling with the main mineru repository and facilitating independent iteration in the future.
- The vlm accelerated inference framework has been switched from `sglang` to `vllm`, achieving full compatibility with the vllm ecosystem, allowing users to use the MinerU2.5 model and accelerated inference on any platform that supports the vllm framework.
- Due to major upgrades in the vlm model supporting more layout types, we have made some adjustments to the structure of the parsing intermediate file `middle.json` and result file `content_list.json`. Please refer to the [documentation](https://opendatalab.github.io/MinerU/reference/output_files/) for details.

**Other Repository Optimizations**

- Removed file extension whitelist validation for input files. When input files are PDF documents or images, there are no longer requirements for file extensions, improving usability.

---

## 2.2 - 2.4 Series Versions

### 2.2.2 (2025/09/10)

- Fixed the issue where the new table recognition model would affect the overall parsing task when some table parsing failed

### 2.2.1 (2025/09/08)

- Fixed the issue where some newly added models were not downloaded when using the model download command.

### 2.2.0 (2025/09/05)

**Major Updates**

- In this version, we focused on improving table parsing accuracy by introducing a new [wired table recognition model](https://github.com/RapidAI/TableStructureRec) and a brand-new hybrid table structure parsing algorithm, significantly enhancing the table recognition capabilities of the `pipeline` backend.
- We also added support for cross-page table merging, which is supported by both `pipeline` and `vlm` backends, further improving the completeness and accuracy of table parsing.

**Other Updates**

- The `pipeline` backend now supports 270-degree rotated table parsing, bringing support for table parsing in 0/90/270-degree orientations
- `pipeline` added OCR capability support for Thai and Greek, and updated the English OCR model to the latest version. English recognition accuracy improved by 11%, Thai recognition model accuracy is 82.68%, and Greek recognition model accuracy is 89.28% (by PPOCRv5)
- Added `bbox` field (mapped to 0-1000 range) in the output `content_list.json`, making it convenient for users to directly obtain position information for each content block
- Removed the `pipeline_old_linux` installation option, no longer supporting legacy Linux systems such as `CentOS 7`, to provide better support for `uv`'s `sync`/`run` commands

---

## 2.1 Series Versions

### 2.1.10 (2025/08/01)

- Fixed an issue in the `pipeline` backend where block overlap caused the parsing results to deviate from expectations #3232

### 2.1.9 (2025/07/30)

- `transformers` 4.54.1 version adaptation

### 2.1.8 (2025/07/28)

- `sglang` 0.4.9.post5 version adaptation

### 2.1.7 (2025/07/27)

- `transformers` 4.54.0 version adaptation

### 2.1.6 (2025/07/26)

- Fixed table parsing issues in handwritten documents when using `vlm` backend
- Fixed visualization box position drift issue when document is rotated #3175

### 2.1.5 (2025/07/24)

- `sglang` 0.4.9 version adaptation, synchronously upgrading the dockerfile base image to sglang 0.4.9.post3

### 2.1.4 (2025/07/23)

**Bug Fixes**

- Fixed the issue of excessive memory consumption during the `MFR` step in the `pipeline` backend under certain scenarios #2771
- Fixed the inaccurate matching between `image`/`table` and `caption`/`footnote` under certain conditions #3129

### 2.1.1 (2025/07/16)

**Bug fixes**

- Fixed text block content loss issue that could occur in certain `pipeline` scenarios #3005
- Fixed issue where `sglang-client` required unnecessary packages like `torch` #2968
- Updated `dockerfile` to fix incomplete text content parsing due to missing fonts in Linux #2915

**Usability improvements**

- Updated `compose.yaml` to facilitate direct startup of `sglang-server`, `mineru-api`, and `mineru-gradio` services
- Launched brand new [online documentation site](https://opendatalab.github.io/MinerU/), simplified readme, providing better documentation experience

### 2.1.0 (2025/07/05)

This is the first major update of MinerU 2, which includes a large number of new features and improvements, covering significant performance optimizations, user experience enhancements, and bug fixes. The detailed update contents are as follows:

**Performance Optimizations**

- Significantly improved preprocessing speed for documents with specific resolutions (around 2000 pixels on the long side).
- Greatly enhanced post-processing speed when the `pipeline` backend handles batch processing of documents with fewer pages (<10 pages).
- Layout analysis speed of the `pipeline` backend has been increased by approximately 20%.

**Experience Enhancements**

- Built-in ready-to-use `fastapi service` and `gradio webui`. For detailed usage instructions, please refer to [Documentation](https://opendatalab.github.io/MinerU/usage/quick_usage/#advanced-usage-via-api-webui-sglang-clientserver).
- Adapted to `sglang` version `0.4.8`, significantly reducing the GPU memory requirements for the `vlm-sglang` backend. It can now run on graphics cards with as little as `8GB GPU memory` (Turing architecture or newer).
- Added transparent parameter passing for all commands related to `sglang`, allowing the `sglang-engine` backend to receive all `sglang` parameters consistently with the `sglang-server`.
- Supports feature extensions based on configuration files, including `custom formula delimiters`, `enabling heading classification`, and `customizing local model directories`. For detailed usage instructions, please refer to [Documentation](https://opendatalab.github.io/MinerU/usage/quick_usage/#extending-mineru-functionality-with-configuration-files).

**New Features**

- Updated the `pipeline` backend with the PP-OCRv5 multilingual text recognition model, supporting text recognition in 37 languages such as French, Spanish, Portuguese, Russian, and Korean, with an average accuracy improvement of over 30%. [Details](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
- Introduced limited support for vertical text layout in the `pipeline` backend.

---

## 2.0 Series Versions

### 2.0.6 (2025/06/20)

- Fixed occasional parsing interruptions caused by invalid block content in `vlm` mode
- Fixed parsing interruptions caused by incomplete table structures in `vlm` mode

### 2.0.5 (2025/06/17)

- Fixed the issue where models were still required to be downloaded in the `sglang-client` mode
- Fixed the issue where the `sglang-client` mode unnecessarily depended on packages like `torch` during runtime.
- Fixed the issue where only the first instance would take effect when attempting to launch multiple `sglang-client` instances via multiple URLs within the same process

### 2.0.3 (2025/06/15)

- Fixed a configuration file key-value update error that occurred when downloading model type was set to `all`
- Fixed the issue where the formula and table feature toggle switches were not working in `command line mode`, causing the features to remain enabled.
- Fixed compatibility issues with sglang version 0.4.7 in the `sglang-engine` mode.
- Updated Dockerfile and installation documentation for deploying the full version of MinerU in sglang environment

### 2.0.0 (2025/06/13)

**New Architecture**

MinerU 2.0 has been deeply restructured in code organization and interaction methods, significantly improving system usability, maintainability, and extensibility.

- **Removal of Third-party Dependency Limitations**: Completely eliminated the dependency on `pymupdf`, moving the project toward a more open and compliant open-source direction.
- **Ready-to-use, Easy Configuration**: No need to manually edit JSON configuration files; most parameters can now be set directly via command line or API.
- **Automatic Model Management**: Added automatic model download and update mechanisms, allowing users to complete model deployment without manual intervention.
- **Offline Deployment Friendly**: Provides built-in model download commands, supporting deployment requirements in completely offline environments.
- **Streamlined Code Structure**: Removed thousands of lines of redundant code, simplified class inheritance logic, significantly improving code readability and development efficiency.
- **Unified Intermediate Format Output**: Adopted standardized `middle_json` format, compatible with most secondary development scenarios based on this format, ensuring seamless ecosystem business migration.

**New Model**

MinerU 2.0 integrates our latest small-parameter, high-performance multimodal document parsing model, achieving end-to-end high-speed, high-precision document understanding.

- **Small Model, Big Capabilities**: With parameters under 1B, yet surpassing traditional 72B-level vision-language models (VLMs) in parsing accuracy.
- **Multiple Functions in One**: A single model covers multilingual recognition, handwriting recognition, layout analysis, table parsing, formula recognition, reading order sorting, and other core tasks.
- **Ultimate Inference Speed**: Achieves peak throughput exceeding 10,000 tokens/s through `sglang` acceleration on a single NVIDIA 4090 card, easily handling large-scale document processing requirements.
- **Online Experience**: You can experience our brand-new VLM model on [MinerU.net](https://mineru.net/OpenSourceTools/Extractor), [Hugging Face](https://huggingface.co/spaces/opendatalab/MinerU), and [ModelScope](https://www.modelscope.cn/studios/OpenDataLab/MinerU).

**Incompatible Changes Notice**

To improve overall architectural rationality and long-term maintainability, this version contains some incompatible changes:

- Python package name changed from `magic-pdf` to `mineru`, and the command-line tool changed from `magic-pdf` to `mineru`. Please update your scripts and command calls accordingly.
- For modular system design and ecosystem consistency considerations, MinerU 2.0 no longer includes the LibreOffice document conversion module. If you need to process Office documents, we recommend converting them to PDF format through an independently deployed LibreOffice service before proceeding with subsequent parsing operations.

---

## 1.x Series Historical Versions

### 1.3.12 (2025/05/24)

Added support for PPOCRv5 models, updated `ch_server` model to `PP-OCRv5_rec_server`, and `ch_lite` model to `PP-OCRv5_rec_mobile` (model update required)

- In testing, we found that PPOCRv5(server) has some improvement for handwritten documents, but has slightly lower accuracy than v4_server_doc for other document types, so the default ch model remains unchanged as `PP-OCRv4_server_rec_doc`.
- Since PPOCRv5 has enhanced recognition capabilities for handwriting and special characters, you can manually choose the PPOCRv5 model for Japanese-Traditional Chinese mixed scenarios and handwritten documents
- You can select the appropriate model through the lang parameter `lang='ch_server'` (Python API) or `--lang ch_server` (command line):
  - `ch`: `PP-OCRv4_server_rec_doc` (default) (Chinese/English/Japanese/Traditional Chinese mixed/15K dictionary)
  - `ch_server`: `PP-OCRv5_rec_server` (Chinese/English/Japanese/Traditional Chinese mixed + handwriting/18K dictionary)
  - `ch_lite`: `PP-OCRv5_rec_mobile` (Chinese/English/Japanese/Traditional Chinese mixed + handwriting/18K dictionary)
  - `ch_server_v4`: `PP-OCRv4_rec_server` (Chinese/English mixed/6K dictionary)
  - `ch_lite_v4`: `PP-OCRv4_rec_mobile` (Chinese/English mixed/6K dictionary)

Added support for handwritten documents through optimized layout recognition of handwritten text areas

- This feature is supported by default, no additional configuration required
- You can refer to the instructions above to manually select the PPOCRv5 model for better handwritten document parsing results

The `huggingface` and `modelscope` demos have been updated to versions that support handwriting recognition and PPOCRv5 models, which you can experience online

### 1.3.10 (2025/04/29)

- Added support for custom formula delimiters, which can be configured by modifying the `latex-delimiter-config` section in the `magic-pdf.json` file in your user directory.

### 1.3.9 (2025/04/27)

- Optimized formula parsing functionality, improved formula rendering success rate

### 1.3.8 (2025/04/23)

The default `ocr` model (`ch`) has been updated to `PP-OCRv4_server_rec_doc` (model update required)

- `PP-OCRv4_server_rec_doc` is trained on a mixture of more Chinese document data and PP-OCR training data based on `PP-OCRv4_server_rec`, adding recognition capabilities for some traditional Chinese characters, Japanese, and special characters. It can recognize over 15,000 characters and improves both document-specific and general text recognition abilities.
- [Performance comparison of PP-OCRv4_server_rec_doc/PP-OCRv4_server_rec/PP-OCRv4_mobile_rec](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_recognition.html#_3)
- After verification, the `PP-OCRv4_server_rec_doc` model shows significant accuracy improvements in Chinese/English/Japanese/Traditional Chinese in both single language and mixed language scenarios, with comparable speed to `PP-OCRv4_server_rec`, making it suitable for most use cases.
- In some pure English scenarios, `PP-OCRv4_server_rec_doc` may have word adhesion issues, while `PP-OCRv4_server_rec` performs better in these cases. Therefore, we've kept the `PP-OCRv4_server_rec` model, which users can access by adding the parameter `lang='ch_server'` (Python API) or `--lang ch_server` (command line).

### 1.3.7 (2025/04/22)

- Fixed the issue where the lang parameter was ineffective during table parsing model initialization
- Fixed the significant speed reduction of OCR and table parsing in `cpu` mode

### 1.3.4 (2025/04/16)

- Slightly improved OCR-det speed by removing some unnecessary blocks
- Fixed page-internal sorting errors caused by footnotes in certain cases

### 1.3.2 (2025/04/12)

- Fixed dependency version incompatibility issues when installing on Windows with Python 3.13
- Optimized memory usage during batch inference
- Improved parsing of tables rotated 90 degrees
- Enhanced parsing of oversized tables in financial report samples
- Fixed the occasional word adhesion issue in English text areas when OCR language is not specified (model update required)

### 1.3.1 (2025/04/08)

Fixed several compatibility issues

- Added support for Python 3.13
- Made final adaptations for outdated Linux systems (such as CentOS 7) with no guarantee of continued support in future versions, [installation instructions](https://github.com/opendatalab/MinerU/issues/1004)

### 1.3.0 (2025/04/03)

**Installation and compatibility optimizations**

- Resolved compatibility issues caused by `detectron2` by removing `layoutlmv3` usage in layout
- Extended torch version compatibility to 2.2~2.6 (excluding 2.5)
- Added CUDA compatibility for versions 11.8/12.4/12.6/12.8 (CUDA version determined by torch), solving compatibility issues for users with 50-series and H-series GPUs
- Extended Python compatibility to versions 3.10~3.12, fixing the issue of automatic downgrade to version 0.6.1 when installing in non-3.10 environments
- Optimized offline deployment process, eliminating the need to download any model files after successful deployment

**Performance optimizations**

- Enhanced parsing speed for batches of small files by supporting batch processing of multiple PDF files ([script example](demo/batch_demo.py)), with formula parsing speed improved by up to 1400% and overall parsing speed improved by up to 500% compared to version 1.0.1
- Reduced memory usage and improved parsing speed by optimizing MFR model loading and usage (requires re-running the [model download process](docs/how_to_download_models_zh_cn.md) to get incremental updates to model files)
- Optimized GPU memory usage, requiring only 6GB minimum to run this project
- Improved running speed on MPS devices

**Parsing effect optimizations**

- Updated MFR model to `unimernet(2503)`, fixing line break loss issues in multi-line formulas

**Usability optimizations**

- Completely replaced the `paddle` framework and `paddleocr` in the project by using `paddleocr2torch`, resolving conflicts between `paddle` and `torch`, as well as thread safety issues caused by the `paddle` framework
- Added real-time progress bar display during parsing, allowing precise tracking of parsing progress and making the waiting process more bearable

### 1.2.1 (2025/03/03)

Fixed some issues

- Fixed the impact on punctuation marks during full-width to half-width conversion of letters and numbers
- Fixed caption matching inaccuracies in certain scenarios
- Fixed formula span loss issues in certain scenarios

### 1.2.0 (2025/02/24)

This version includes several fixes and improvements to enhance parsing efficiency and accuracy:

**Performance Optimization**

- Increased classification speed for PDF documents in auto mode.

**Parsing Optimization**

- Improved parsing logic for documents containing watermarks, significantly enhancing the parsing results for such documents.
- Enhanced the matching logic for multiple images/tables and captions within a single page, improving the accuracy of image-text matching in complex layouts.

**Bug Fixes**

- Fixed an issue where image/table spans were incorrectly filled into text blocks under certain conditions.
- Resolved an issue where title blocks were empty in some cases.

### 1.1.0 (2025/01/22)

In this version we have focused on improving parsing accuracy and efficiency:

**Model capability upgrade** (requires re-executing the [model download process](https://github.com/opendatalab/MinerU/blob/master/docs/how_to_download_models_en.md) to obtain incremental updates of model files)

- The layout recognition model has been upgraded to the latest `doclayout_yolo(2501)` model, improving layout recognition accuracy.
- The formula parsing model has been upgraded to the latest `unimernet(2501)` model, improving formula recognition accuracy.

**Performance optimization**

- On devices that meet certain configuration requirements (16GB+ VRAM), by optimizing resource usage and restructuring the processing pipeline, overall parsing speed has been increased by more than 50%.

**Parsing effect optimization**

- Added a new heading classification feature (testing version, enabled by default) to the online demo ([mineru.net](https://mineru.net/OpenSourceTools/Extractor)/[huggingface](https://huggingface.co/spaces/opendatalab/MinerU)/[modelscope](https://www.modelscope.cn/studios/OpenDataLab/MinerU)), which supports hierarchical classification of headings, thereby enhancing document structuring.

### 1.0.1 (2025/01/10)

This is our first official release, where we have introduced a completely new API interface and enhanced compatibility through extensive refactoring, as well as a brand new automatic language identification feature:

**New API Interface**

- For the data-side API, we have introduced the Dataset class, designed to provide a robust and flexible data processing framework. This framework currently supports a variety of document formats, including images (.jpg and .png), PDFs, Word documents (.doc and .docx), and PowerPoint presentations (.ppt and .pptx). It ensures effective support for data processing tasks ranging from simple to complex.
- For the user-side API, we have meticulously designed the MinerU processing workflow as a series of composable Stages. Each Stage represents a specific processing step, allowing users to define new Stages according to their needs and creatively combine these stages to customize their data processing workflows.

**Enhanced Compatibility**

- By optimizing the dependency environment and configuration items, we ensure stable and efficient operation on ARM architecture Linux systems.
- We have deeply integrated with Huawei Ascend NPU acceleration, providing autonomous and controllable high-performance computing capabilities. This supports the localization and development of AI application platforms in China. [Ascend NPU Acceleration](https://github.com/opendatalab/MinerU/blob/master/docs/README_Ascend_NPU_Acceleration_zh_CN.md)

**Automatic Language Identification**

- By introducing a new language recognition model, setting the `lang` configuration to `auto` during document parsing will automatically select the appropriate OCR language model, improving the accuracy of scanned document parsing.

---

## 0.x Series Historical Versions

### 0.10.0 (2024/11/22)

Introducing hybrid OCR text extraction capabilities:

- Significantly improved parsing performance in complex text distribution scenarios such as dense formulas, irregular span regions, and text represented by images.
- Combines the dual advantages of accurate content extraction and faster speed in text mode, and more precise span/line region recognition in OCR mode.

### 0.9.3 (2024/11/15)

Integrated [RapidTable](https://github.com/RapidAI/RapidTable) for table recognition, improving single-table parsing speed by more than 10 times, with higher accuracy and lower GPU memory usage.

### 0.9.2 (2024/11/06)

Integrated the [StructTable-InternVL2-1B](https://huggingface.co/U4R/StructTable-InternVL2-1B) model for table recognition functionality.

### 0.9.0 (2024/10/31)

This is a major new version with extensive code refactoring, addressing numerous issues, improving performance, reducing hardware requirements, and enhancing usability:

- Refactored the sorting module code to use [layoutreader](https://github.com/ppaanngggg/layoutreader) for reading order sorting, ensuring high accuracy in various layouts.
- Refactored the paragraph concatenation module to achieve good results in cross-column, cross-page, cross-figure, and cross-table scenarios.
- Refactored the list and table of contents recognition functions, significantly improving the accuracy of list blocks and table of contents blocks, as well as the parsing of corresponding text paragraphs.
- Refactored the matching logic for figures, tables, and descriptive text, greatly enhancing the accuracy of matching captions and footnotes to figures and tables, and reducing the loss rate of descriptive text to near zero.
- Added multi-language support for OCR, supporting detection and recognition of 84 languages. For the list of supported languages, see [OCR Language Support List](https://paddlepaddle.github.io/PaddleOCR/latest/en/ppocr/blog/multi_languages.html#5).
- Added memory recycling logic and other memory optimization measures, significantly reducing memory usage. The memory requirement for enabling all acceleration features except table acceleration (layout/formula/OCR) has been reduced from 16GB to 8GB, and the memory requirement for enabling all acceleration features has been reduced from 24GB to 10GB.
- Optimized configuration file feature switches, adding an independent formula detection switch to significantly improve speed and parsing results when formula detection is not needed.
- Integrated [PDF-Extract-Kit 1.0](https://github.com/opendatalab/PDF-Extract-Kit):
  - Added the self-developed `doclayout_yolo` model, which speeds up processing by more than 10 times compared to the original solution while maintaining similar parsing effects, and can be freely switched with `layoutlmv3` via the configuration file.
  - Upgraded formula parsing to `unimernet 0.2.1`, improving formula parsing accuracy while significantly reducing memory usage.
  - Due to the repository change for `PDF-Extract-Kit 1.0`, you need to re-download the model. Please refer to [How to Download Models](https://github.com/opendatalab/MinerU/blob/master/docs/how_to_download_models_en.md) for detailed steps.

### 0.8.1 (2024/09/27)

Fixed some bugs, and providing a [localized deployment version](https://github.com/opendatalab/MinerU/blob/master/projects/web_demo/README.md) of the [online demo](https://opendatalab.com/OpenSourceTools/Extractor/PDF/) and the [front-end interface](https://github.com/opendatalab/MinerU/blob/master/projects/web/README.md).

### 0.8.0 (2024/09/09)

Supporting fast deployment with Dockerfile, and launching demos on Huggingface and Modelscope.

### 0.7.1 (2024/08/30)

Add paddle tablemaster table recognition option

### 0.7.0b1 (2024/08/09)

Simplified installation process, added table recognition functionality

### 0.6.2b1 (2024/08/01)

Optimized dependency conflict issues and installation documentation

### Initial Open-Source Release (2024/07/05)

MinerU project's first open-source release

