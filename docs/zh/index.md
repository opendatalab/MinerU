# MinerU 4.0

![MinerU](../images/MinerU-logo.png){ width="300" }

MinerU 4.0 将多格式文档解析、文档库和服务工具整合到统一工作流，面向文档转换、应用集成和 Agent 阅读。

- **四档解析**：Flash 用于快速预览与索引，Basic 提供基础 OCR/模型解析，Standard 与 Advanced 面向更复杂的版面和更高的质量需求。
- **多格式输入**：PDF、图片，以及 DOC/DOCX、PPT/PPTX、XLS/XLSX、RTF、ODT/ODS/ODP、EPUB、OFD、HTML、CSV/TSV。原生文档由 DocVortex 提供解析能力。
- **文档库与 Agent 阅读**：发现文件、缓存结果、搜索内容，按页或块继续阅读，并保留稳定引用位置。
- **独立模型配置**：小模型选择 ONNX 或 Torch；VLM 独立选择 llama.cpp、vLLM、LMDeploy，或手动安装并显式配置 MLX。
- **统一工具入口**：Python SDK、V1 API、无状态批处理、多服务 Router 和基于 Gradio 的 WebUI。
- **结构化结果与渲染**：统一文档模型支持 Markdown、HTML、LaTeX、DOCX、EPUB、PDF、Structured Content、Content List V1/V2 九种渲染目标；各 CLI/API 的导出范围见输出说明。

PDF 和图片支持四档解析；Office、OpenDocument、EPUB、OFD、HTML、CSV/TSV 使用本地 Flash 原生解析。纯文本直接读取，不进入解析流程。默认不会自动将文档上传到官网服务，远程解析需要显式配置。

## 开始使用

- [安装与快速入门](quick_start/index.md)
- [档位与运行环境](usage/tiers.md)
- [Python SDK 与 V1 API](usage/sdk_api.md)
- [3.x → 4.0 迁移](reference/migration_4.md)
- [旧平台适配（MinerU <4）](usage/compatibility.md)
- [更新历史](reference/changelog.md)

[GitHub](https://github.com/opendatalab/MinerU) · [MinerU](https://mineru.net/) · [License](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md)
