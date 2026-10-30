# MinerU

![MinerU](../images/MinerU-logo.png){ width="300" }

## MinerU 4.1：自动加速与解析接口升级

- **主流平台与 GPU 自动加速**：Windows、Linux 新增面向 Intel、AMD、NVIDIA 兼容 GPU 的 ONNX 自动加速，结合 macOS 已有的 MPS/Metal 路径，进一步简化跨平台部署。自动识别设备、优先选择独显，并根据集显性能与兼容性选择 GPU 或 CPU；代表性核心推理场景的速度约为 CPU 的 **2～3 倍**。
- **统一 Tasks 异步接口**：以 `/v1/tasks` 替换旧 `/v1/uploads`、`/v1/parse/jobs` 工作流，支持文件清单、URL 输入、缓存免传、逐文件执行与结果下载，并统一状态查询和取消操作。此次包含破坏性接口变更，升级前请查阅 [4.1 迁移说明](reference/migration_4_1.md)和 [Tasks 接口详情](usage/tasks.md)。
- **新增可恢复的同步解析服务**：独立的 `file-parse-server` 提供 `/v1/file_parse`，上传单个文件即可等待并获取附件；等待超时后可继续查询，并通过请求 ID 恢复已知任务。详见 [file-parse 接口说明](usage/file_parse.md)。
- **更清晰的硬件诊断与档位推荐**：新增 `mineru-kit hardware detect` 和 `hardware info`，帮助选择合适的服务档位，并查看设备、运行环境与校准信息。

**升级提示：** Python 支持范围调整为 **3.11–3.14**。使用旧 HTTP 接口的部署，请协调升级服务端、Router、SDK 和客户端。其他修复见[完整更新日志](reference/changelog.md)。

## 产品概览

MinerU 4.x 将多格式文档解析、文档库和服务工具整合到统一工作流，面向文档转换、应用集成和 Agent 阅读。

- **四档解析**：Flash 用于快速预览与索引，Basic 提供基础 OCR/模型解析，Standard 与 Advanced 面向更复杂的版面和更高的质量需求。
- **多格式输入**：PDF、图片，以及 DOC/DOCX、PPT/PPTX、XLS/XLSX、RTF、ODT/ODS/ODP、EPUB、OFD、HTML/MHTML、CSV/TSV。原生文档由 [DocVortex](https://github.com/myhloli/docvortex) 提供解析能力。
- **文档库与 Agent 阅读**：发现文件、缓存结果、搜索内容，按页或块继续阅读，并保留稳定引用位置。
- **独立模型配置**：小模型选择 ONNX 或 Torch；VLM 独立选择 llama.cpp、vLLM、LMDeploy。
- **统一工具入口**：Python SDK、V1 API、无状态批处理、多服务 Router 和基于 Gradio 的 WebUI。
- **结构化结果与渲染**：统一文档模型支持 Markdown、HTML、LaTeX、DOCX、EPUB、PDF、Structured Content、Content List V1/V2 九种渲染目标；各 CLI/API 的导出范围见[输出格式与结果协议](reference/output_files.md)。

PDF 和图片支持四档解析；Office、OpenDocument、EPUB、OFD、HTML/MHTML、CSV/TSV 使用本地 Flash 原生解析。纯文本直接读取，不进入解析流程。默认不会自动将文档上传到官网服务，远程解析需要显式配置。

## 开始使用

- [安装与快速入门](quick_start/index.md)
- [档位与运行环境](usage/tiers.md)
- [Python SDK](usage/sdk_api.md)
- [V1 HTTP API](usage/http_api.md)
- [3.x → 4.0 迁移](reference/migration_4.md)
- [旧平台适配（MinerU <4）](usage/compatibility.md)
- [更新历史](reference/changelog.md)

[GitHub](https://github.com/opendatalab/MinerU) · [MinerU](https://mineru.net/) · [License](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md)
