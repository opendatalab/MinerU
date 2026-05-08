# MinerU 模块功能总览

## 概览

本文档用于概括 `mineru` 包内各个主要模块的职责，帮助开发者快速定位代码。

目标不是把每个函数都讲一遍，而是说明每个模块组“负责什么”以及“遇到什么问题时应该先看哪里”。

## 顶层包结构

`mineru` 包大体按五类职责组织：

- `cli`：请求入口、API 服务、客户端工具、路由与运行时编排
- `backend`：文档解析后端与结果渲染
- `model`：模型封装与各类文档格式转换器
- `data`：存储、IO 抽象与路径/Schema 辅助
- `utils`：跨模块复用的底层工具函数

## 结构示意

```text
mineru/
├── cli/
├── backend/
│   ├── pipeline/
│   ├── vlm/
│   ├── hybrid/
│   ├── office/
│   └── utils/
├── model/
│   ├── docx/
│   ├── pptx/
│   ├── xlsx/
│   ├── layout/
│   ├── ocr/
│   ├── mfr/
│   ├── table/
│   ├── ori_cls/
│   └── vlm/
├── data/
│   ├── data_reader_writer/
│   ├── io/
│   └── utils/
└── utils/
```

## `mineru.cli`

这个包是 MinerU 的运行时入口层。

### 主要职责

- 暴露 FastAPI 接口
- 提供本地和远程客户端调用流程
- 标准化上传文件
- 将任务分发到不同解析后端
- 管理 worker 路由和本地 API 启动流程

### 重要模块

- `fast_api.py`
  - HTTP 服务主入口。
  - 负责校验请求、保存上传文件、构造解析任务并返回结果。
- `common.py`
  - CLI 与 API 共用的核心编排层。
  - 负责读取文件字节、按后端分发、写出最终结果文件。
- `api_client.py`
  - 调用 MinerU API 的客户端辅助模块。
  - 负责提交任务、轮询状态、下载 ZIP 结果、启动临时本地服务。
- `router.py`
  - 多 worker 路由层。
  - 负责任务分发和本地 worker 进程管理。
- `client.py`
  - 命令行客户端入口，封装基于 API 的解析调用。
- `gradio_app.py`
  - Gradio Web UI 入口。
- `vlm_server.py`
  - VLM 服务进程入口。
- `vlm_preload.py`
  - VLM worker 预加载逻辑，用于减少冷启动开销。
- `models_download.py`
  - 模型下载与安装相关 CLI 入口。
- `output_paths.py`
  - 输出目录和文件布局的规范化逻辑。
- `api_protocol.py`
  - 协议常量和版本元数据。
- `public_http_client_policy.py`
  - 面向公网绑定场景的安全限制策略。

## `mineru.backend`

这个包放的是实际的文档解析后端，以及最终输出渲染逻辑。

### `backend.pipeline`

传统 OCR + layout 解析链路。

#### 职责

- 执行 OCR 导向的文档分析
- 批量处理页面图像
- 把模型结果转换为内部 `middle_json`
- 渲染 Markdown 和结构化输出

#### 重要模块

- `pipeline_analyze.py`
  - pipeline 主执行流程。
- `batch_analyze.py`
  - 批量图像推理执行层。
- `model_init.py`
  - 初始化 pipeline 所需模型并做实例缓存。
- `model_json_to_middle_json.py`
  - 将原始模型输出转换成 MinerU 中间格式。
- `pipeline_middle_json_mkcontent.py`
  - 将 pipeline 的 `middle_json` 渲染成 Markdown 和 content list。
- `para_split.py`
  - 将识别结果切分成段落级结构。
- `pipeline_magic_model.py`、`model_list.py`
  - pipeline 推理相关的辅助结构和映射逻辑。

### `backend.vlm`

纯 VLM 的文档解析路径。

#### 职责

- 对 PDF 页面图像执行视觉语言模型抽取
- 将每页 VLM 输出合并为 `middle_json`
- 渲染 VLM 风格的输出结果

#### 重要模块

- `vlm_analyze.py`
  - VLM 后端主执行链路。
- `model_output_to_middle_json.py`
  - 将 VLM 输出转换为内部结构。
- `vlm_middle_json_mkcontent.py`
  - 将 VLM `middle_json` 渲染成 Markdown 和 content list。
- `vlm_magic_model.py`、`utils.py`
  - VLM 特有的辅助逻辑与块处理逻辑。

### `backend.hybrid`

融合 VLM + OCR + 公式补强的解析路径。

#### 职责

- 判断当前文档是否需要 OCR
- 把 VLM 的版面理解和 OCR 增强结合起来
- 补强公式和 OCR 困难区域
- 输出统一的 `middle_json`

#### 重要模块

- `hybrid_analyze.py`
  - hybrid 主执行流程，以及 OCR/VLM 决策逻辑。
- `hybrid_model_output_to_middle_json.py`
  - 将 hybrid 结果转换为内部格式。
- `hybrid_magic_model.py`
  - hybrid 相关的辅助结构。

### `backend.office`

用于 `docx`、`pptx`、`xlsx` 的 Office 文档解析链路。

#### 职责

- 不经过 PDF 布局推理，直接解析 Office 文档
- 将 office 解析结果转为 `middle_json`
- 输出 Markdown 和结构化结果

#### 重要模块

- `docx_analyze.py`
  - DOCX 解析入口。
- `pptx_analyze.py`
  - PPTX 解析入口。
- `xlsx_analyze.py`
  - XLSX 解析入口。
- `model_output_to_middle_json.py`
  - 将 office 解析结果转换为 `middle_json`。
- `office_middle_json_mkcontent.py`
  - 将 office 结果渲染成 Markdown 和 content list。
- `office_magic_model.py`
  - office 相关辅助结构。

### `backend.utils`

多个后端共用的辅助模块。

#### 职责

- Markdown 渲染辅助
- OCR 检测辅助
- 运行时计时和进度控制
- Office 图片/图表处理
- 段落块后处理

#### 重要模块

- `markdown_utils.py`
- `ocr_det_utils.py`
- `runtime_utils.py`
- `office_chart.py`
- `office_image.py`
- `html_image_utils.py`
- `para_block_utils.py`

## `mineru.model`

这个包主要放模型侧封装，以及不同文档格式的转换器。

它位于 `backend` 的下层。

### Office 转换器

- `model/docx/`
  - DOCX 解析与中间结构转换。
- `model/pptx/`
  - PPTX 解析、规范化和块排序辅助。
- `model/xlsx/`
  - XLSX 解析与工作簿结构转换。
- `office_stream.py`
  - Office 流式读取相关的共享辅助逻辑。

### 模型封装

- `model/layout/`
  - 版面分析模型封装，例如文档布局检测。
- `model/ocr/`
  - OCR 模型封装与 OCR 预处理辅助。
- `model/mfr/`
  - 公式识别支持。
- `model/table/`
  - 表格识别与表格解析支持。
- `model/ori_cls/`
  - 文档方向分类。
- `model/vlm/`
  - VLM 服务支持，例如 `vllm` 和 `lmdeploy` 的 server 适配。

### `model/utils`

模型侧共用工具和 OCR/推理辅助逻辑。

## `mineru.data`

这个包负责存储、IO 和路径/schema 抽象。

### `data_reader_writer`

不同存储目标上的读写抽象层。

#### 重要模块

- `base.py`
  - Reader/Writer 的基础接口。
- `filebase.py`
  - 本地文件系统实现。
- `s3.py`
  - S3 存储实现。
- `multi_bucket_s3.py`
  - 多 bucket S3 支持。
- `dummy.py`
  - 占位或空实现。

### `io`

面向传输层的 IO 辅助模块。

#### 重要模块

- `base.py`
- `http.py`
- `s3.py`

主要用于通过外部 IO 通道读取或传输数据。

### `data.utils`

路径、schema、异常类型等辅助模块。

#### 重要模块

- `path_utils.py`
- `schemas.py`
- `exceptions.py`

## `mineru.utils`

这个包是整个 repo 里跨层复用的通用工具层。

### 主要职责方向

- PDF 处理与页面/图像转换
- 文件后缀和语言猜测
- 推理引擎选择与环境配置
- OCR 和 bbox 工具
- 可视化与调试辅助
- 模型下载和系统环境检查

### 重要模块

- `pdf_image_tools.py`
  - PDF 页面转图像，以及图片输入转 PDF 字节。
- `pdfium_guard.py`
  - 安全封装 PDFium 的打开、关闭和页数读取。
- `pdf_classify.py`
  - 判断当前文档应走 OCR 还是文本抽取。
- `guess_suffix_or_lang.py`
  - 检测文件后缀或语言提示。
- `engine_utils.py`
  - 选择运行时推理引擎。
- `config_reader.py`
  - 从环境变量或配置源读取运行时配置。
- `draw_bbox.py`
  - 生成 layout/span 可视化 PDF。
- `ocr_utils.py`
  - OCR 结果处理和归一化辅助。
- `model_utils.py`
  - 裁剪、显存清理等模型侧共享辅助逻辑。
- `models_download_utils.py`
  - 模型下载与路径解析。
- `check_sys_env.py`
  - 运行环境兼容性检查。
- `enum_class.py`
  - 各解析层共享的枚举定义。
- `bbox_utils.py`、`boxbase.py`
  - bbox 操作辅助。
- `table_merge.py`
  - 跨页表格合并。
- `pdf_reader.py`、`pdf_text_tool.py`、`pdf_page_id.py`
  - 其他 PDF 读取与索引辅助能力。

## 建议阅读顺序

如果是第一次进入这个 repo，通常按下面顺序阅读最省时间：

1. `mineru/cli/fast_api.py`
2. `mineru/cli/common.py`
3. 选择一个后端入口：
   - `mineru/backend/hybrid/hybrid_analyze.py`
   - 或 `mineru/backend/pipeline/pipeline_analyze.py`
4. 对应的 `*_middle_json_mkcontent.py`
5. `mineru/utils/` 中的共用辅助模块

如果你只关心 Office 文档解析，建议从这里开始：

1. `mineru/cli/common.py`
2. `mineru/backend/office/docx_analyze.py`
3. `mineru/model/docx/main.py`

## 快速定位指南

- 想看请求和服务入口：`cli/`
- 想看文件如何被解析：`backend/`
- 想看模型封装或格式转换：`model/`
- 想改存储和 IO：`data/`
- 想找 PDF/OCR/通用辅助逻辑：`utils/`
