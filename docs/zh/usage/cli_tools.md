# 命令行工具使用说明

## 查看帮助信息
要查看 MinerU 命令行工具的帮助信息，可以使用 `--help` 参数。以下是各个命令行工具的帮助信息示例：
```bash
mineru --help
Usage: mineru [OPTIONS]

Options:
  -v, --version                   显示版本并退出
  -p, --path PATH                 输入文件路径或目录（必填）
  -o, --output PATH               输出目录（必填）
  -m, --method [auto|txt|ocr]     解析方法：auto（默认）、txt、ocr（仅用于 pipeline 与 hybrid* 后端）
  -b, --backend [pipeline|hybrid-auto-engine|hybrid-http-client|vlm-auto-engine|vlm-http-client]
                                  解析后端（默认为 hybrid-auto-engine）
  -l, --lang [ch|ch_server|ch_lite|en|korean|japan|chinese_cht|ta|te|ka|th|el|latin|arabic|east_slavic|cyrillic|devanagari]
                                  指定文档语言（可提升 OCR 准确率，仅用于 pipeline 与 hybrid* 后端）
  -u, --url TEXT                  当使用 http-client 时，需指定服务地址
  -s, --start INTEGER             开始解析的页码（从 0 开始）
  -e, --end INTEGER               结束解析的页码（从 0 开始）
  -f, --formula BOOLEAN           是否启用公式解析（默认开启）
  -t, --table BOOLEAN             是否启用表格解析（默认开启）
  -d, --device TEXT               推理设备（如 cpu/cuda/cuda:0/npu/mps，仅 pipeline 后端）
  --vram INTEGER                  单进程最大 GPU 显存占用(GB)（仅 pipeline 后端）
  --source [huggingface|modelscope|local]
                                  模型来源，默认 huggingface
  --help                          显示帮助信息
```
```bash
mineru-api --help
Usage: mineru-api [OPTIONS]

Options:
  --host TEXT     服务器主机地址（默认：127.0.0.1）
  --port INTEGER  服务器端口（默认：8000）
  --reload        启用自动重载（开发模式）
  --help          显示此帮助信息并退出
```
```bash
mineru-gradio --help
Usage: mineru-gradio [OPTIONS]

Options:
  --enable-example BOOLEAN        启用示例文件输入(需要将示例文件放置在当前
                                  执行命令目录下的 `example` 文件夹中)
  --enable-http-client BOOLEAN    在后端选项中启用 HTTP 客户端选项
  --enable-api BOOLEAN            启用 Gradio API 以提供应用程序服务
  --max-convert-pages INTEGER     设置从 PDF 转换为 Markdown 的最大页数
  --server-name TEXT              设置 Gradio 应用程序的服务器主机名
  --server-port INTEGER           设置 Gradio 应用程序的服务器端口
  --latex-delimiters-type [a|b|all]
                                  设置在 Markdown 渲染中使用的 LaTeX 分隔符类型
                                  ('a' 表示 '$' 类型，'b' 表示 '()[]' 类型，
                                  'all' 表示两种类型都使用)
  --help                          显示此帮助信息并退出
```

## 环境变量说明

MinerU命令行工具的某些参数存在相同功能的环境变量配置，通常环境变量配置的优先级高于命令行参数，且在所有命令行工具中都生效。
以下是常用的环境变量及其说明： 

- `MINERU_DEVICE_MODE`：
    * 用于指定推理设备
    * 支持`cpu/cuda/cuda:0/npu/mps`等设备类型
    * 仅对`pipeline`后端生效。
  
- `MINERU_VIRTUAL_VRAM_SIZE`：
    * 用于指定单进程最大 GPU 显存占用(GB)
    * 仅对`pipeline`后端生效。
  
- `MINERU_MODEL_SOURCE`：
    * 用于指定模型来源
    * 支持`huggingface/modelscope/local`
    * 默认为`huggingface`可通过环境变量切换为`modelscope`或使用本地模型。
  
- `MINERU_TOOLS_CONFIG_JSON`：
    * 用于指定配置文件路径
    * 默认为用户目录下的`mineru.json`，可通过环境变量指定其他配置文件路径。
  
- `MINERU_FORMULA_ENABLE`：
    * 用于启用公式解析
    * 默认为`true`，可通过环境变量设置为`false`来禁用公式解析。

- `MINERU_FORMULA_CH_SUPPORT`：
    * 用于启用中文公式解析优化（实验性功能）
    * 默认为`false`，可通过环境变量设置为`true`来启用中文公式解析优化。
    * 仅对`pipeline`后端生效。
  
- `MINERU_TABLE_ENABLE`：
    * 用于启用表格解析
    * 默认为`true`，可通过环境变量设置为`false`来禁用表格解析。

- `MINERU_TABLE_MERGE_ENABLE`：
    * 用于启用表格合并功能
    * 默认为`true`，可通过环境变量设置为`false`来禁用表格合并功能。

- `MINERU_PDF_RENDER_TIMEOUT`：
    * 用于设置将PDF渲染为图片的超时时间（秒）
    * 默认为`300`秒，可通过环境变量设置为其他值以调整渲染图片的超时时间。

- `MINERU_INTRA_OP_NUM_THREADS`：
    * 用于设置onnx模型的intra_op线程数，影响单个算子的计算速度
    * 默认为`-1`（自动选择），可通过环境变量设置为其他值以调整线程数。

- `MINERU_INTER_OP_NUM_THREADS`：
    * 用于设置onnx模型的inter_op线程数，影响多个算子的并行执行
    * 默认为`-1`（自动选择），可通过环境变量设置为其他值以调整线程数。

- `MINERU_HYBRID_BATCH_RATIO`：
    * 用于设置 hybrid-* 后端中 小模型处理的batch倍率
    * 在hybrid-http-client中较为常用，可以通过控制小模型的batch倍率来调整单个客户端的显存占用量
    * 单个client端显存大小 | MINERU_HYBRID_BATCH_RATIO
      ------------------|------------------------
      <= 6   GB         | 8
      <= 4.5 GB         | 4
      <= 3   GB         | 2
      <= 2.5 GB         | 1

- `MINERU_HYBRID_FORCE_PIPELINE_ENABLE`：
    * 用于强制将 hybrid-* 后端中的 文本提取部分使用 小模型 进行处理
    * 默认为`false`，可通过环境变量设置为`true`来启用该功能，从而在某些极端情况下减少幻觉的发生。

- `MINERU_VL_MODEL_NAME`：
    * 用于指定 vlm/hybrid 后端使用的模型名称，这将允许您在同时存在多个模型的远程openai-server中指定 MinerU 运行所需的模型。

- `MINERU_VL_API_KEY`:
    * 用于指定 vlm/hybrid 后端使用的API Key，这将允许您在远程openai-server中进行身份验证。