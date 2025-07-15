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
  -m, --method [auto|txt|ocr]     解析方法：auto（默认）、txt、ocr（仅用于 pipeline 后端）
  -b, --backend [pipeline|vlm-transformers|vlm-sglang-engine|vlm-sglang-client]
                                  解析后端（默认为 pipeline）
  -l, --lang [ch|ch_server|ch_lite|en|korean|japan|chinese_cht|ta|te|ka|latin|arabic|east_slavic|cyrillic|devanagari]
                                  指定文档语言（可提升 OCR 准确率，仅用于 pipeline 后端）
  -u, --url TEXT                  当使用 sglang-client 时，需指定服务地址
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
  --host TEXT     Server host (default: 127.0.0.1)
  --port INTEGER  Server port (default: 8000)
  --reload        Enable auto-reload (development mode)
  --help          Show this message and exit.
```
```bash
mineru-gradio --help
Usage: mineru-gradio [OPTIONS]

Options:
  --enable-example BOOLEAN        Enable example files for input.The example
                                  files to be input need to be placed in the
                                  `example` folder within the directory where
                                  the command is currently executed.
  --enable-sglang-engine BOOLEAN  Enable SgLang engine backend for faster
                                  processing.
  --enable-api BOOLEAN            Enable gradio API for serving the
                                  application.
  --max-convert-pages INTEGER     Set the maximum number of pages to convert
                                  from PDF to Markdown.
  --server-name TEXT              Set the server name for the Gradio app.
  --server-port INTEGER           Set the server port for the Gradio app.
  --latex-delimiters-type [a|b|all]
                                  Set the type of LaTeX delimiters to use in
                                  Markdown rendering:'a' for type '$', 'b' for
                                  type '()[]', 'all' for both types.
  --help                          Show this message and exit.
```

## 环境变量说明

MinerU命令行工具的某些参数存在相同功能的环境变量配置，通常环境变量配置的优先级高于命令行参数，且在所有命令行工具中都生效。
以下是常用的环境变量及其说明： 

- `MINERU_DEVICE_MODE`：用于指定推理设备，支持`cpu/cuda/cuda:0/npu/mps`等设备类型，仅对`pipeline`后端生效。
- `MINERU_VIRTUAL_VRAM_SIZE`：用于指定单进程最大 GPU 显存占用(GB)，仅对`pipeline`后端生效。
- `MINERU_MODEL_SOURCE`：用于指定模型来源，支持`huggingface/modelscope/local`，默认为`huggingface`，可通过环境变量切换为`modelscope`或使用本地模型。
- `MINERU_TOOLS_CONFIG_JSON`：用于指定配置文件路径，默认为用户目录下的`mineru.json`，可通过环境变量指定其他配置文件路径。
- `MINERU_FORMULA_ENABLE`：用于启用公式解析，默认为`true`，可通过环境变量设置为`false`来禁用公式解析。
- `MINERU_TABLE_ENABLE`：用于启用表格解析，默认为`true`，可通过环境变量设置为`false`来禁用表格解析。


