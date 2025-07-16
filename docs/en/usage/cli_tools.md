# Command Line Tools Usage Instructions

## View Help Information
To view help information for MinerU command line tools, you can use the `--help` parameter. Here are help information examples for various command line tools:
```bash
mineru --help
Usage: mineru [OPTIONS]

Options:
  -v, --version                   Show version and exit
  -p, --path PATH                 Input file path or directory (required)
  -o, --output PATH               Output directory (required)
  -m, --method [auto|txt|ocr]     Parsing method: auto (default), txt, ocr (pipeline backend only)
  -b, --backend [pipeline|vlm-transformers|vlm-sglang-engine|vlm-sglang-client]
                                  Parsing backend (default: pipeline)
  -l, --lang [ch|ch_server|ch_lite|en|korean|japan|chinese_cht|ta|te|ka|latin|arabic|east_slavic|cyrillic|devanagari]
                                  Specify document language (improves OCR accuracy, pipeline backend only)
  -u, --url TEXT                  Service address when using sglang-client
  -s, --start INTEGER             Starting page number for parsing (0-based)
  -e, --end INTEGER               Ending page number for parsing (0-based)
  -f, --formula BOOLEAN           Enable formula parsing (default: enabled)
  -t, --table BOOLEAN             Enable table parsing (default: enabled)
  -d, --device TEXT               Inference device (e.g., cpu/cuda/cuda:0/npu/mps, pipeline backend only)
  --vram INTEGER                  Maximum GPU VRAM usage per process (GB) (pipeline backend only)
  --source [huggingface|modelscope|local]
                                  Model source, default: huggingface
  --help                          Show help information
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
  --enable-example BOOLEAN        Enable example files for input. The example
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
                                  Markdown rendering: 'a' for type '$', 'b' for
                                  type '()[]', 'all' for both types.
  --help                          Show this message and exit.
```

## Environment Variables Description

Some parameters of MinerU command line tools have equivalent environment variable configurations. Generally, environment variable configurations have higher priority than command line parameters and take effect across all command line tools.
Here are the environment variables and their descriptions:

- `MINERU_DEVICE_MODE`: Used to specify inference device, supports device types like `cpu/cuda/cuda:0/npu/mps`, only effective for `pipeline` backend.
- `MINERU_VIRTUAL_VRAM_SIZE`: Used to specify maximum GPU VRAM usage per process (GB), only effective for `pipeline` backend.
- `MINERU_MODEL_SOURCE`: Used to specify model source, supports `huggingface/modelscope/local`, defaults to `huggingface`, can be switched to `modelscope` or local models through environment variables.
- `MINERU_TOOLS_CONFIG_JSON`: Used to specify configuration file path, defaults to `mineru.json` in user directory, can specify other configuration file paths through environment variables.
- `MINERU_FORMULA_ENABLE`: Used to enable formula parsing, defaults to `true`, can be set to `false` through environment variables to disable formula parsing.
- `MINERU_TABLE_ENABLE`: Used to enable table parsing, defaults to `true`, can be set to `false` through environment variables to disable table parsing.
