# Using MinerU

## Command Line Usage

### Basic Usage

The simplest command line invocation is:

```bash
mineru -p <input_path> -o <output_path>
```

- `<input_path>`: Local PDF/Image file or directory (supports pdf/png/jpg/jpeg/webp/gif)
- `<output_path>`: Output directory

### View Help Information

Get all available parameter descriptions:

```bash
mineru --help
```

### Parameter Details

```text
Usage: mineru [OPTIONS]

Options:
  -v, --version                   Show version and exit
  -p, --path PATH                 Input file path or directory (required)
  -o, --output PATH              Output directory (required)
  -m, --method [auto|txt|ocr]     Parsing method: auto (default), txt, ocr (pipeline backend only)
  -b, --backend [pipeline|vlm-transformers|vlm-sglang-engine|vlm-sglang-client]
                                  Parsing backend (default: pipeline)
  -l, --lang [ch|ch_server|ch_lite|en|korean|japan|chinese_cht|ta|te|ka|latin|arabic|east_slavic|cyrillic|devanagari]
                                  Specify document language (improves OCR accuracy, pipeline backend only)
  -u, --url TEXT                  Service address when using sglang-client
  -s, --start INTEGER             Starting page number (0-based)
  -e, --end INTEGER               Ending page number (0-based)
  -f, --formula BOOLEAN           Enable formula parsing (default: on)
  -t, --table BOOLEAN             Enable table parsing (default: on)
  -d, --device TEXT               Inference device (e.g., cpu/cuda/cuda:0/npu/mps, pipeline backend only)
  --vram INTEGER                  Maximum GPU VRAM usage per process (GB)(pipeline backend only)
  --source [huggingface|modelscope|local]
                                  Model source, default: huggingface
  --help                          Show help information
```

---

## Model Source Configuration

MinerU automatically downloads required models from HuggingFace on first run. If HuggingFace is inaccessible, you can switch model sources:

### Switch to ModelScope Source

```bash
mineru -p <input_path> -o <output_path> --source modelscope
```

Or set environment variable:

```bash
export MINERU_MODEL_SOURCE=modelscope
mineru -p <input_path> -o <output_path>
```

### Using Local Models

#### 1. Download Models Locally

```bash
mineru-models-download --help
```

Or use interactive command-line tool to select models:

```bash
mineru-models-download
```

After download, model paths will be displayed in current terminal and automatically written to `mineru.json` in user directory.

#### 2. Parse Using Local Models

```bash
mineru -p <input_path> -o <output_path> --source local
```

Or enable via environment variable:

```bash
export MINERU_MODEL_SOURCE=local
mineru -p <input_path> -o <output_path>
```

---

## Using sglang to Accelerate VLM Model Inference

### Through the sglang-engine Mode

```bash
mineru -p <input_path> -o <output_path> -b vlm-sglang-engine
```

### Through the sglang-server/client Mode

1. Start Server:

```bash
mineru-sglang-server --port 30000
```

2. Use Client in another terminal:

```bash
mineru -p <input_path> -o <output_path> -b vlm-sglang-client -u http://127.0.0.1:30000
```

> [!TIP]
> For more information about output files, please refer to [Output File Documentation](../output_file.md)

---