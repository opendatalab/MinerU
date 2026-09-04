# 使用 MinerU

## 快速配置模型源
MinerU默认使用`huggingface`作为模型源，若用户网络无法访问`huggingface`，可以通过环境变量便捷地切换模型源为`modelscope`：
```bash
export MINERU_MODEL_SOURCE=modelscope
```
有关模型源配置和自定义本地模型路径的更多信息，请参考文档中的[模型源说明](./model_source.md)。

## 通过命令行快速使用
MinerU内置了命令行工具，用户可以通过命令行快速使用MinerU进行文档解析：
```bash
mineru parse <input_path> --pages all -o <output_path>
```
> [!TIP]
> - `<input_path>`：单个本地 `PDF` / `OFD` / `EPUB` / 静态 `HTML` / 图片 / `CSV` / `RTF` / `DOC`/`DOCX` / `PPT`/`PPTX` / `XLS`/`XLSX` / `ODT`/`ODS`/`ODP` 文件
> - `<output_path>`：可选输出文件；未指定时 Markdown 写入标准输出
> - PDF 默认解析前 10 页；使用 `--pages all` 解析整份文档
> 
> 更多关于输出文件的信息，请参考[输出文件说明](../reference/output_files.md)。

> [!NOTE]
> 命令行工具会在Linux和macOS系统自动尝试cuda/mps加速。Windows用户如需使用cuda加速，
> 请前往 [Pytorch官网](https://pytorch.org/get-started/locally/) 选择适合自己cuda版本的命令安装支持加速的`torch`和`torchvision`。

如果需要通过自定义参数调整解析选项，您也可以在文档中查看更详细的[命令行工具使用说明](./cli_tools.md)。

## 通过 API、WebUI 和服务进阶使用

- 启动自部署 V1 API：
  ```bash
  mineru-kit api-server --host 0.0.0.0 --port 8000 --tier standard
  ```
  >[!TIP]
  >在浏览器中访问 `http://127.0.0.1:8000/docs` 查看 OpenAPI 文档。服务只提供 `/v1/*` 接口，包括健康检查、能力发现、上传、文件、解析任务和用量查询。
  >
  >http异步调用代码示例：[Python版本](https://github.com/opendatalab/MinerU/blob/master/demo/demo.py)

- 启动gradio webui 可视化前端：
  ```bash
  mineru-kit gradio --server-name 0.0.0.0 --server-port 7860
  ```
  >[!TIP]
  > 
  >- 在浏览器中访问 `http://127.0.0.1:7860` 使用 Gradio WebUI。
  >- 未传 `--api-url` 时，Gradio 会托管 loopback `mineru-kit api-server`；传入后只连接指定的 V1 服务。
  >- 使用 `--api-server-preload-models` 为托管的本地服务预加载模型。
  >- `mineru-gradio` 仍作为命令名兼容别名，使用相同的新版参数。

- 通过 `mineru-router` 进行多服务 / 多 GPU 编排：
  ```bash
  mineru-router --host 0.0.0.0 --port 8002 --local-gpus auto --worker-tier standard
  ```
  >[!TIP]
  >
  >- `mineru-router` 与 `mineru-kit router` 都只暴露完整 `/v1/*` API。
  >- 可重复使用 `--upstream-url` 聚合多个 V1 api-server，也可通过 `--local-gpus` 自动拉起 `mineru-kit api-server` worker。
  >- `--preload-models` 只作用于 Router 托管的本地 worker，远端 upstream 保持自己的启动配置。
  >- Router 不再透传未知模型引擎参数。
  >- 适用于多服务、多 GPU 和统一入口部署场景。

- 启动 OpenAI 兼容 VLM 服务：
  ```bash
  mineru-kit vlm-server --engine auto --port 30000
  ```

> [!NOTE]
> 模型引擎参数只适用于显式声明它们的命令；`mineru-router` 仅接受文档列出的 Router/worker 参数，不透传未知参数。
> 我们整理了一些`vllm/lmdeploy`使用中的常用参数和使用方法，可以在文档[命令行进阶参数](./advanced_cli_parameters.md)中获取。

## 使用 config.yaml 配置 LLM 辅助后处理

LLM 辅助标题分级和跨页表格单元格续接读取 `$MINERU_HOME/config.yaml`，兼容 OpenAI 协议的模型服务：

```yaml
llm_aided:
  api_key: ${MINERU_LLM_API_KEY:-}
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  model: qwen3.5-plus
  enable_thinking: false
  max_concurrency: 16
  features:
    title_leveling: false
    cross_page_table_cell_merge: false
```

- `title_leveling`：仅在 `MiddleJson.is_full_document=true` 的整本 PDF 输入中，以文档标题为边界分组优化 2～6 级段落标题；抽页结果持久化为 `false` 并跳过该功能。
- `cross_page_table_cell_merge`：在现有规则确认跨页续表后，通过 LLM 判断边界行中各组相邻单元格是否续接。
- table cell merge 不要求整本输入；两个功能默认关闭，并通过同一个异步客户端共享连接参数和 `max_concurrency` 请求上限，默认值为 16。
- `max_concurrency` 必须是不小于 1 的整数，可通过 `MINERU_LLM_AIDED_MAX_CONCURRENCY` 覆盖。
- 启用任一功能前必须配置非空的 `api_key`、`base_url` 和 `model`。
- `enable_thinking` 可省略；省略后不会向模型服务发送该扩展参数。
- 旧 `mineru.json` 中的 `llm-aided-config` 不再读取。

## 基于配置文件扩展 MinerU 功能

MinerU 可开箱即用，并从 `$MINERU_HOME/config.yaml` 读取当前配置；可通过 `MINERU_CONFIG` 指定其他配置文件。旧 `mineru.json` CLI 配置不再支持，Gradio 的 LaTeX 分隔符通过 `--latex-delimiters-type` 选择。

模型目录和模型源使用 `model` 配置段：

```yaml
model:
  base_dir: ~/.mineru/models
  source: auto
  stack: auto
```

模型下载和本地模型源的详细说明见[模型源说明](./model_source.md)。
