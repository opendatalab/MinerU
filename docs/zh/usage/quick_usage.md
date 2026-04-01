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
mineru -p <input_path> -o <output_path>
```
> [!TIP]
> - `<input_path>`：本地 `PDF` / 图片 / `DOCX` 文件或目录
> - `<output_path>`：输出目录
> - 未传 `--api-url` 时，CLI 会自动拉起本地临时 `mineru-api`
> - 传入 `--api-url` 时，CLI 会直连远端或已有本地 FastAPI 服务
> 
> 更多关于输出文件的信息，请参考[输出文件说明](../reference/output_files.md)。

> [!NOTE]
> 命令行工具会在Linux和macOS系统自动尝试cuda/mps加速。Windows用户如需使用cuda加速，
> 请前往 [Pytorch官网](https://pytorch.org/get-started/locally/) 选择适合自己cuda版本的命令安装支持加速的`torch`和`torchvision`。

如果需要通过自定义参数调整解析选项，您也可以在文档中查看更详细的[命令行工具使用说明](./cli_tools.md)。

## 通过api、webui、http-client/server进阶使用

- 通过fast api方式调用：
  ```bash
  mineru-api --host 0.0.0.0 --port 8000
  ```
  >[!TIP]
  >在浏览器中访问 `http://127.0.0.1:8000/docs` 查看API文档。
  >
  >- 健康检查接口：`GET /health`
  >  返回 `protocol_version`、`processing_window_size`、`max_concurrent_requests` 等服务信息
  >- 异步任务提交接口：`POST /tasks`
  >- 同步解析接口：`POST /file_parse`
  >- 任务查询接口：`GET /tasks/{task_id}`、`GET /tasks/{task_id}/result`
  >- API 输出目录由服务端固定控制，默认写入 `./output`
  >- 上传文件当前支持 `PDF`、图片与 `DOCX`
  >
  >- `POST /tasks` 会立即返回 `task_id`；`POST /file_parse` 会在内部提交到同一个任务管理器，等待任务完成后同步返回最终结果。
  >- 当任务处于排队状态时，任务提交结果和状态查询结果中可能会返回 `queued_ahead` 字段，用于表示前方排队任务数。
  >- 任务为单进程、进程内状态实现，服务重启、`--reload` 热重载或多进程部署后不保证仍可查询历史任务状态。
  >- 默认任务完成或失败后保留 24 小时，随后自动清理任务状态和输出目录；清理后访问任务状态或结果会返回 `404`。
  >- 可通过环境变量 `MINERU_API_TASK_RETENTION_SECONDS` 和 `MINERU_API_TASK_CLEANUP_INTERVAL_SECONDS` 调整保留时长与清理轮询间隔。
  >- 可通过 `--enable-vlm-preload true` 在服务启动阶段预热本地 VLM 模型，避免首次 VLM 或 hybrid 请求时再初始化。
  >
  >异步任务提交示例：
  >```bash
  >curl -X POST http://127.0.0.1:8000/tasks \
  >  -F "files=@demo/pdfs/demo1.pdf" \
  >  -F "return_md=true"
  >```
  >
  >同步解析示例：
  >```bash
  >curl -X POST http://127.0.0.1:8000/file_parse \
  >  -F "files=@demo/pdfs/demo1.pdf" \
  >  -F "return_md=true" \
  >  -F "response_format_zip=true" \
  >  -F "return_original_file=true"
  >```
  >
  >轮询任务状态与结果：
  >```bash
  >curl http://127.0.0.1:8000/tasks/<task_id>
  >curl http://127.0.0.1:8000/tasks/<task_id>/result
  >curl http://127.0.0.1:8000/health
  >```
  >
  >http异步调用代码示例：[Python版本](https://github.com/opendatalab/MinerU/blob/master/demo/demo.py)

- 启动gradio webui 可视化前端：
  ```bash
  mineru-gradio --server-name 0.0.0.0 --server-port 7860
  ```
  >[!TIP]
  > 
  >- 在浏览器中访问 `http://127.0.0.1:7860` 使用 Gradio WebUI。
  >- 未传 `--api-url` 时，Gradio 会自动拉起可复用的本地 `mineru-api`；传入 `--api-url` 时则会复用已有本地或远端服务。
  >- `--enable-vlm-preload true` 会让 Gradio 在 WebUI 启动阶段主动拉起本地 `mineru-api` 并等待 VLM 预加载完成；传入 `--api-url` 时会被忽略。
  >- WebUI 当前支持上传 `PDF`、图片与 `DOCX` 文件。

- 通过 `mineru-router` 进行多服务 / 多 GPU 编排：
  ```bash
  mineru-router --host 0.0.0.0 --port 8002 --local-gpus auto
  ```
  >[!TIP]
  >
  >- `mineru-router` 对外暴露与 `mineru-api` 一致的 `/health`、`/tasks`、`/file_parse`、`/tasks/{task_id}`、`/tasks/{task_id}/result` 接口。
  >- 可重复使用 `--upstream-url` 聚合多个已有 `mineru-api` 服务，也可通过 `--local-gpus` 自动拉起本地 worker。
  >- `--enable-vlm-preload true` 仅作用于 router 托管的本地 worker，不会影响通过 `--upstream-url` 接入的远端服务。
  >- 适用于多服务、多 GPU 和统一入口部署场景。

- 使用`http-client/server`方式调用：
  ```bash
  # 启动openai兼容服务器(需要安装vllm或lmdeploy环境)
  mineru-openai-server --port 30000
  ``` 
  >[!TIP]
  >在另一个终端中通过http client连接openai server
  > ```bash
  > mineru -p <input_path> -o <output_path> -b hybrid-http-client -u http://127.0.0.1:30000
  > ```
  >`vlm-http-client` 是轻量远程 client，用法上不要求本地安装 `torch`。
  >`hybrid-http-client` 需要本地具备 `mineru[pipeline]` 及 `torch` 等 pipeline 依赖。

> [!NOTE]
> 所有`vllm/lmdeploy`官方支持的参数都可用通过命令行参数传递给 MinerU，包括以下命令:`mineru`、`mineru-openai-server`、`mineru-gradio`、`mineru-api`、`mineru-router`，
> 我们整理了一些`vllm/lmdeploy`使用中的常用参数和使用方法，可以在文档[命令行进阶参数](./advanced_cli_parameters.md)中获取。

## 基于配置文件扩展 MinerU 功能

MinerU 现已实现开箱即用，但也支持通过配置文件扩展功能。您可通过编辑用户目录下的 `mineru.json` 文件，添加自定义配置。

>[!IMPORTANT]
>`mineru.json` 文件会在您使用内置模型下载命令 `mineru-models-download` 时自动生成，也可以通过将[配置模板文件](https://github.com/opendatalab/MinerU/blob/master/mineru.template.json)复制到用户目录下并重命名为 `mineru.json` 来创建。  

以下是一些可用的配置选项： 

- `latex-delimiter-config`：
    * 用于配置 LaTeX 公式的分隔符
    * 默认为`$`符号，可根据需要修改为其他符号或字符串。
  
- `llm-aided-config`：
    * 用于配置 LLM 辅助标题分级的相关参数，兼容所有支持`openai协议`的 LLM 模型
    * 默认使用`阿里云百炼`的`qwen3-next-80b-a3b-instruct`模型
    * 您需要自行配置 API 密钥并将`enable`设置为`true`来启用此功能
    * 如果您的api供应商不支持`enable_thinking`参数，请手动将该参数删除
        * 例如，在您的配置文件中，`llm-aided-config` 部分可能如下所示：
          ```json
          "llm-aided-config": {
             "api_key": "your_api_key",
             "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
             "model": "qwen3-next-80b-a3b-instruct",
             "enable_thinking": false,
             "enable": false
          }
          ```
        * 要移除`enable_thinking`参数，只需删除包含`"enable_thinking": false`的那一行，结果如下:
          ```json
          "llm-aided-config": {
             "api_key": "your_api_key",
             "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
             "model": "qwen3-next-80b-a3b-instruct",
             "enable": false
          }
          ```
  
- `models-dir`：
    * 用于指定本地模型存储目录，请为`pipeline`和`vlm`后端分别指定模型目录，
    * 指定目录后您可通过配置环境变量`export MINERU_MODEL_SOURCE=local`来使用本地模型。
