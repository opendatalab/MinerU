# 命令行参数进阶

## 使用外部 VLM 增强视觉详情

MinerU 可以选择性地调用外部 OpenAI-compatible VLM endpoint，为已有的 `image` 和 `chart` 详情追加教学式解释。该功能主要面向 RAG 场景：MinerU 提取的视觉数据仍作为基础内容，外部 VLM 只补充更利于检索和问答的语义解释。

该功能默认关闭。启用后，MinerU 仍负责主要的文档解析、版面识别、图片裁剪、表格提取、视觉 `sub_type` 分类以及基础视觉内容生成。外部 VLM 只会处理已经存在 MinerU `content` 的 `image` / `chart` 块，并将结果追加为 `### Didactic interpretation` 小节。

示例：

```bash
mineru \
  -p input.pdf \
  -o output \
  -b vlm-auto-engine \
  --details-image-analysis true \
  --details-vlm-url http://127.0.0.1:11434/v1 \
  --details-vlm-model qwen-vl-model \
  --details-vlm-timeout 180 \
  --details-vlm-max-concurrency 2 \
  --details-vlm-language zh
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--details-image-analysis` | `false` | 启用外部 VLM 对已引用的 `image` / `chart` 详情进行增强。 |
| `--details-vlm-url` | 未设置 | 外部 VLM 的 OpenAI-compatible base URL。启用增强时必填。 |
| `--details-vlm-model` | 未设置 | 发送给外部 VLM endpoint 的模型名称。启用增强时必填。 |
| `--details-vlm-api-key` | 空 | 可选 API key，会作为 bearer token 发送给外部 VLM endpoint。 |
| `--details-vlm-timeout` | `120` | 单张图片请求超时时间，单位为秒。 |
| `--details-vlm-max-concurrency` | `1` | 外部 VLM 并发请求数上限。请根据 endpoint 能力和调用成本谨慎调高。 |
| `--details-vlm-language` | `auto` | 追加解释的输出语言。使用 `auto` 时会尽量根据文档语言自动判断。 |

> [!IMPORTANT]
> 配置的外部 VLM endpoint 可能会接收裁剪后的视觉图片、caption、footnote 以及附近正文上下文。请仅在符合隐私和数据治理要求的 endpoint 上启用该功能。

## 推理引擎参数透传

### 参数传递说明
> [!TIP]
> - 所有vllm/lmdeploy官方支持的参数都可用通过命令行参数传递给 MinerU，包括以下命令:`mineru`、`mineru-openai-server`、`mineru-gradio`、`mineru-api`、`mineru-router`
> - 命令行参数同时支持 `--foo value` 与 `--foo=value` 两种写法
> - 如果您想了解更多有关`vllm`的参数使用方法，请参考 [vllm官方文档](https://docs.vllm.ai/en/latest/cli/serve.html)
> - 如果您想了解更多有关`lmdeploy`的参数使用方法，请参考 [lmdeploy官方文档](https://lmdeploy.readthedocs.io/en/latest/llm/api_server.html)

## GPU 设备选择与配置

### CUDA_VISIBLE_DEVICES 基本用法
> [!TIP]
> - 任何情况下，您都可以通过在命令行的开头添加`CUDA_VISIBLE_DEVICES` 环境变量来指定可见的 GPU 设备：
>   ```bash
>   CUDA_VISIBLE_DEVICES=1 mineru -p <input_path> -o <output_path>
>   ```
> - 这种指定方式对所有的命令行调用都有效，包括 `mineru`、`mineru-openai-server`、`mineru-gradio`、`mineru-api`和`mineru-router`，且对`pipeline`、`vlm`后端均适用。

### 常见设备配置示例
> [!TIP]
> 以下是一些常见的 `CUDA_VISIBLE_DEVICES` 设置示例：
>   ```bash
>   CUDA_VISIBLE_DEVICES=1  # Only device 1 will be seen
>   CUDA_VISIBLE_DEVICES=0,1  # Devices 0 and 1 will be visible
>   CUDA_VISIBLE_DEVICES="0,1"  # Same as above, quotation marks are optional
>   CUDA_VISIBLE_DEVICES=0,2,3  # Devices 0, 2, 3 will be visible; device 1 is masked
>   CUDA_VISIBLE_DEVICES=""  # No GPU will be visible
>   ```

## 实际应用场景

> [!TIP]
> 以下是一些可能的使用场景：
> 
> - 如果您有多张显卡，需要在卡0和卡1上启动两个`openai-server`服务，并分别监听不同的端口，可以使用以下命令： 
>   ```bash
>   # 在终端1中
>   CUDA_VISIBLE_DEVICES=0 mineru-openai-server --engine vllm --port 30000
>   # 在终端2中
>   CUDA_VISIBLE_DEVICES=1 mineru-openai-server --engine vllm --port 30001
>   ```
>   
> - 如果您有多张显卡，需要在卡0和卡1上启动两个`fastapi`服务，并分别监听不同的端口，可以使用以下命令： 
>   ```bash
>   # 在终端1中
>   CUDA_VISIBLE_DEVICES=0 mineru-api --host 127.0.0.1 --port 8000
>   # 在终端2中
>   CUDA_VISIBLE_DEVICES=1 mineru-api --host 127.0.0.1 --port 8001
>   ```
>   
> - 如果您有多张显卡，需要通过`router`在其中4张卡上启动`fastapi`服务并统一管理，可以使用以下命令： 
>   ```bash
>   CUDA_VISIBLE_DEVICES=0,1,2,3 mineru-router --host 127.0.0.1 --port 8002
>   ```
