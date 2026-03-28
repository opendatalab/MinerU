# 命令行参数进阶

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
