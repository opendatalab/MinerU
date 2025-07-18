# 使用 MinerU

## 快速配置模型源
MinerU默认使用`huggingface`作为模型源，若用户网络无法访问`huggingface`，可以通过环境变量便捷地切换模型源为`modelscope`：
```bash
export MINERU_MODEL_SOURCE=modelscope
```
有关模型源配置和自定义本地模型路径的更多信息，请参考文档中的[模型源说明](./model_source.md)。

## 通过命令行快速使用
MinerU内置了命令行工具，用户可以通过命令行快速使用MinerU进行PDF解析：
```bash
# 默认使用pipeline后端解析
mineru -p <input_path> -o <output_path>
```
> [!TIP]
> - `<input_path>`：本地 PDF/图片 文件或目录
> - `<output_path>`：输出目录
> 
> 更多关于输出文件的信息，请参考[输出文件说明](../reference/output_files.md)。

> [!NOTE]
> 命令行工具会在Linux和macOS系统自动尝试cuda/mps加速。Windows用户如需使用cuda加速，
> 请前往 [Pytorch官网](https://pytorch.org/get-started/locally/) 选择适合自己cuda版本的命令安装支持加速的`torch`和`torchvision`。

```bash
# 或指定vlm后端解析
mineru -p <input_path> -o <output_path> -b vlm-transformers
```
> [!TIP]
> vlm后端另外支持`sglang`加速，与`transformers`后端相比，`sglang`的加速比可达20～30倍，可以在[扩展模块安装指南](../quick_start/extension_modules.md)中查看支持`sglang`加速的完整包安装方法。

如果需要通过自定义参数调整解析选项，您也可以在文档中查看更详细的[命令行工具使用说明](./cli_tools.md)。

## 通过api、webui、sglang-client/server进阶使用

- 通过python api直接调用：[Python 调用示例](https://github.com/opendatalab/MinerU/blob/master/demo/demo.py)
- 通过fast api方式调用：
  ```bash
  mineru-api --host 0.0.0.0 --port 8000
  ```
  >[!TIP]
  >在浏览器中访问 `http://127.0.0.1:8000/docs` 查看API文档。
- 启动gradio webui 可视化前端：
  ```bash
  # 使用 pipeline/vlm-transformers/vlm-sglang-client 后端
  mineru-gradio --server-name 0.0.0.0 --server-port 7860
  # 或使用 vlm-sglang-engine/pipeline 后端（需安装sglang环境）
  mineru-gradio --server-name 0.0.0.0 --server-port 7860 --enable-sglang-engine true
  ```
  >[!TIP]
  > 
  >- 在浏览器中访问 `http://127.0.0.1:7860` 使用 Gradio WebUI。
  >- 访问 `http://127.0.0.1:7860/?view=api` 使用 Gradio API。
- 使用`sglang-client/server`方式调用：
  ```bash
  # 启动sglang server(需要安装sglang环境)
  mineru-sglang-server --port 30000
  ``` 
  >[!TIP]
  >在另一个终端中通过sglang client连接sglang server（只需cpu与网络，不需要sglang环境）
  > ```bash
  > mineru -p <input_path> -o <output_path> -b vlm-sglang-client -u http://127.0.0.1:30000
  > ```

> [!NOTE]
> 所有sglang官方支持的参数都可用通过命令行参数传递给 MinerU，包括以下命令:`mineru`、`mineru-sglang-server`、`mineru-gradio`、`mineru-api`，
> 我们整理了一些`sglang`使用中的常用参数和使用方法，可以在文档[命令行进阶参数](./advanced_cli_parameters.md)中获取。

## 基于配置文件扩展 MinerU 功能

MinerU 现已实现开箱即用，但也支持通过配置文件扩展功能。您可通过编辑用户目录下的 `mineru.json` 文件，添加自定义配置。

>[!IMPORTANT]
>`mineru.json` 文件会在您使用内置模型下载命令 `mineru-models-download` 时自动生成，也可以通过将[配置模板文件](https://github.com/opendatalab/MinerU/blob/master/mineru.template.json)复制到用户目录下并重命名为 `mineru.json` 来创建。  

以下是一些可用的配置选项： 

- `latex-delimiter-config`：用于配置 LaTeX 公式的分隔符，默认为`$`符号，可根据需要修改为其他符号或字符串。
- `llm-aided-config`：用于配置 LLM 辅助标题分级的相关参数，兼容所有支持`openai协议`的 LLM 模型，默认使用`阿里云百炼`的`qwen2.5-32b-instruct`模型，您需要自行配置 API 密钥并将`enable`设置为`true`来启用此功能。
- `models-dir`：用于指定本地模型存储目录，请为`pipeline`和`vlm`后端分别指定模型目录，指定目录后您可通过配置环境变量`export MINERU_MODEL_SOURCE=local`来使用本地模型。
