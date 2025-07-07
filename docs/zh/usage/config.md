
# 基于配置文件扩展 MinerU 功能

- MinerU 现已实现开箱即用，但也支持通过配置文件扩展功能。您可以在用户目录下创建 `mineru.json` 文件，添加自定义配置。
- `mineru.json` 文件会在您使用内置模型下载命令 `mineru-models-download` 时自动生成，也可以通过将[配置模板文件](./mineru.template.json)复制到用户目录下并重命名为 `mineru.json` 来创建。
- 以下是一些可用的配置选项：
  - `latex-delimiter-config`：用于配置 LaTeX 公式的分隔符，默认为`$`符号，可根据需要修改为其他符号或字符串。
  - `llm-aided-config`：用于配置 LLM 辅助标题分级的相关参数，兼容所有支持`openai协议`的 LLM 模型，默认使用`阿里云百炼`的`qwen2.5-32b-instruct`模型，您需要自行配置 API 密钥并将`enable`设置为`true`来启用此功能。
  - `models-dir`：用于指定本地模型存储目录，请为`pipeline`和`vlm`后端分别指定模型目录，指定目录后您可通过配置环境变量`export MINERU_MODEL_SOURCE=local`来使用本地模型。

---