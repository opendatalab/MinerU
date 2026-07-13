# 模型源说明

MinerU使用 `HuggingFace` 和 `ModelScope` 作为模型仓库，用户可以根据需要切换模型源或使用本地模型。

- `auto` 是默认的模型源策略，会优先探测 HuggingFace 是否可访问；可访问时使用 `HuggingFace`，不可访问时自动回退到 `ModelScope`。
- `HuggingFace` 在全球范围内提供了优异的加载速度和极高稳定性。
- `ModelScope` 是中国大陆地区用户的最佳选择，提供了无缝兼容的SDK模块，适用于无法访问`HuggingFace`的用户。

## 模型源的切换方法

### 通过环境变量切换
MinerU 通过 `MINERU_MODEL_SOURCE` 环境变量配置模型源，这适用于所有命令行工具和 API 调用。支持的取值为 `auto`、`huggingface`、`modelscope` 和 `local`，环境变量优先级高于 `config.yaml` 中的 `model.source`。
```bash
export MINERU_MODEL_SOURCE=modelscope
mineru -p <input_path> -o <output_path>
```
或在代码中设置：
```python
import os
os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
```
>[!TIP]
> MinerU 已不再提供用于切换模型源的命令行参数。通过环境变量设置的模型源会在当前终端会话中生效，直到终端关闭或环境变量被修改。

### 通过配置文件切换
如果未设置 `MINERU_MODEL_SOURCE`，MinerU 会读取 `config.yaml` 中的 `model.source` 字段。`model.source` 支持 `auto`、`huggingface`、`modelscope` 和 `local`。当值为 `auto` 或字段缺失时，会先自动探测实际来源；如果该值来自配置文件或内置默认值，首次自动探测完成后会将 `model.source` 写回为 `huggingface` 或 `modelscope`，避免后续启动时因网络波动反复切换来源。
```yaml
model:
  source: auto
  base_dir: ~/.mineru/models
```


## 使用本地模型

### 1. 下载模型到本地
```bash
mineru-kit models download --help
```
或下载全部内置模型包：
```bash
mineru-kit models download --tier high
```
> [!NOTE]
>- 模型会下载到 `config.model.base_dir` 下，默认路径为 `~/.mineru/models`。
>- `mineru-kit models download` 不会把模型路径写入 `mineru.json`。
>- 如需自定义模型目录，请先在 `config.yaml` 中设置 `model.base_dir`，再执行下载。
>- 如需更新模型文件，可以再次运行 `mineru-kit models download --tier high`；在同一个 `model.base_dir` 下，provider SDK 会尽量复用已有文件做增量更新。
>- `mineru-kit models download` 必须使用远端模型源执行真实下载；如果当前配置为 `model.source: local`，该命令会仅在本次执行中临时按 `auto` 处理。

### 2. 使用本地模型进行解析

通过环境变量启用本地模型：
```bash
export MINERU_MODEL_SOURCE=local
mineru -p <input_path> -o <output_path>
```
