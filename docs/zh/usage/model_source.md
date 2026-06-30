# 模型源说明

MinerU使用 `HuggingFace` 和 `ModelScope` 作为模型仓库，用户可以根据需要切换模型源或使用本地模型。

- `auto` 是默认的模型源策略，会先请求 `https://huggingface.co/models` 探测 HuggingFace 是否可访问；可访问时使用 `HuggingFace`，不可访问时自动回退到 `ModelScope`。
- `HuggingFace` 在全球范围内提供了优异的加载速度和极高稳定性。
- `ModelScope` 是中国大陆地区用户的最佳选择，提供了无缝兼容的SDK模块，适用于无法访问`HuggingFace`的用户。

## 模型源的切换方法

### 通过环境变量切换
MinerU 通过 `MINERU_MODEL_SOURCE` 环境变量配置模型源，这适用于所有命令行工具和 API 调用。支持的取值为 `huggingface`、`modelscope` 和 `local`，环境变量优先级高于 `mineru.json` 中的 `model-source`。请不要将环境变量设置为 `auto`；如需自动选择来源，请删除该环境变量。
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
如果未设置 `MINERU_MODEL_SOURCE`，MinerU 会读取用户目录下 `mineru.json` 中的 `model-source` 字段。`model-source` 支持固定值 `huggingface`、`modelscope`，也支持模板中的首次解析占位值 `auto`。当值为 `auto` 或字段缺失时，会先自动探测实际来源；首次自动探测完成后，会将 `model-source` 写回为 `huggingface` 或 `modelscope`，避免后续启动时因网络波动反复切换来源。
```json
{
    "model-source": "auto"
}
```


## 使用本地模型

### 1. 下载模型到本地
```bash
mineru-kit models download --help
```
或下载全部内置模型包：
```bash
mineru-kit models download all
```
> [!NOTE]
>- 下载完成后，模型路径会在当前终端窗口输出，并自动写入用户目录下的 `mineru.json`。配置文件中的 `model-source` 会记录本次实际使用的远端来源，即 `huggingface` 或 `modelscope`。
>- 您也可以通过将[配置模板文件](https://github.com/opendatalab/MinerU/blob/master/mineru.template.json)复制到用户目录下并重命名为 `mineru.json` 来创建配置文件；模板中的 `model-source` 默认为 `auto`，首次使用时会自动探测并写回实际来源。
>- 模型下载到本地后，您可以自由移动模型文件夹到其他位置，同时需要在 `mineru.json` 中更新模型路径。
>- 如您将模型文件夹部署到其他服务器上，请确保将 `mineru.json`文件一同移动到新设备的用户目录中并正确配置模型路径。
>- 如您需要更新模型文件，可以再次运行 `mineru-kit models download all` 命令，模型更新暂不支持自定义路径，如您没有移动本地模型文件夹，模型文件会增量更新；如您移动了模型文件夹，模型文件会重新下载到默认位置并更新`mineru.json`。
>- `mineru-kit models download` 必须使用远端模型源执行真实下载；如果当前终端已设置 `MINERU_MODEL_SOURCE=local`，该命令会仅在本次执行中临时忽略该值，并改用您选择的 `auto`、`huggingface` 或 `modelscope` 下载模型。

### 2. 使用本地模型进行解析

通过环境变量启用本地模型：
```bash
export MINERU_MODEL_SOURCE=local
mineru -p <input_path> -o <output_path>
```
