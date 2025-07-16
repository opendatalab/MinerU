# 模型源说明

MinerU使用 `HuggingFace` 和 `ModelScope` 作为模型仓库，用户可以根据需要切换模型源或使用本地模型。

- `HuggingFace` 是默认的模型源，在全球范围内提供了优异的加载速度和极高稳定性。
- `ModelScope` 是中国大陆地区用户的最佳选择，提供了无缝兼容的SDK模块，适用于无法访问`HuggingFace`的用户。

## 模型源的切换方法

### 通过命令行参数切换
目前仅`mineru`命令行工具支持通过命令行参数切换模型源，其他命令行工具如`mineru-api`、`mineru-gradio`等暂不支持。
```bash
mineru -p <input_path> -o <output_path> --source modelscope
```

### 通过环境变量切换
在任何情况下可以通过设置环境变量来切换模型源，这适用于所有命令行工具和API调用。
```bash
export MINERU_MODEL_SOURCE=modelscope
```
或
```python
import os
os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
```
>[!TIP]
> 通过环境变量设置的模型源会在当前终端会话中生效，直到终端关闭或环境变量被修改。且优先级高于命令行参数，如同时设置了命令行参数和环境变量，命令行参数将被忽略。


## 使用本地模型

### 1. 下载模型到本地
```bash
mineru-models-download --help
```
或使用交互式命令行工具选择模型下载：
```bash
mineru-models-download
```
> [!NOTE]
>- 下载完成后，模型路径会在当前终端窗口输出，并自动写入用户目录下的 `mineru.json`。
>- 您也可以通过将[配置模板文件](https://github.com/opendatalab/MinerU/blob/master/mineru.template.json)复制到用户目录下并重命名为 `mineru.json` 来创建配置文件。
>- 模型下载到本地后，您可以自由移动模型文件夹到其他位置，同时需要在 `mineru.json` 中更新模型路径。
>- 如您将模型文件夹部署到其他服务器上，请确保将 `mineru.json`文件一同移动到新设备的用户目录中并正确配置模型路径。
>- 如您需要更新模型文件，可以再次运行 `mineru-models-download` 命令，模型更新暂不支持自定义路径，如您没有移动本地模型文件夹，模型文件会增量更新；如您移动了模型文件夹，模型文件会重新下载到默认位置并更新`mineru.json`。

### 2. 使用本地模型进行解析

```bash
mineru -p <input_path> -o <output_path> --source local
```
或通过环境变量启用：
```bash
export MINERU_MODEL_SOURCE=local
mineru -p <input_path> -o <output_path>
```