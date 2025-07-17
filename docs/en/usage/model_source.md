# Model Source Documentation

MinerU uses `HuggingFace` and `ModelScope` as model repositories. Users can switch model sources or use local models as needed.

- `HuggingFace` is the default model source, providing excellent loading speed and high stability globally.
- `ModelScope` is the best choice for users in mainland China, providing seamlessly compatible `hf` SDK modules, suitable for users who cannot access HuggingFace.

## Methods to Switch Model Sources

### Switch via Command Line Parameters
Currently, only the `mineru` command line tool supports switching model sources through command line parameters. Other command line tools such as `mineru-api`, `mineru-gradio`, etc., do not support this yet.
```bash
mineru -p <input_path> -o <output_path> --source modelscope
```

### Switch via Environment Variables
You can switch model sources by setting environment variables in any situation. This applies to all command line tools and API calls.
```bash
export MINERU_MODEL_SOURCE=modelscope
```
or
```python
import os
os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
```
>[!TIP]
> Model sources set through environment variables will take effect in the current terminal session until the terminal is closed or the environment variable is modified. They have higher priority than command line parameters - if both command line parameters and environment variables are set, the command line parameters will be ignored.

## Using Local Models

### 1. Download Models to Local Storage
```bash
mineru-models-download --help
```
or use the interactive command line tool to select model downloads:
```bash
mineru-models-download
```
> [!NOTE]
>- After download completion, the model path will be output in the current terminal window and automatically written to `mineru.json` in the user directory.
>- You can also create it by copying the [configuration template file](https://github.com/opendatalab/MinerU/blob/master/mineru.template.json) to your user directory and renaming it to `mineru.json`.
>- After downloading models locally, you can freely move the model folder to other locations while updating the model path in `mineru.json`.
>- If you deploy the model folder to another server, please ensure you move the `mineru.json` file to the user directory of the new device and configure the model path correctly.
>- If you need to update model files, you can run the `mineru-models-download` command again. Model updates do not support custom paths currently - if you haven't moved the local model folder, model files will be incrementally updated; if you have moved the model folder, model files will be re-downloaded to the default location and `mineru.json` will be updated.

### 2. Use Local Models for Parsing

```bash
mineru -p <input_path> -o <output_path> --source local
```
or enable through environment variables:
```bash
export MINERU_MODEL_SOURCE=local
mineru -p <input_path> -o <output_path>
```
