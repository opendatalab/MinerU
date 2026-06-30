# Model Source Documentation

MinerU uses `HuggingFace` and `ModelScope` as model repositories. Users can switch model sources or use local models as needed.

- `auto` is the default model source policy. It first requests `https://huggingface.co/models` to check whether Hugging Face is reachable. If reachable, MinerU uses `HuggingFace`; otherwise, it falls back to `ModelScope`.
- `HuggingFace` provides excellent loading speed and high stability globally.
- `ModelScope` is the best choice for users in mainland China, providing seamlessly compatible `hf` SDK modules, suitable for users who cannot access HuggingFace.

## Methods to Switch Model Sources

### Configure via Environment Variables
MinerU configures model sources through the `MINERU_MODEL_SOURCE` environment variable. This applies to all command line tools and API calls. Supported values are `huggingface`, `modelscope`, and `local`. The environment variable has higher priority than `model-source` in `mineru.json`. Do not set this environment variable to `auto`; unset it if you want MinerU to choose a source automatically.
```bash
export MINERU_MODEL_SOURCE=modelscope
mineru -p <input_path> -o <output_path>
```
or set it programmatically:
```python
import os
os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
```
>[!TIP]
> MinerU no longer provides a CLI flag for model source selection. Model sources set through environment variables take effect in the current terminal session until the terminal is closed or the environment variable is modified.

### Configure via Configuration File
If `MINERU_MODEL_SOURCE` is not set, MinerU reads the `model-source` field from `mineru.json` in the user directory. `model-source` supports fixed values `huggingface` and `modelscope`, and also supports the template's first-run placeholder value `auto`. When the value is `auto` or the field is missing, MinerU probes the actual source first. After the first auto probe resolves an actual source, MinerU writes `model-source` back as `huggingface` or `modelscope` to avoid switching sources on later startups due to network fluctuations.
```json
{
    "model-source": "auto"
}
```

## Using Local Models

### 1. Download Models to Local Storage
```bash
mineru-kit models download --help
```
or download all built-in model bundles:
```bash
mineru-kit models download all
```
> [!NOTE]
>- After download completion, the model path will be output in the current terminal window and automatically written to `mineru.json` in the user directory. The `model-source` field records the actual remote source used for this download, either `huggingface` or `modelscope`.
>- You can also create it by copying the [configuration template file](https://github.com/opendatalab/MinerU/blob/master/mineru.template.json) to your user directory and renaming it to `mineru.json`. The template sets `model-source` to `auto`, so MinerU auto-detects once and writes back the resolved source on first use.
>- After downloading models locally, you can freely move the model folder to other locations while updating the model path in `mineru.json`.
>- If you deploy the model folder to another server, please ensure you move the `mineru.json` file to the user directory of the new device and configure the model path correctly.
>- If you need to update model files, you can run `mineru-kit models download all` again. Model updates do not support custom paths currently - if you haven't moved the local model folder, model files will be incrementally updated; if you have moved the model folder, model files will be re-downloaded to the default location and `mineru.json` will be updated.
>- `mineru-kit models download` must use a remote model source to perform a real download. If your current shell already sets `MINERU_MODEL_SOURCE=local`, this command will temporarily ignore that value for this invocation and use your selected `auto`, `huggingface`, or `modelscope` source instead.

### 2. Use Local Models for Parsing

Enable local models through environment variables:
```bash
export MINERU_MODEL_SOURCE=local
mineru -p <input_path> -o <output_path>
```
