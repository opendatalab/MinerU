# Model Source Documentation

MinerU uses `HuggingFace` and `ModelScope` as model repositories. Users can switch model sources or use local models as needed.

- `auto` is the default model source policy. It first checks whether Hugging Face is accessible. If accessible, MinerU uses `HuggingFace`, otherwise it automatically falls back to `ModelScope`.
- `HuggingFace` provides excellent loading speed and high stability globally.
- `ModelScope` is the best choice for users in mainland China, providing seamlessly compatible `hf` SDK modules, suitable for users who cannot access HuggingFace.

## Methods to Switch Model Sources

### Configure via Environment Variables
MinerU configures model sources through the `MINERU_MODEL_SOURCE` environment variable. This applies to all command line tools and API calls. Supported values are `auto`, `huggingface`, `modelscope`, and `local`. The environment variable has higher priority than `model.source` in `config.yaml`.
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
If `MINERU_MODEL_SOURCE` is not set, MinerU reads `model.source` from `config.yaml`. `model.source` supports `auto`, `huggingface`, `modelscope`, and `local`. When the value is `auto` or the field is missing, MinerU probes the actual source first. If the value came from the config file or built-in default, MinerU writes the resolved source back as `huggingface` or `modelscope` to avoid switching sources on later startups due to network fluctuations.
```yaml
model:
  source: auto
  base_dir: ~/.mineru/models
```

## Using Local Models

### 1. Download Models to Local Storage
```bash
mineru-kit models download --help
```
or download all built-in model bundles:
```bash
mineru-kit models download --tier standard
```
> [!NOTE]
>- Models are downloaded under `config.model.base_dir`. By default, this is `~/.mineru/models`.
>- `mineru-kit models download` does not write model paths to `mineru.json`.
>- If you need a custom model directory, set `model.base_dir` in `config.yaml` before downloading.
>- If you need to update model files, run `mineru-kit models download --tier standard` again. Existing files in the same `model.base_dir` are incrementally reused by the provider SDK.
>- `mineru-kit models download` must use a remote model source to perform a real download. If your current config sets `model.source: local`, this command temporarily treats it as `auto` for this invocation.
>- MinerU marks a fully downloaded repository with an empty `.mineru_complete` file at its root. Repositories downloaded by required paths instead mark each completed model directory. A directory left by an older release or an interrupted download without its marker is checked through an incremental provider download before use. With `model.source: local`, such a directory is reported as not ready instead of accessing the network.

### 2. Use Local Models for Parsing

Enable local models through environment variables:
```bash
export MINERU_MODEL_SOURCE=local
mineru -p <input_path> -o <output_path>
```
