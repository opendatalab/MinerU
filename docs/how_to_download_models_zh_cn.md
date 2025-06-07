模型下载分为首次下载和更新模型目录，请参考对应的文档内容进行操作

# 首次下载模型文件

模型文件可以从 Hugging Face 或 Model Scope 下载，由于网络原因，国内用户访问HF可能会失败，请使用 ModelScope。

<details>
  <summary>方法一：从 Hugging Face 下载模型</summary>
  <p>使用python脚本 从Hugging Face下载模型文件</p>
  <pre><code>pip install huggingface_hub
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models_hf.py -O download_models_hf.py
python download_models_hf.py</code></pre>
  <p>python脚本会自动下载模型文件并配置好配置文件中的模型目录</p>
</details>

## 方法二：从 ModelScope 下载模型

### 使用python脚本 从ModelScope下载模型文件

```bash
pip install modelscope
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models.py -O download_models.py
python download_models.py
```
python脚本会自动下载模型文件并配置好配置文件中的模型目录

配置文件可以在用户目录中找到，文件名为`magic-pdf.json`

> [!TIP]
> windows的用户目录为 "C:\\Users\\用户名", linux用户目录为 "/home/用户名", macOS用户目录为 "/Users/用户名"


# 此前下载过模型，如何更新

## 1. 通过 Hugging Face 或 Model Scope 下载过模型

如此前通过 HuggingFace 或 Model Scope 下载过模型，可以重复执行此前的模型下载python脚本，将会自动将模型目录更新到最新版本。
