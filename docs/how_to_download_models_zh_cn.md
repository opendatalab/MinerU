# 如何下载模型文件

模型文件可以从 Hugging Face 或 Model Scope 下载，由于网络原因，国内用户访问HF可能会失败，请使用 ModelScope。

<details>
  <summary>方法一：从 Hugging Face 下载模型</summary>
  <p>使用python脚本 从Hugging Face下载模型文件</p>
  <pre><code>pip install huggingface_hub
wget https://gitee.com/myhloli/MinerU/raw/master/docs/download_models_hf.py
python download_models_hf.py</code></pre>
  <p>python脚本执行完毕后，会输出模型下载目录</p>
</details>

## 方法二：从 ModelScope 下载模型

### 使用python脚本 从ModelScope下载模型文件

```bash
pip install modelscope
wget https://gitee.com/myhloli/MinerU/raw/master/docs/download_models.py
python download_models.py
```
python脚本执行完毕后，会输出模型下载目录


## 下载完成后的操作：修改magic-pdf.json中的模型路径
在`~/magic-pdf.json`里修改模型的目录指向上一步脚本输出的models目录的绝对路径，否则会报模型无法加载的错误。
