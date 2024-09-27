模型下载分为首次下载和更新模型目录，请参考对应的文档内容进行操作

# 首次下载模型文件

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



# 此前下载过模型，如何更新

## 1. 通过git lfs下载过模型

>由于部分用户反馈通过git lfs下载模型文件遇到下载不全和模型文件损坏情况，现已不推荐使用该方式下载。

如此前通过 git lfs 下载过模型文件，可以进入到之前的下载目录中，通过`git pull`命令更新模型。

## 2. 通过 Hugging Face 或 Model Scope 下载过模型

如此前通过 HuggingFace 或 Model Scope 下载过模型，可以重复执行此前的模型下载python脚本，将会自动将模型目录更新到最新版本。
