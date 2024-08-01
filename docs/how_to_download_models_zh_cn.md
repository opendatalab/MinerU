# 如何下载模型文件

模型文件可以从Hugging Face 或 Model Scope 下载，由于网络原因，国内用户访问HF 可能会失败，请使用 ModelScope。

[Hugging Face](#从-Hugging-Face-下载模型)

[ModelScope](#从-ModelScope-下载模型)

## 从 Hugging Face 下载模型

### 1.安装 Git LFS
开始之前，请确保您的系统上已安装 Git 大文件存储 (Git LFS)。使用以下命令进行安装

```bash
git lfs install
```


### 2.从 Hugging Face 下载模型
请使用以下命令从 Hugging Face 下载 PDF-Extract-Kit 模型：

```bash
git lfs clone https://huggingface.co/wanderkid/PDF-Extract-Kit
```

确保在克隆过程中启用了 Git LFS，以便正确下载所有大文件。


## 从 ModelScope 下载模型
ModelScope 支持SDK或模型下载

[SDK下载](#sdk下载)

[Git下载](#git下载)

### SDK下载

```bash
# 首先安装modelscope
pip install modelscope
```

```python
# 使用modelscope sdk下载模型
from modelscope import snapshot_download
model_dir = snapshot_download('wanderkid/PDF-Extract-Kit')
```

### Git下载
也可以使用git clone从 ModelScope 下载模型:

#### 1.安装 Git LFS
开始之前，请确保您的系统上已安装 Git 大文件存储 (Git LFS)。使用以下命令进行安装

```bash
git lfs install
```


#### 2.然后通过git lfs下载模型
```bash
git lfs clone https://www.modelscope.cn/wanderkid/PDF-Extract-Kit.git
```


## 额外步骤

### 1.检查模型目录是否下载完整
模型文件夹的结构如下，包含了不同组件的配置文件和权重文件：
```
./
├── Layout
│   ├── config.json
│   └── model_final.pth
├── MFD
│   └── weights.pt
├── MFR
│   └── UniMERNet
│       ├── config.json
│       ├── preprocessor_config.json
│       ├── pytorch_model.bin
│       ├── README.md
│       ├── tokenizer_config.json
│       └── tokenizer.json
└── README.md
```

### 2.检查模型文件是否下载完整
请检查目录下的模型文件大小与网页上描述是否一致，如果可以的话，最好通过sha256校验模型是否下载完整

### 3.移动模型到固态硬盘
将 'models' 目录移动到具有较大磁盘空间的目录中，最好是在固态硬盘(SSD)上。
