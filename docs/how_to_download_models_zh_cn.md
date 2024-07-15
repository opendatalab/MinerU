### 安装 Git LFS
开始之前，请确保您的系统上已安装 Git 大文件存储 (Git LFS)。使用以下命令进行安装

```bash
git lfs install
```

### 从 Hugging Face 下载模型
请使用以下命令从 Hugging Face 下载 PDF-Extract-Kit 模型：

```bash
git lfs clone https://huggingface.co/wanderkid/PDF-Extract-Kit
```

确保在克隆过程中启用了 Git LFS，以便正确下载所有大文件。


### 从 ModelScope 下载模型

#### SDK下载

```bash
# 首先安装modelscope
pip install modelscope
```

```python
# 使用modelscope sdk下载模型
from modelscope import snapshot_download
model_dir = snapshot_download('wanderkid/PDF-Extract-Kit')
```

#### Git下载
也可以使用git clone从 ModelScope 下载模型:

```bash
git clone https://www.modelscope.cn/wanderkid/PDF-Extract-Kit.git
```


将 'models' 目录移动到具有较大磁盘空间的目录中，最好是在固态硬盘(SSD)上。


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
