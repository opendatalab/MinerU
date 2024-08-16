# 如何下载模型文件

模型文件可以从Hugging Face 或 Model Scope 下载，由于网络原因，国内用户访问HF 可能会失败，请使用 ModelScope。


方法一：[从 Hugging Face 下载模型](#方法一从-hugging-face-下载模型)

方法二：[从 ModelScope 下载模型](#方法二从-modelscope-下载模型)

## 方法一：从 Hugging Face 下载模型

使用Git LFS 从Hugging Face下载模型文件

```bash
git lfs install # 安装 Git 大文件存储插件 (Git LFS) 
git lfs clone https://huggingface.co/wanderkid/PDF-Extract-Kit # 从 Hugging Face 下载 PDF-Extract-Kit 模型
```


## 方法二：从 ModelScope 下载模型
ModelScope 支持SDK或模型下载，任选一个即可。

[Git lsf下载](#1利用git-lsf下载)

[SDK下载](#2利用sdk下载)

### 1）利用Git lsf下载

```bash
git lfs install
git lfs clone https://www.modelscope.cn/wanderkid/PDF-Extract-Kit.git
```

### 2）利用SDK下载

```bash
# 首先安装modelscope
pip install modelscope
```

```python
# 使用modelscope sdk下载模型
from modelscope import snapshot_download
model_dir = snapshot_download('wanderkid/PDF-Extract-Kit')
print(f"模型文件下载路径为：{model_dir}/models")
```

## 【❗️必须要做❗️】的额外步骤（模型下载完成后请务必完成以下操作）

### 1.检查模型目录是否下载完整
模型文件夹的结构如下，包含了不同组件的配置文件和权重文件：
```
./
├── Layout  # 布局检测模型
│   ├── config.json
│   └── model_final.pth
├── MFD  # 公式检测
│   └── weights.pt
├── MFR  # 公式识别模型
│   └── UniMERNet
│       ├── config.json
│       ├── preprocessor_config.json
│       ├── pytorch_model.bin
│       ├── README.md
│       ├── tokenizer_config.json
│       └── tokenizer.json
│── TabRec # 表格识别模型
│   └─StructEqTable
│       ├── config.json
│       ├── generation_config.json
│       ├── model.safetensors
│       ├── preprocessor_config.json
│       ├── special_tokens_map.json
│       ├── spiece.model
│       ├── tokenizer.json
│       └── tokenizer_config.json 
└── README.md
```

### 2.检查模型文件是否下载完整
请检查目录下的模型文件大小与网页上描述是否一致，如果可以的话，最好通过sha256校验模型是否下载完整

### 3.移动模型到固态硬盘
将 'models' 目录移动到具有较大磁盘空间的目录中，最好是在固态硬盘(SSD)上。
此外在 `~/magic-pdf.json`里修改模型的目录指向最终的模型存放位置，否则会报模型无法加载的错误。
