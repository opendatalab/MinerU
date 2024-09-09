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
│   └─ TableMaster 
│       └─ ch_PP-OCRv3_det_infer
│           ├── inference.pdiparams
│           ├── inference.pdiparams.info
│           └── inference.pdmodel
│       └─ ch_PP-OCRv3_rec_infer
│           ├── inference.pdiparams
│           ├── inference.pdiparams.info
│           └── inference.pdmodel
│       └─ table_structure_tablemaster_infer
│           ├── inference.pdiparams
│           ├── inference.pdiparams.info
│           └── inference.pdmodel
│       ├── ppocr_keys_v1.txt
│       └── table_master_structure_dict.txt
└── README.md
```

### 2.检查模型文件是否下载完整
请检查目录下的模型文件大小与网页上描述是否一致，如果可以的话，最好通过sha256校验模型是否下载完整

### 3.修改magic-pdf.json中的模型路径
此外在 `~/magic-pdf.json`里修改模型的目录指向之前python脚本输出的models目录的绝对路径，否则会报模型无法加载的错误。
