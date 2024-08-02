### Install Git LFS
Before you begin, make sure Git Large File Storage (Git LFS) is installed on your system. Install it using the following command:

```bash
git lfs install
```

### Download the Model from Hugging Face
To download the `PDF-Extract-Kit` model from Hugging Face, use the following command:

```bash
git lfs clone https://huggingface.co/wanderkid/PDF-Extract-Kit
```

Ensure that Git LFS is enabled during the clone to properly download all large files.



### Download the Model from ModelScope

#### SDK Download

```bash
# First, install the ModelScope library using pip:
pip install modelscope
```

```python
# Use the following Python code to download the model using the ModelScope SDK:
from modelscope import snapshot_download
model_dir = snapshot_download('wanderkid/PDF-Extract-Kit')
```

#### Git Download
Alternatively, you can use Git to clone the model repository from ModelScope:

```bash
git lfs clone https://www.modelscope.cn/wanderkid/PDF-Extract-Kit.git
```


Put [model files]() here:

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
