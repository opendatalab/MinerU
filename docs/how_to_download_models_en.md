### 1. Install Git LFS
Before you begin, make sure Git Large File Storage (Git LFS) is installed on your system. Install it using the following command:

```bash
git lfs install
```

### 2. Download the Model from Hugging Face
To download the `PDF-Extract-Kit` model from Hugging Face, use the following command:

```bash
git lfs clone https://huggingface.co/wanderkid/PDF-Extract-Kit
```

Ensure that Git LFS is enabled during the clone to properly download all large files.

### 3. Additional steps

#### 1. Check whether the model directory is downloaded completely.

The structure of the model folder is as follows, including configuration files and weight files of different components:
```
../
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
│── TabRec
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
#### 2. Check whether the model file is fully downloaded.

Please check whether the size of the model file in the directory is consistent with the description on the web page. If possible, it is best to check whether the model is downloaded completely through sha256.

#### 3. Move the model to the solid-state drive

Move the 'models' directory to a directory with large disk space, preferably on a solid-state drive (SSD). In addition, modify the model directory in `~/magic-pdf.json` to point to the final model storage location, otherwise the model cannot be loaded.