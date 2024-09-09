### 1. Download the Model from Hugging Face
Use a Python Script to Download Model Files from Hugging Face
```bash
pip install huggingface_hub
wget https://github.com/opendatalab/MinerU/raw/master/docs/download_models_hf.py
python download_models_hf.py
```
After the Python script finishes executing, it will output the directory where the models are downloaded.
### 2. Additional steps

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
#### 2. Check whether the model file is fully downloaded.

Please check whether the size of the model file in the directory is consistent with the description on the web page. If possible, it is best to check whether the model is downloaded completely through sha256.

#### 3. 

Additionally, in `~/magic-pdf.json`, update the model directory path to the absolute path of the `models` directory output by the previous Python script. Otherwise, you will encounter an error indicating that the model cannot be loaded.

