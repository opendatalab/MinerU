### 1. Download the Model from Hugging Face
Use a Python Script to Download Model Files from Hugging Face
```bash
pip install huggingface_hub
wget https://github.com/opendatalab/MinerU/raw/master/docs/download_models_hf.py
python download_models_hf.py
```
After the Python script finishes executing, it will output the directory where the models are downloaded.

### 2. To modify the model path address in the configuration file

Additionally, in `~/magic-pdf.json`, update the model directory path to the absolute path of the `models` directory output by the previous Python script. Otherwise, you will encounter an error indicating that the model cannot be loaded.

