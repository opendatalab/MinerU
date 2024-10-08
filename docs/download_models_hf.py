from huggingface_hub import snapshot_download
model_dir = snapshot_download('opendatalab/PDF-Extract-Kit')
layoutreader_model_dir = snapshot_download('hantian/layoutreader')
print(f"model dir is: {model_dir}/models")
