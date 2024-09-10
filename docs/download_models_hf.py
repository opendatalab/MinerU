from huggingface_hub import snapshot_download
model_dir = snapshot_download('opendatalab/PDF-Extract-Kit')
print(f"model dir is: {model_dir}/models")
