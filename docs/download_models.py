# use modelscope sdk download models
from modelscope import snapshot_download
model_dir = snapshot_download('opendatalab/PDF-Extract-Kit')
print(f"model dir is: {model_dir}/models")
