#!/usr/bin/env python
from huggingface_hub import snapshot_download

if __name__ == "__main__":

    mineru_patterns = [
        "models/Layout/LayoutLMv3/*",
        "models/Layout/YOLO/*",
        "models/MFD/YOLO/*",
        "models/MFR/unimernet_small_2501/*",
        "models/TabRec/TableMaster/*",
        "models/TabRec/StructEqTable/*",
    ]
    model_dir = snapshot_download(
        "opendatalab/PDF-Extract-Kit-1.0",
        allow_patterns=mineru_patterns,
        local_dir="/opt/",
    )

    layoutreader_pattern = [
        "*.json",
        "*.safetensors",
    ]
    layoutreader_model_dir = snapshot_download(
        "hantian/layoutreader",
        allow_patterns=layoutreader_pattern,
        local_dir="/opt/layoutreader/",
    )

    model_dir = model_dir + "/models"
    print(f"model_dir is: {model_dir}")
    print(f"layoutreader_model_dir is: {layoutreader_model_dir}")
