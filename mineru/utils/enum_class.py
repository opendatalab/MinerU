# Copyright (c) Opendatalab. All rights reserved.


class ModelPath:
    vlm_root_hf = "opendatalab/MinerU2.5-Pro-2605-1.2B"
    vlm_root_modelscope = "OpenDataLab/MinerU2.5-Pro-2605-1.2B"
    pipeline_root_modelscope = "OpenDataLab/PDF-Extract-Kit-1.0"
    pipeline_root_hf = "opendatalab/PDF-Extract-Kit-1.0"
    pp_doclayout_v2 = "models/Layout/PP-DocLayoutV2"
    unimernet_small = "models/MFR/unimernet_hf_small_2503"
    pp_formulanet_plus_m = "models/MFR/pp_formulanet_plus_m"
    pytorch_paddle = "models/OCR/paddleocr_torch"
    slanet_plus = "models/TabRec/SlanetPlus/slanet-plus.onnx"
    unet_structure = "models/TabRec/UnetStructure/unet.onnx"
    paddle_table_cls = "models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx"


class ImageType:
    PIL = "pil_img"
    BASE64 = "base64_img"
