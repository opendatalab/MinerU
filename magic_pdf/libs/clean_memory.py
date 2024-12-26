# Copyright (c) Opendatalab. All rights reserved.
import torch
import gc


def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.npu.is_available():
        torch.npu.empty_cache()
        torch.npu.ipc_collect()
    gc.collect()