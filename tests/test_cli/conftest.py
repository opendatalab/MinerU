import pytest
import torch

def clear_gpu_memory():
    '''
    clear GPU memory
    '''
    torch.cuda.empty_cache()
    print("GPU memory cleared.")

