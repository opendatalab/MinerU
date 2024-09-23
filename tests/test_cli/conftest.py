import pytest
import torch

def clear_gpu_memory():
    '''
    clear GPU memory
    '''
    torch.cuda.empty_cache()
    print("GPU memory cleared.")

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    '''
    clear GPU memory after each test
    '''
    yield
    clear_gpu_memory()