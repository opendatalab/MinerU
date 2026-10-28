"""Torch 线程上限、配置优先级与惰性导入回归。"""

from __future__ import annotations

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from mineru.model.runtime import cpu_threads


@pytest.mark.parametrize(
    "env,existing,expected",
    [
        ({}, 32, 8),
        ({"OPENBLAS_NUM_THREADS": "1"}, 32, 8),
        ({"MINERU_CPU_NUM_THREADS": "6"}, 32, 6),
        ({"MINERU_INTRA_OP_NUM_THREADS": "3"}, 32, 3),
        ({"OMP_NUM_THREADS": "2", "MINERU_CPU_NUM_THREADS": "8"}, 32, 2),
        ({"MKL_NUM_THREADS": "4"}, 32, 4),
        ({}, 2, 2),
        ({"MINERU_CPU_NUM_THREADS": "bad"}, 32, 8),
        ({"MINERU_CPU_NUM_THREADS": "100"}, 32, 12),
    ],
)
def test_torch_thread_limits(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], existing: int, expected: int) -> None:
    """显式 OMP/MKL 优先，OpenBLAS 不阻止限制，已有小值和有效 CPU 数受到保护。"""
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MINERU_CPU_NUM_THREADS",
        "MINERU_INTRA_OP_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(cpu_threads, "_limit", None)
    monkeypatch.setattr(cpu_threads, "_threads", threading.local())
    monkeypatch.setattr(cpu_threads, "effective_cpu_count", lambda: 12)
    local = threading.local()
    calls: list[int] = []
    nnpack_calls: list[bool] = []

    def init() -> None:
        """模拟新 OpenMP 工作线程需要显式初始化线程状态。"""
        local.count = getattr(local, "count", existing)
        calls.append(threading.get_ident())

    def set_threads(count: int) -> None:
        """只修改当前模拟工作线程，检测配置是否传播。"""
        local.count = count

    fake = SimpleNamespace(
        init_num_threads=init,
        get_num_threads=lambda: local.count,
        set_num_threads=set_threads,
        backends=SimpleNamespace(nnpack=SimpleNamespace(set_flags=nnpack_calls.append)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert cpu_threads.initialize_torch_threads() == expected
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert pool.submit(cpu_threads.initialize_torch_threads).result() == expected
    assert len(set(calls)) >= 2
    assert nnpack_calls == [False]


def test_thread_module_import_and_uninitialized_stage_are_light() -> None:
    """未初始化 Torch 的线程执行边界不加载重依赖。"""
    code = """from mineru.model.runtime.cpu_threads import prepare_torch_thread
from mineru.model.runtime.execution import local_model_stage
import sys
prepare_torch_thread()
with local_model_stage("cpu"): pass
assert "torch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.torch
def test_real_torch_thread_pool_propagation() -> None:
    """用独立进程验证实际 PyTorch 的后续线程按首次选定上限执行算子。"""
    # 基础 ONNX 安装不要求 Torch，真实算子验证仅在可选后端可用时执行。
    pytest.importorskip("torch")
    code = """import os
os.environ["MINERU_CPU_NUM_THREADS"]="2"
os.environ.pop("OMP_NUM_THREADS",None)
os.environ.pop("MKL_NUM_THREADS",None)
from mineru.model.runtime.cpu_threads import initialize_torch_threads
from concurrent.futures import ThreadPoolExecutor
import torch
torch.backends.nnpack.set_flags(True)
initialize_torch_threads()
assert not torch._C._get_nnpack_enabled()
def run(_):
    initialize_torch_threads()
    assert torch.get_num_threads() <= 2
    assert not torch._C._get_nnpack_enabled()
    return (torch.ones(64,64) @ torch.ones(64,64)).sum().item()
with ThreadPoolExecutor(max_workers=3) as pool:
    assert list(pool.map(run,range(6))) == [262144.0]*6
with torch.backends.nnpack.flags(enabled=True):
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(initialize_torch_threads).result()
    assert torch._C._get_nnpack_enabled()
assert not torch._C._get_nnpack_enabled()
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
