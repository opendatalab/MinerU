# Copyright (c) Opendatalab. All rights reserved.
"""CPU 线程上限的回归测试，覆盖 issue #5472 中导致线程泄漏的场景。

libgomp 会为每个进入过 OpenMP 并行区的 master 线程保留一组常驻 worker，
而 ``asyncio.to_thread`` 使用的 ``ThreadPoolExecutor`` 线程与进程同生命周期，
因此单个算子的线程数如果不设上限，长时间运行的 ``mineru-api`` 进程线程数会
朝着 ``线程池大小 * os.cpu_count()`` 增长。
"""
import sys
import types

import pytest

from mineru.utils import os_env_config
from mineru.utils.os_env_config import (
    CPU_THREAD_LIMIT_ENV_NAMES,
    DEFAULT_CPU_THREAD_LIMIT,
    apply_cpu_thread_limit,
    cpu_thread_limit_already_configured,
    resolve_cpu_thread_limit,
)


class _FakeTorch(types.ModuleType):
    """记录 set_num_threads 调用的 torch 替身。"""

    def __init__(self, num_threads: int) -> None:
        super().__init__("torch")
        self._num_threads = num_threads
        self.set_calls: list[int] = []

    def get_num_threads(self) -> int:
        """返回当前 torch CPU 线程数。"""
        return self._num_threads

    def set_num_threads(self, num_threads: int) -> None:
        """记录并应用 torch CPU 线程数。"""
        self.set_calls.append(num_threads)
        self._num_threads = num_threads


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """清空所有相关环境变量，模拟未做任何线程配置的默认部署。"""
    for env_name in CPU_THREAD_LIMIT_ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.delenv("MINERU_CPU_NUM_THREADS", raising=False)
    monkeypatch.delenv("MINERU_INTRA_OP_NUM_THREADS", raising=False)
    return monkeypatch


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, num_threads: int) -> _FakeTorch:
    fake_torch = _FakeTorch(num_threads)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


def test_resolve_defaults_to_thread_limit_when_unset(clean_env: pytest.MonkeyPatch) -> None:
    """未做任何配置时应回落到默认上限，而不是整机核心数。"""
    assert resolve_cpu_thread_limit(cpu_count=96) == DEFAULT_CPU_THREAD_LIMIT


def test_resolve_never_exceeds_cpu_count(clean_env: pytest.MonkeyPatch) -> None:
    """小核心机器上的上限不得超过实际核心数。"""
    assert resolve_cpu_thread_limit(cpu_count=2) == 2
    assert resolve_cpu_thread_limit(cpu_count=1) == 1


def test_resolve_prefers_mineru_cpu_num_threads(clean_env: pytest.MonkeyPatch) -> None:
    """MINERU_CPU_NUM_THREADS 优先级最高。"""
    clean_env.setenv("MINERU_CPU_NUM_THREADS", "4")
    clean_env.setenv("MINERU_INTRA_OP_NUM_THREADS", "16")
    assert resolve_cpu_thread_limit(cpu_count=96) == 4


def test_resolve_falls_back_to_intra_op_num_threads(clean_env: pytest.MonkeyPatch) -> None:
    """未设置专用变量时复用既有的 MINERU_INTRA_OP_NUM_THREADS。"""
    clean_env.setenv("MINERU_INTRA_OP_NUM_THREADS", "6")
    assert resolve_cpu_thread_limit(cpu_count=96) == 6


@pytest.mark.parametrize("bad_value", ["abc", "0", "-4", ""])
def test_resolve_falls_back_on_invalid_values(
    clean_env: pytest.MonkeyPatch,
    bad_value: str,
) -> None:
    """非法或非正数取值必须回落到默认上限，而不是崩溃或放开上限。"""
    clean_env.setenv("MINERU_CPU_NUM_THREADS", bad_value)
    assert resolve_cpu_thread_limit(cpu_count=96) == DEFAULT_CPU_THREAD_LIMIT


def test_apply_caps_torch_threads_on_many_core_host(clean_env: pytest.MonkeyPatch) -> None:
    """核心场景：96 核机器上默认必须收敛线程数，这正是 #5472 的泄漏场景。"""
    fake_torch = _install_fake_torch(clean_env, num_threads=96)

    applied = apply_cpu_thread_limit(cpu_count=96)

    assert applied == DEFAULT_CPU_THREAD_LIMIT
    assert fake_torch.set_calls == [DEFAULT_CPU_THREAD_LIMIT]
    assert fake_torch.get_num_threads() == DEFAULT_CPU_THREAD_LIMIT


@pytest.mark.parametrize("env_name", CPU_THREAD_LIMIT_ENV_NAMES)
def test_apply_respects_explicit_env_configuration(
    clean_env: pytest.MonkeyPatch,
    env_name: str,
) -> None:
    """部署方已显式配置线程数时不得覆盖，也不得改动 torch 线程数。"""
    clean_env.setenv(env_name, "2")
    fake_torch = _install_fake_torch(clean_env, num_threads=96)

    assert cpu_thread_limit_already_configured() is True
    assert apply_cpu_thread_limit(cpu_count=96) is None
    assert fake_torch.set_calls == []


def test_apply_does_not_raise_torch_threads(clean_env: pytest.MonkeyPatch) -> None:
    """torch 线程数已低于上限时不得反向调高，避免放大 OpenMP team。"""
    fake_torch = _install_fake_torch(clean_env, num_threads=4)

    assert apply_cpu_thread_limit(cpu_count=96) is None
    assert fake_torch.set_calls == []
    assert fake_torch.get_num_threads() == 4


def test_apply_is_a_noop_without_torch(clean_env: pytest.MonkeyPatch) -> None:
    """未安装 torch 的环境下必须安静跳过，不能让 API 启动失败。"""
    clean_env.setitem(sys.modules, "torch", None)

    assert apply_cpu_thread_limit(cpu_count=96) is None


def test_apply_is_idempotent(clean_env: pytest.MonkeyPatch) -> None:
    """重复调用不应反复设置，便于在启动路径上安全调用。"""
    fake_torch = _install_fake_torch(clean_env, num_threads=96)

    assert apply_cpu_thread_limit(cpu_count=96) == DEFAULT_CPU_THREAD_LIMIT
    assert apply_cpu_thread_limit(cpu_count=96) is None
    assert fake_torch.set_calls == [DEFAULT_CPU_THREAD_LIMIT]


def test_vlm_omp_default_is_not_shadowed(clean_env: pytest.MonkeyPatch) -> None:
    """修复不得通过导出 OMP_NUM_THREADS 生效，否则会顶掉 VLM 引擎自己的 =1 默认值。"""
    _install_fake_torch(clean_env, num_threads=96)

    apply_cpu_thread_limit(cpu_count=96)

    for env_name in CPU_THREAD_LIMIT_ENV_NAMES:
        assert os_env_config.os.getenv(env_name) is None
