# Copyright (c) Opendatalab. All rights reserved.
"""macOS MPS / MLX 加速相关的单元测试。

覆盖 MPS 统一内存估算（get_vram）与表格 ONNX 模型的
CoreML/CUDA provider 选择逻辑，无需真实 GPU 或模型文件即可运行。
"""
import pytest


@pytest.fixture(autouse=True)
def _clean_device_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINERU_DEVICE_MODE", raising=False)
    monkeypatch.delenv("MINERU_VIRTUAL_VRAM_SIZE", raising=False)


class TestGetVramMps:
    def test_mps_uses_physical_memory(self) -> None:
        from mineru.utils import model_utils

        # MPS 使用统一内存，不应再回退到 1GB 的默认值。
        assert model_utils.get_vram("mps") >= 1

    def test_mps_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mineru.utils import model_utils

        monkeypatch.setenv("MINERU_VIRTUAL_VRAM_SIZE", "24")
        assert model_utils.get_vram("mps") == 24

    def test_cpu_returns_default(self) -> None:
        from mineru.utils import model_utils

        assert model_utils.get_vram("cpu") == 1

    def test_physical_memory_positive(self) -> None:
        from mineru.utils import model_utils

        assert model_utils._get_physical_memory_gb() >= 1


class TestBuildTableOnnxProviders:
    def _build(self, device: str, available: list[str]) -> list[tuple[str, dict[str, object]]]:
        import os

        os.environ["MINERU_DEVICE_MODE"] = device
        from mineru.model.table.rec.onnxruntime_provider import build_table_onnx_providers

        return build_table_onnx_providers(available)

    def test_mps_prefers_coreml(self) -> None:
        providers = self._build("mps", ["CoreMLExecutionProvider", "CPUExecutionProvider"])
        assert providers[0][0] == "CoreMLExecutionProvider"
        assert providers[-1][0] == "CPUExecutionProvider"

    def test_mps_prefers_mps_ep_without_coreml(self) -> None:
        providers = self._build("mps", ["MPSExecutionProvider", "CPUExecutionProvider"])
        assert providers[0][0] == "MPSExecutionProvider"

    def test_mps_falls_back_to_cpu(self) -> None:
        providers = self._build("mps", ["CPUExecutionProvider"])
        assert [name for name, _ in providers] == ["CPUExecutionProvider"]

    def test_cuda_prefers_cuda(self) -> None:
        providers = self._build("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert providers[0][0] == "CUDAExecutionProvider"
        assert providers[-1][0] == "CPUExecutionProvider"

    def test_cuda_falls_back_to_cpu(self) -> None:
        providers = self._build("cuda", ["CPUExecutionProvider"])
        assert [name for name, _ in providers] == ["CPUExecutionProvider"]

    def test_cpu_device_keeps_cpu(self) -> None:
        providers = self._build("cpu", ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"])
        assert [name for name, _ in providers] == ["CPUExecutionProvider"]
