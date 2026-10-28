"""只读诊断报告、三平台库存解析和隔离进程协议的回归。"""

from __future__ import annotations

from dataclasses import asdict
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from mineru.config import get_config
from mineru.kit.commands import hardware as command
from mineru.kit.main import app
from mineru.model.runtime import hardware, hardware_probe, hardware_system, onnx_calibration_cache, onnx_device
from mineru.model.runtime.hardware_types import (
    CpuInfo,
    GpuInfo,
    HardwareInfo,
    MemoryInfo,
    PackageInfo,
    RuntimeProbe,
    SystemHardware,
)


def _report() -> HardwareInfo:
    """构造最小有效报告，测试文本和 JSON 渲染使用相同语义数据。"""
    return HardwareInfo(
        system={"name": "Linux", "release": "6.0", "architecture": "x86_64", "distribution": "Test Linux", "version": "test"},
        python={"version": "3.12", "executable": "/env/bin/python"},
        mineru={"version": "4.0", "module_path": "/source/mineru/__init__.py"},
        cpu=CpuInfo(model="test CPU", physical_cores=4, logical_cores=8, available_cores=2),
        memory=MemoryInfo(16 * 1024**3, 8 * 1024**3),
        gpus=[GpuInfo(name="integrated", kind="integrated"), GpuInfo(name="discrete", kind="discrete")],
        packages=[PackageInfo("mineru", "4.0", "/source"), PackageInfo("torch", "2.0", "/env")],
        runtimes=[RuntimeProbe("torch", "error", error="driver unavailable")],
        effective_config={"resolved": {"small_backend": "torch"}},
        calibration=onnx_calibration_cache.CalibrationInfo("/tmp/config.yaml.onnx-calibration.json", "missing"),
        warnings=["partial GPU runtime inventory"],
    )


@pytest.mark.parametrize("json_mode", [False, True])
def test_info_command_output(monkeypatch: pytest.MonkeyPatch, json_mode: bool) -> None:
    """文本与 JSON 均包含诊断信息，局部失败保留可用报告并返回成功。"""
    report = asdict(_report())
    monkeypatch.setattr(command, "_info_data", lambda: report)
    result = CliRunner().invoke(app, ["hardware", "info", *(["--json"] if json_mode else [])])
    assert result.exit_code == 0, result.output
    if json_mode:
        assert json.loads(result.stdout) == report
    else:
        assert "test CPU" in result.stdout and "未知" in result.stdout
        assert "driver unavailable" in result.stdout
    assert "vllm" not in result.stdout


@pytest.mark.parametrize(
    "arguments", [["hardware"], ["hardware", "--help"], ["hardware", "info", "--help"], ["hardware", "detect", "--help"]]
)
def test_hardware_help_never_probes(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    """无参数与帮助入口不执行诊断、下载或推荐。"""
    forbidden = Mock(side_effect=AssertionError("help must not probe"))
    monkeypatch.setattr(command, "_info_data", forbidden)
    monkeypatch.setattr(command, "_detect_data", forbidden)
    result = CliRunner().invoke(app, arguments)
    assert "Usage:" in result.output
    assert result.exit_code in {0, 2}
    forbidden.assert_not_called()


@pytest.mark.parametrize("arguments", [["hardware", "info", "--quiet"], ["hardware", "detect", "--quiet"], ["auto-tier"]])
def test_no_quiet_or_auto_tier_alias(arguments: list[str]) -> None:
    """只提供确定的两个入口，不保留未批准的参数或命令别名。"""
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code != 0
    assert "No such" in result.output


def test_installed_package_allowlist_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """未安装的包不出现在版本列表中，也不会触发运行时导入。"""

    def distribution(name: str) -> SimpleNamespace:
        """仅模拟两项安装元数据，其他发行包按真实缺失异常处理。"""
        if name not in {"mineru", "mineru-vl-utils"}:
            raise metadata.PackageNotFoundError(name)
        return SimpleNamespace(version="1.2.3", locate_file=lambda path: Path("/site-packages"))

    monkeypatch.setattr(hardware.metadata, "distribution", distribution)
    packages = hardware.installed_packages()
    assert [package.name for package in packages] == ["mineru", "mineru-vl-utils"]
    assert all(package.location == "/site-packages" for package in packages)


def test_configuration_summary_excludes_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """配置白名单保留复现设备决策的信息，排除认证和完整远程地址。"""
    monkeypatch.setattr(get_config().model.vlm, "api_key", "secret-api-token")
    monkeypatch.setattr(get_config().model.vlm, "server_url", "http://private-user:secret@host")
    monkeypatch.setenv("MINERU_VL_API_KEY", "other-secret")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    report = hardware.config_summary()
    encoded = json.dumps(report)
    assert report["remote_vlm"]["enabled"] is True
    assert report["environment"]["CUDA_VISIBLE_DEVICES"] == "1"
    for secret in ("secret-api-token", "private-user", "other-secret", "MINERU_VL_API_KEY"):
        assert secret not in encoded


def test_linux_cpu_quota_frequency_and_mixed_pci_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """模拟 Linux 文件系统并保留全部独显/集显库存，频率和内存按明确单位转换。"""
    proc, system = tmp_path / "proc", tmp_path / "sys"
    proc.mkdir()
    records = []
    for core in (0, 0, 1, 1):
        records.append(f"model name : Example CPU\nphysical id : 0\ncore id : {core}\ncpu MHz : 2400.0")
    (proc / "cpuinfo").write_text("\n\n".join(records), encoding="utf-8")
    (proc / "meminfo").write_text("MemTotal: 16384 kB\nMemAvailable: 8192 kB\n", encoding="utf-8")
    frequency = system / "devices/system/cpu/cpu0/cpufreq"
    frequency.mkdir(parents=True)
    (frequency / "base_frequency").write_text("2000000")
    (frequency / "cpuinfo_max_freq").write_text("4200000")
    for bus, vendor, chip in (("0000:00:02.0", "0x8086", "0x7d51"), ("0000:01:00.0", "0x10de", "0x2489")):
        pci = system / "bus/pci/devices" / bus
        pci.mkdir(parents=True)
        for name, value in (("class", "0x030000"), ("vendor", vendor), ("device", chip)):
            (pci / name).write_text(value)
    monkeypatch.setattr(hardware_system.shutil, "which", lambda name: None)
    result = SystemHardware()
    hardware_system._linux_cpu(result, proc, system)
    hardware_system._linux_gpus(result, system)
    assert result.cpu.model == "Example CPU" and result.cpu.physical_cores == 2
    assert (result.cpu.base_mhz, result.cpu.current_mhz, result.cpu.max_mhz) == (2000, 2400, 4200)
    assert result.memory == MemoryInfo(16384 * 1024, 8192 * 1024)
    assert [gpu.vendor for gpu in result.gpus] == ["Intel", "NVIDIA"]
    runtime = RuntimeProbe(
        "vulkan",
        "ok",
        {
            "devices": [
                {"name": "Intel GPU", "vendor_id": 0x8086, "device_id": 0x7D51, "kind": "integrated", "kind_source": "vulkan"}
            ]
        },
    )
    hardware.merge_gpu_inventory(result.gpus, runtime)
    assert len(result.gpus) == 2 and result.gpus[0].kind == "integrated"
    assert result.gpus[1].kind == "unknown"


def test_macos_unified_memory_and_unknown_frequency() -> None:
    """Apple GPU 报告统一内存，不伪造独立显存、主频或驱动版本。"""
    result = SystemHardware(memory=MemoryInfo(16 * 1024**3))
    hardware_system.parse_macos_displays(
        {
            "SPDisplaysDataType": [
                {
                    "_name": "Apple M4",
                    "sppci_model": "Apple M4",
                    "spdisplays_vendor": "sppci_vendor_Apple",
                    "sppci_bus": "spdisplays_builtin",
                    "spdisplays_revision-id": "0x01",
                    "spdisplays_ndrvs": [{"serial-number": "do-not-report"}],
                }
            ]
        },
        result,
        "arm64",
    )
    gpu = result.gpus[0]
    assert gpu.kind == "integrated" and gpu.unified_memory is True
    assert gpu.dedicated_memory_bytes is None and gpu.shared_memory_bytes == 16 * 1024**3
    assert gpu.driver is None and result.cpu.base_mhz is None
    assert "do-not-report" not in json.dumps(asdict(result))


def test_windows_cim_and_dxgi_memory() -> None:
    """CIM 提供型号/核数和全部 GPU，64 位 DXGI 显存覆盖不可靠的 AdapterRAM。"""
    result = SystemHardware()
    hardware_system.parse_windows_hardware(
        {
            "cpus": [
                {
                    "Name": "Test CPU",
                    "NumberOfCores": 8,
                    "NumberOfLogicalProcessors": 16,
                    "CurrentClockSpeed": 2800,
                    "MaxClockSpeed": 4200,
                }
            ],
            "memory": {"TotalVisibleMemorySize": 16 * 1024**2, "FreePhysicalMemory": 8 * 1024**2},
            "gpus": [{"Name": "RTX", "PNPDeviceID": "PCI\\VEN_10DE&DEV_2489", "AdapterRAM": 1, "DriverVersion": "test"}],
        },
        result,
    )
    hardware.merge_gpu_inventory(
        result.gpus,
        RuntimeProbe(
            "dxgi",
            "ok",
            {
                "devices": [
                    {
                        "name": "RTX",
                        "vendor_id": 0x10DE,
                        "device_id": 0x2489,
                        "kind": "discrete",
                        "dedicated_memory_bytes": 8 * 1024**3,
                        "shared_memory_bytes": 8 * 1024**3,
                    }
                ]
            },
        ),
    )
    assert (result.cpu.physical_cores, result.cpu.logical_cores) == (8, 16)
    assert result.cpu.base_mhz is None and result.cpu.max_mhz == 4200
    assert result.memory.total_bytes == 16 * 1024**3
    assert len(result.gpus) == 1 and result.gpus[0].dedicated_memory_bytes == 8 * 1024**3


def test_windows_cim_partial_failure_keeps_memory_and_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """一个 CIM 类不可用时，其余硬件查询仍独立运行并进入报告。"""
    run = Mock(side_effect=[None, '{"TotalVisibleMemorySize": 1024}', '{"Name": "GPU"}'])
    monkeypatch.setattr(hardware_system, "_command", run)
    result = SystemHardware()
    hardware_system._windows_hardware(result)
    assert run.call_count == 3
    assert result.cpu.model is None and result.memory.total_bytes == 1024**2
    assert result.gpus[0].name == "GPU"


def test_evidence_uses_eligible_visible_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """证据匹配实际可选适配器，不能因 ORT 原始库存仍列出隐藏设备而误判。"""
    from mineru.model.runtime.webgpu import WebGPUDevice

    path = tmp_path / "record.json"
    monkeypatch.setattr(onnx_calibration_cache, "calibration_record_path", lambda: path)
    adapter = WebGPUDevice(0x8086, 0x7D51, (("driver", "v1"),), (0x8086, 0x7D51))
    onnx_calibration_cache.save_calibration_record(
        adapter,
        {
            "intra_threads": 4,
            "inter_threads": 1,
            "cpu_elapsed": 2.0,
            "gpu_elapsed": 1.0,
            "decision_threshold": 0.8,
            "outputs_match": True,
            "decision_valid": True,
            "model_sha256": "test",
        },
    )
    payload = {"backend": "webgpu", "adapters": [asdict(WebGPUDevice(0x8086, 0x7D51))], "selected_adapter": asdict(adapter)}
    probes = [RuntimeProbe("onnx", "ok", payload)]
    assert onnx_calibration_cache.inspect_calibration_record(hardware._calibration_adapters(probes)).matches_environment is True
    payload["selected_adapter"] = None
    assert (
        onnx_calibration_cache.inspect_calibration_record(hardware._calibration_adapters(probes)).matches_environment is False
    )


def test_nvidia_tool_matches_existing_windows_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    """厂商工具补充 CIM 同一设备，不因 CIM 缺少总线地址而重复列出显卡。"""
    monkeypatch.setattr(hardware_system.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(hardware_system, "_command", lambda command, report: "00000000:01:00.0, RTX, 999.0, 8192\n")
    report = SystemHardware(gpus=[GpuInfo(name="RTX", vendor="NVIDIA", source="cim")])
    hardware_system._nvidia_details(report)
    assert len(report.gpus) == 1
    assert report.gpus[0].driver == "999.0" and report.gpus[0].dedicated_memory_bytes == 8 * 1024**3


def test_probe_timeout_and_native_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """原生日志不污染结构化结果，每项运行时导入都有独立超时边界。"""
    call = Mock(side_effect=subprocess.TimeoutExpired("probe", 15))
    monkeypatch.setattr(hardware_probe.subprocess, "run", call)
    assert hardware_probe.run_probe("torch").status == "timeout"
    assert call.call_args.kwargs["timeout"] == 15
    output = "native stdout\n" + hardware_probe.PROBE_MARKER + json.dumps(asdict(RuntimeProbe("torch", "ok"))) + "\n"
    call.side_effect = None
    call.return_value = SimpleNamespace(returncode=0, stdout=output, stderr="native stderr")
    assert hardware_probe.run_probe("torch").status == "ok"
    call.return_value.stdout = "no marker"
    assert hardware_probe.run_probe("torch").status == "error"


def test_detection_protocol_moves_native_noise_to_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """detect 只交付结果，原生 stdout 被保留到 stderr 便于诊断。"""
    payload = {"schema_version": 1, "ceiling": "standard"}
    process = SimpleNamespace(returncode=0, stdout="native output\n" + hardware_probe.PROBE_MARKER + json.dumps(payload))
    monkeypatch.setattr(hardware_probe.subprocess, "run", Mock(return_value=process))
    assert hardware_probe.run_detection() == payload
    captured = capsys.readouterr()
    assert not captured.out and "native output" in captured.err


def test_info_partial_failure_is_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """报告可在运行时探测失败时完成，并保留原有校准字节、无配置写入。"""
    record = tmp_path / "config.yaml.onnx-calibration.json"
    record.write_text("broken evidence", encoding="utf-8")
    monkeypatch.setattr(onnx_calibration_cache, "calibration_record_path", lambda: record)
    monkeypatch.setattr(
        hardware, "collect_system_hardware", lambda: SystemHardware(cpu=CpuInfo(logical_cores=8, available_cores=2))
    )
    monkeypatch.setattr(hardware, "installed_packages", lambda: [PackageInfo("mineru", "4", "/env")])
    monkeypatch.setattr(hardware, "run_probe", lambda name: RuntimeProbe(name, "error", error="probe failed"))
    forbidden = Mock(side_effect=AssertionError("info must remain read only"))
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", forbidden)
    monkeypatch.setattr(onnx_calibration_cache, "save_calibration_record", forbidden)
    result = hardware.collect_hardware_info()
    assert result.cpu.available_cores == 2 and result.calibration.status == "invalid"
    assert result.warnings and result.effective_config["resolved"] is None
    assert record.read_text(encoding="utf-8") == "broken evidence"
    assert [package.name for package in result.packages] == ["mineru"]
    forbidden.assert_not_called()


def test_help_imports_no_heavy_runtime() -> None:
    """在新进程验证帮助不会提前导入 Torch、ORT 或原生 llama 运行时。"""
    script = (
        "import sys; from typer.testing import CliRunner; from mineru.kit.main import app; "
        "r=CliRunner().invoke(app,['hardware','--help']); assert r.exit_code == 0; "
        "assert not {'torch','onnxruntime','mineru_llama_cpp'}.intersection(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
