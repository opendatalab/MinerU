"""验证解析成功不能绕过发布包的 Python 和二进制支持声明。"""
import json
from pathlib import Path

import pytest

from scripts.check_transformers5_dependencies import check_backend_dependencies, verify_wheels


@pytest.mark.parametrize("platform", ["windows", "linux", "macos-intel"])
def test_runtime_and_magika_distribution_identity(platform: str) -> None:
    """求解成功后仍拒绝共享模块的发行包冲突及旧 Magika 依赖。"""
    runtime = "onnxruntime-directml" if platform == "windows" else "onnxruntime"
    other_runtime = "onnxruntime" if platform == "windows" else "onnxruntime-directml"
    selected = {"gradio", runtime, "mineru-llama-cpp", "mineru-magika"}
    assert check_backend_dependencies(selected, "mineru", "base", platform) == []
    assert check_backend_dependencies(selected | {other_runtime}, "mineru", "base", platform)
    assert check_backend_dependencies(selected | {"magika"}, "mineru", "base", platform)
    assert check_backend_dependencies(selected - {"mineru-magika"}, "mineru", "base", platform)


def test_python_metadata_is_checked_without_binary_wheel_option(tmp_path: Path) -> None:
    """模拟求解器放过旧包的情况，即使不查 wheel 也必须拒绝超出声明的 Python。"""
    metadata = {"info": {"requires_python": ">=3.10,<3.14"}, "urls": [{"filename": "sample-1.0-py3-none-any.whl"}]}
    (tmp_path / "sample-1.0.json").write_text(json.dumps(metadata))
    errors = verify_wheels("sample==1.0", "linux", "3.14", tmp_path, check_wheels=False)
    assert len(errors) == 1
    assert "Python 3.14 excluded" in errors[0]


def test_binary_wheel_check_uses_target_python_and_platform(tmp_path: Path) -> None:
    """其他平台的旧 ABI wheel 不能冒充目标环境的可安装发行文件。"""
    metadata = {"info": {"requires_python": ">=3.10"}, "urls": [{"filename": "sample-1.0-cp313-cp313-win_amd64.whl"}]}
    (tmp_path / "sample-1.0.json").write_text(json.dumps(metadata))
    assert not verify_wheels("sample==1.0", "linux", "3.14", tmp_path, check_wheels=False)
    errors = verify_wheels("sample==1.0", "linux", "3.14", tmp_path, check_wheels=True)
    assert len(errors) == 1
    assert "no wheel for linux Python 3.14" in errors[0]
