"""自有上传入口只通过公共传输接口工作，并隔离迟到回执。"""

from pathlib import Path
import shutil
import subprocess


def test_upload_frontend_lifecycle() -> None:
    """运行真实脚本验证同名替换、准备重试、粘贴及清除。"""
    node = shutil.which("node")
    assert node is not None
    subprocess.run([node, str(Path(__file__).with_suffix(".cjs"))], check=True)
