"""验证 GPU 验收入口的报告与退出码一致，不加载模型或替代真机验收。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.acceptance import lmdeploy_lifecycle


@pytest.mark.parametrize("engine,platform", [("lmdeploy", "win32"), ("vllm", "linux")])
@pytest.mark.parametrize("scenario", ["success", "execution_failure", "cleanup_failure", "incomplete"])
def test_acceptance_report_matches_exit_status(
    engine: str,
    platform: str,
    scenario: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """清理失败必须撤销先前的成功标记，未完成全部验收也不能因正常返回而变为通过。"""
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.7\n")
    evidence = tmp_path / "evidence"
    monkeypatch.setattr(sys, "argv", ["acceptance", "--sample", str(sample), "--evidence", str(evidence)])
    # 只替换脚本自身的平台视图，避免影响 macOS 上 asyncio 的事件循环选择。
    monkeypatch.setattr(lmdeploy_lifecycle, "sys", SimpleNamespace(platform=platform, stderr=sys.stderr))
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setenv("MINERU_LMDEPLOY_BACKEND", "turbomind")
    monkeypatch.setattr(lmdeploy_lifecycle, "_versions", lambda engine: {"synthetic": True})
    cleanup_calls = []

    class Pool:
        """提供可控关闭故障，保持验收执行完成后才清理的时序。"""

        async def close(self) -> None:
            """分别模拟正常关闭与 Windows 上传目录清理失败。"""
            cleanup_calls.append("close")
            if scenario == "cleanup_failure":
                raise PermissionError("synthetic upload directory cleanup failure")

    async def accept(args: argparse.Namespace, report: dict[str, Any]) -> None:
        """隔离 GPU 操作，保留成功记录、执行异常与 finally 关闭的入口契约。"""
        assert args.engine == engine
        pool = Pool()
        try:
            if scenario == "execution_failure":
                raise RuntimeError("synthetic inference failure")
            report["inflight_task_terminal_states_observed"] = scenario != "incomplete"
            report["passed"] = report["inflight_task_terminal_states_observed"]
        finally:
            await pool.close()

    monkeypatch.setattr(lmdeploy_lifecycle, "_accept", accept)
    exit_code = lmdeploy_lifecycle.main(engine)
    report = json.loads((evidence / "report.json").read_text(encoding="utf-8"))
    captured = capsys.readouterr()
    expected_pass = scenario == "success"
    assert report["passed"] is expected_pass
    assert exit_code == (0 if expected_pass else 1)
    assert report["engine"] == engine
    assert cleanup_calls == ["close"]
    if scenario in {"execution_failure", "cleanup_failure"}:
        error = "PermissionError" if scenario == "cleanup_failure" else "RuntimeError"
        assert error in report["error"]
        assert "Traceback" in report["error"]
        assert report["error"] in captured.err
    else:
        assert "error" not in report
        assert not captured.err
