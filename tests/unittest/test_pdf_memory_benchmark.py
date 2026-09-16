"""守卫外部内存采样的预热排除、同时刻合计和未知 USS 语义。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import benchmark_pdf_memory


def test_benchmark_reports_simultaneous_peaks_and_unknown_uss(tmp_path: Path) -> None:
    """排除预热峰值，不能把不同时间的 PID 峰值相加或把未知 USS 当作零。"""
    rounds = [
        {"warmup": True, "started": 0, "released": 1, "parse_seconds": 50},
        {"warmup": False, "started": 2, "released": 3, "parse_seconds": 1},
        {"warmup": False, "started": 4, "released": 5, "parse_seconds": 3},
    ]
    samples = [
        {"time": 0.5, "processes": [{"pid": 1, "role": "parser", "rss": 10000, "uss": 10000}]},
        {
            "time": 2.5,
            "processes": [
                {"pid": 1, "role": "parser", "rss": 100, "uss": 50},
                {"pid": 2, "role": "child", "rss": 200, "uss": 150},
            ],
        },
        {
            "time": 4.5,
            "processes": [
                {"pid": 1, "role": "parser", "rss": 200, "uss": None},
                {"pid": 2, "role": "child", "rss": 50, "uss": 30},
            ],
        },
    ]
    (tmp_path / "rounds.jsonl").write_text("\n".join(json.dumps(record) for record in rounds))
    (tmp_path / "samples.jsonl").write_text("\n".join(json.dumps(record) for record in samples))
    args = SimpleNamespace(
        output=tmp_path, repo=tmp_path, trim="0", tier="flash", mode="txt", window_size=2, fingerprints=False
    )
    summary = benchmark_pdf_memory._summarize(args, 0)
    assert summary["measured_rounds"] == 2
    assert summary["median_parse_seconds"] == 2
    assert summary["sampled_total_rss_peak"] == 300
    assert summary["sampled_total_uss_peak"] == 200
    assert [record["rss"] for record in summary["process_peaks"]] == [200, 200]

    # 某个 PID 的 USS 始终不可读时，不生成不完整的总体 USS 峰值。
    for sample in samples:
        sample["processes"][0]["uss"] = None
    (tmp_path / "samples.jsonl").write_text("\n".join(json.dumps(record) for record in samples))
    summary = benchmark_pdf_memory._summarize(args, 0)
    assert summary["sampled_total_uss_peak"] is None
    assert summary["process_peaks"][0]["uss"] is None
