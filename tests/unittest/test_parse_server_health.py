# Copyright (c) Opendatalab. All rights reserved.
"""Parse-server 健康探测的失败退避节奏。"""

from mineru.doclib.background.parse_server_health import ParseServerHealthCheck


def _checker(interval_sec: int = 30) -> ParseServerHealthCheck:
    return ParseServerHealthCheck(
        config_svc=None,
        interval_sec=interval_sec,
        probe_timeout_sec=10,
        startup_grace_sec=30,
        stop_timeout_sec=10,
    )


def test_healthy_or_unprobed_cycles_keep_normal_interval() -> None:
    checker = _checker()
    assert checker._next_interval_sec(0) == 30
    # 退避序列走完后封顶回正常间隔（长期宕机不放大探测频率）。
    assert checker._next_interval_sec(9) == 30


def test_failure_backoff_speeds_up_probe() -> None:
    checker = _checker()
    # 连续失败按 5/10/15/20/25/30 秒加速重试：start 阶段 server ready 后 5 秒内被发现。
    assert checker._next_interval_sec(1) == 5
    assert checker._next_interval_sec(2) == 10
    assert checker._next_interval_sec(3) == 15
    assert checker._next_interval_sec(4) == 20
    assert checker._next_interval_sec(5) == 25
    assert checker._next_interval_sec(6) == 30
    assert checker._next_interval_sec(7) == 30


def test_backoff_never_exceeds_normal_interval() -> None:
    """interval 配置小于退避值时，取较小值。"""
    checker = _checker(interval_sec=8)
    assert checker._next_interval_sec(1) == 5
    assert checker._next_interval_sec(2) == 8
    assert checker._next_interval_sec(0) == 8
