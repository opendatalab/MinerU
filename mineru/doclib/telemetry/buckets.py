"""Telemetry bucket helpers."""

from __future__ import annotations

DURATION_BUCKETS = ("lt_1s", "1_5s", "5_30s", "30_120s", "2_10m", "gt_10m")
PAGES_BUCKETS = ("1", "2_5", "6_20", "21_100", "101_500", "gt_500")
FILE_SIZE_BUCKETS = ("lt_1mb", "1_10mb", "10_50mb", "50_200mb", "gt_200mb")
RESULTS_BUCKETS = ("0", "1_5", "6_20", "21_100", "gt_100")


def duration_bucket(duration_ms: int) -> str:
    seconds = max(0, duration_ms) / 1000
    if seconds < 1:
        return "lt_1s"
    if seconds < 5:
        return "1_5s"
    if seconds < 30:
        return "5_30s"
    if seconds < 120:
        return "30_120s"
    if seconds < 600:
        return "2_10m"
    return "gt_10m"


def pages_bucket(pages: int) -> str:
    value = max(1, pages)
    if value == 1:
        return "1"
    if value <= 5:
        return "2_5"
    if value <= 20:
        return "6_20"
    if value <= 100:
        return "21_100"
    if value <= 500:
        return "101_500"
    return "gt_500"


def file_size_bucket(size_bytes: int) -> str:
    value = max(0, size_bytes)
    mb = 1024 * 1024
    if value < mb:
        return "lt_1mb"
    if value < 10 * mb:
        return "1_10mb"
    if value < 50 * mb:
        return "10_50mb"
    if value < 200 * mb:
        return "50_200mb"
    return "gt_200mb"


def results_bucket(results: int) -> str:
    value = max(0, results)
    if value == 0:
        return "0"
    if value <= 5:
        return "1_5"
    if value <= 20:
        return "6_20"
    if value <= 100:
        return "21_100"
    return "gt_100"
