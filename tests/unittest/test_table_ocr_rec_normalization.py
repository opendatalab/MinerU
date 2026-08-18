# Copyright (c) Opendatalab. All rights reserved.
from mineru.backend.pipeline.batch_analyze import BatchAnalyze


def test_normalize_table_ocr_rec_unit_corrections():
    normalize = BatchAnalyze._normalize_table_ocr_rec_text
    # 数字/字母混淆：cmH2O 被识别为 cmH20
    assert normalize("4~30cmH20") == "4~30cmH2O"
    # 单位内部多余空格：bpm 被识别为 bp m
    assert normalize("30 ~ 300 bp m") == "30 ~ 300 bpm"


def test_normalize_table_ocr_rec_keeps_legacy_rules():
    normalize = BatchAnalyze._normalize_table_ocr_rec_text
    assert normalize("香") == "否"
    assert normalize("3號") == "3"


def test_normalize_table_ocr_rec_untouched_text():
    normalize = BatchAnalyze._normalize_table_ocr_rec_text
    assert normalize("10號") == "10號"
    assert normalize("30 ~ 300 bpm") == "30 ~ 300 bpm"
    assert normalize("cmH2O") == "cmH2O"
