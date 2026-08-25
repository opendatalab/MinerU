# Copyright (c) Opendatalab. All rights reserved.
"""模型层共享的稳定标识类型。"""


class AtomicModelName:
    """本地原子模型名称集合，供运行时和模型实现统一索引。"""

    Layout = "layout"
    MFR = "mfr"
    OCR = "ocr"
    WirelessTable = "wireless_table"
    WiredTable = "wired_table"
    TableCls = "table_cls"
    TableOrientationCls = "table_ori_cls"


__all__ = ["AtomicModelName"]
