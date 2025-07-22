from __future__ import absolute_import, division, print_function, unicode_literals

import copy

__all__ = [
    "build_post_process",
    "ClsPostProcess",
    "DBPostProcess",
    "AttnLabelDecode",
    "CANLabelDecode",
    "CTCLabelDecode",
    "NRTRLabelDecode",
    "RFLLabelDecode",
    "SARLabelDecode",
    "SRNLabelDecode",
    "TableLabelDecode",
    "ViTSTRLabelDecode",
]


def build_post_process(config, global_config=None):
    from .cls_postprocess import ClsPostProcess  # noqa: F401
    from .db_postprocess import DBPostProcess  # noqa: F401
    from .rec_postprocess import (  # noqa: F401
        AttnLabelDecode,
        CANLabelDecode,
        CTCLabelDecode,
        NRTRLabelDecode,
        RFLLabelDecode,
        SARLabelDecode,
        SRNLabelDecode,
        TableLabelDecode,
        ViTSTRLabelDecode,
    )

    support_dict = [
        "DBPostProcess",
        "CTCLabelDecode",
        "AttnLabelDecode",
        "ClsPostProcess",
        "SRNLabelDecode",
        "TableLabelDecode",
        "NRTRLabelDecode",
        "SARLabelDecode",
        "ViTSTRLabelDecode",
        "CANLabelDecode",
        "RFLLabelDecode",
    ]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        "post process only support {}, but got {}".format(support_dict, module_name)
    )
    module_class = eval(module_name)(**config)
    return module_class
