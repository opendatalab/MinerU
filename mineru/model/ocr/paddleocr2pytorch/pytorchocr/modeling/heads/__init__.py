# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ["build_head", "ClsHead", "DBHead", "PFHeadLocal", "CTCHead", "MultiHead"]


def build_head(config, **kwargs):
    # det head
    # cls head
    from .cls_head import ClsHead  # noqa: F401
    from .det_db_head import DBHead, PFHeadLocal  # noqa: F401

    # rec head
    from .rec_ctc_head import CTCHead  # noqa: F401
    from .rec_multi_head import MultiHead  # noqa: F401

    support_dict = [
        "DBHead",
        "CTCHead",
        "ClsHead",
        "MultiHead",
        "PFHeadLocal",
    ]

    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "head only support {}".format(support_dict)
    )
    module_class = eval(module_name)(**config, **kwargs)
    return module_class
