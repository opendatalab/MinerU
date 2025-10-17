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

__all__ = ["build_backbone"]


def build_backbone(config, model_type):
    if model_type == "det":
        from .det_mobilenet_v3 import MobileNetV3
        from .rec_hgnet import PPHGNet_small
        from .rec_lcnetv3 import PPLCNetV3
        from .rec_pphgnetv2 import PPHGNetV2_B4

        support_dict = [
            "MobileNetV3",
            "ResNet",
            "ResNet_vd",
            "ResNet_SAST",
            "PPLCNetV3",
            "PPHGNet_small",
            'PPHGNetV2_B4',
        ]
    elif model_type == "rec" or model_type == "cls":
        from .rec_hgnet import PPHGNet_small
        from .rec_lcnetv3 import PPLCNetV3
        from .rec_mobilenet_v3 import MobileNetV3
        from .rec_svtrnet import SVTRNet
        from .rec_mv1_enhance import MobileNetV1Enhance
        from .rec_pphgnetv2 import PPHGNetV2_B4, PPHGNetV2_B6_Formula
        support_dict = [
            "MobileNetV1Enhance",
            "MobileNetV3",
            "ResNet",
            "ResNetFPN",
            "MTB",
            "ResNet31",
            "SVTRNet",
            "ViTSTR",
            "DenseNet",
            "PPLCNetV3",
            "PPHGNet_small",
            "PPHGNetV2_B4",
            "PPHGNetV2_B6_Formula"
        ]
    else:
        raise NotImplementedError

    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "when model typs is {}, backbone only support {}".format(
            model_type, support_dict
        )
    )
    module_class = eval(module_name)(**config)
    return module_class
