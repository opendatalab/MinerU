"""
This code is refer from:
https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/ppdet/modeling/necks/fpn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.nn.init import xavier_uniform_
# import paddle.nn as nn
# import paddle.nn.functional as F
# from paddle import ParamAttr
# from paddle.nn.initializer import XavierUniform
# from paddle.nn.initializer import Normal
# from paddle.regularizer import L2Decay

__all__ = ['FCEFPN']


class ConvNormLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 norm_type='bn',
                 norm_decay=0.,
                 norm_groups=32,
                 lr_scale=1.,
                 freeze_norm=False,
                 initializer=None):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn']

        bias_attr = False

        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias=bias_attr)

        norm_lr = 0. if freeze_norm else 1.
        # param_attr = ParamAttr(
        #     learning_rate=norm_lr,
        #     regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        # bias_attr = ParamAttr(
        #     learning_rate=norm_lr,
        #     regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(
                ch_out,
            )
        elif norm_type == 'sync_bn':
            self.norm = nn.SyncBatchNorm(
                ch_out,
            )
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(
                num_groups=norm_groups,
                num_channels=ch_out,
                affine=bias_attr)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        return out


class FCEFPN(nn.Module):
    """
    Feature Pyramid Network, see https://arxiv.org/abs/1612.03144
    Args:
        in_channels (list[int]): input channels of each level which can be
            derived from the output shape of backbone by from_config
        out_channels (list[int]): output channel of each level
        spatial_scales (list[float]): the spatial scales between input feature
            maps and original input image which can be derived from the output
            shape of backbone by from_config
        has_extra_convs (bool): whether to add extra conv to the last level.
            default False
        extra_stage (int): the number of extra stages added to the last level.
            default 1
        use_c5 (bool): Whether to use c5 as the input of extra stage,
            otherwise p5 is used. default True
        norm_type (string|None): The normalization type in FPN module. If
            norm_type is None, norm will not be used after conv and if
            norm_type is string, bn, gn, sync_bn are available. default None
        norm_decay (float): weight decay for normalization layer weights.
            default 0.
        freeze_norm (bool): whether to freeze normalization layer.
            default False
        relu_before_extra_convs (bool): whether to add relu before extra convs.
            default False

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 spatial_scales=[0.25, 0.125, 0.0625, 0.03125],
                 has_extra_convs=False,
                 extra_stage=1,
                 use_c5=True,
                 norm_type=None,
                 norm_decay=0.,
                 freeze_norm=False,
                 relu_before_extra_convs=True):
        super(FCEFPN, self).__init__()
        self.out_channels = out_channels
        for s in range(extra_stage):
            spatial_scales = spatial_scales + [spatial_scales[-1] / 2.]
        self.spatial_scales = spatial_scales
        self.has_extra_convs = has_extra_convs
        self.extra_stage = extra_stage
        self.use_c5 = use_c5
        self.relu_before_extra_convs = relu_before_extra_convs
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm

        self.lateral_convs = []#nn.ModuleList()
        self.lateral_convs_module = nn.ModuleList()
        self.fpn_convs = []#nn.ModuleList()
        self.fpn_convs_module = nn.ModuleList()
        fan = out_channels * 3 * 3

        # stage index 0,1,2,3 stands for res2,res3,res4,res5 on ResNet Backbone
        # 0 <= st_stage < ed_stage <= 3
        st_stage = 4 - len(in_channels)
        ed_stage = st_stage + len(in_channels) - 1
        for i in range(st_stage, ed_stage + 1):
            if i == 3:
                lateral_name = 'fpn_inner_res5_sum'
            else:
                lateral_name = 'fpn_inner_res{}_sum_lateral'.format(i + 2)
            in_c = in_channels[i - st_stage]
            if self.norm_type is not None:
                # self.lateral_convs_module.add_module(
                #     lateral_name,
                #     ConvNormLayer(
                #         ch_in=in_c,
                #         ch_out=out_channels,
                #         filter_size=1,
                #         stride=1,
                #         norm_type=self.norm_type,
                #         norm_decay=self.norm_decay,
                #         freeze_norm=self.freeze_norm,
                #         initializer=None))
                lateral = ConvNormLayer(
                        ch_in=in_c,
                        ch_out=out_channels,
                        filter_size=1,
                        stride=1,
                        norm_type=self.norm_type,
                        norm_decay=self.norm_decay,
                        freeze_norm=self.freeze_norm,
                        initializer=None)
            else:
                # self.lateral_convs_module.add_module(
                #     lateral_name,
                #     nn.Conv2d(
                #         in_channels=in_c,
                #         out_channels=out_channels,
                #         kernel_size=1,
                #     )
                # )
                lateral = nn.Conv2d(
                        in_channels=in_c,
                        out_channels=out_channels,
                        kernel_size=1,
                    )
            self.lateral_convs_module.add_module(lateral_name, lateral)
            self.lateral_convs.append(lateral)

        for i in range(st_stage, ed_stage + 1):
            fpn_name = 'fpn_res{}_sum'.format(i + 2)
            fpn_conv_module = nn.Sequential()
            if self.norm_type is not None:
                # fpn_conv_module.add_module(
                #     fpn_name,
                #     ConvNormLayer(
                #         ch_in=out_channels,
                #         ch_out=out_channels,
                #         filter_size=3,
                #         stride=1,
                #         norm_type=self.norm_type,
                #         norm_decay=self.norm_decay,
                #         freeze_norm=self.freeze_norm,
                #         initializer=None))
                fpn_conv = ConvNormLayer(
                        ch_in=out_channels,
                        ch_out=out_channels,
                        filter_size=3,
                        stride=1,
                        norm_type=self.norm_type,
                        norm_decay=self.norm_decay,
                        freeze_norm=self.freeze_norm,
                        initializer=None)
            else:
                # fpn_conv_module.add_module(
                #     fpn_name,
                #     nn.Conv2d(
                #         in_channels=out_channels,
                #         out_channels=out_channels,
                #         kernel_size=3,
                #         padding=1,
                #         )
                # )
                fpn_conv = nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        )
            self.fpn_convs_module.add_module(fpn_name, fpn_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv levels for RetinaNet(use_c5)/FCOS(use_p5)
        if self.has_extra_convs:
            for i in range(self.extra_stage):
                lvl = ed_stage + 1 + i
                if i == 0 and self.use_c5:
                    in_c = in_channels[-1]
                else:
                    in_c = out_channels
                extra_fpn_name = 'fpn_{}'.format(lvl + 2)
                extra_fpn_conv_module = nn.Sequential()
                if self.norm_type is not None:
                    # extra_fpn_conv_module.add_module(
                    #     extra_fpn_name,
                    #     ConvNormLayer(
                    #         ch_in=in_c,
                    #         ch_out=out_channels,
                    #         filter_size=3,
                    #         stride=2,
                    #         norm_type=self.norm_type,
                    #         norm_decay=self.norm_decay,
                    #         freeze_norm=self.freeze_norm,
                    #         initializer=None))
                    extra_fpn_conv = ConvNormLayer(
                            ch_in=in_c,
                            ch_out=out_channels,
                            filter_size=3,
                            stride=2,
                            norm_type=self.norm_type,
                            norm_decay=self.norm_decay,
                            freeze_norm=self.freeze_norm,
                            initializer=None)
                else:
                    # extra_fpn_conv_module.add_module(
                    #     extra_fpn_name,
                    #     nn.Conv2d(
                    #         in_channels=in_c,
                    #         out_channels=out_channels,
                    #         kernel_size=3,
                    #         stride=2,
                    #         padding=1,
                    #     )
                    # )
                    extra_fpn_conv = nn.Conv2d(
                            in_channels=in_c,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        )

                self.fpn_convs_module.add_module(extra_fpn_name, extra_fpn_conv)
                self.fpn_convs.append(extra_fpn_conv)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'spatial_scales': [1.0 / i.stride for i in input_shape],
        }

    def forward(self, body_feats):
        laterals = []
        num_levels = len(body_feats)

        for i in range(num_levels):
            laterals.append(self.lateral_convs[i](body_feats[i]))

        for i in range(1, num_levels):
            lvl = num_levels - i
            upsample = F.interpolate(
                laterals[lvl],
                scale_factor=2.,
                mode='nearest', )
            laterals[lvl - 1] += upsample

        fpn_output = []
        for lvl in range(num_levels):
            fpn_output.append(self.fpn_convs[lvl](laterals[lvl]))

        if self.extra_stage > 0:
            # use max pool to get more levels on top of outputs (Faster R-CNN, Mask R-CNN)
            if not self.has_extra_convs:
                assert self.extra_stage == 1, 'extra_stage should be 1 if FPN has not extra convs'
                fpn_output.append(torch.max_pool2d(fpn_output[-1], 1, stride=2))
            # add extra conv levels for RetinaNet(use_c5)/FCOS(use_p5)
            else:
                if self.use_c5:
                    extra_source = body_feats[-1]
                else:
                    extra_source = fpn_output[-1]
                fpn_output.append(self.fpn_convs[num_levels](extra_source))

                for i in range(1, self.extra_stage):
                    if self.relu_before_extra_convs:
                        fpn_output.append(self.fpn_convs[num_levels + i](F.relu(
                            fpn_output[-1])))
                    else:
                        fpn_output.append(self.fpn_convs[num_levels + i](
                            fpn_output[-1]))
        return fpn_output
