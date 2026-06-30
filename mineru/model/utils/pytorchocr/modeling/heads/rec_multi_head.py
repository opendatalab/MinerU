# Copyright (c) Opendatalab. All rights reserved.
import torch.nn.functional as F
from torch import nn

from ..necks.rnn import EncoderWithLightSVTR, Im2Seq, SequenceEncoder
from .rec_ctc_head import CTCHead


class FCTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, only_transpose=False):
        super().__init__()
        self.only_transpose = only_transpose
        if not self.only_transpose:
            self.fc = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        if self.only_transpose:
            return x.permute([0, 2, 1])
        else:
            return self.fc(x.permute([0, 2, 1]))


class MultiHead(nn.Module):
    def __init__(self, in_channels, out_channels_list, **kwargs):
        """初始化多头识别 Head，v6 LightSVTR 分支使用 HF safetensors 命名。"""
        super().__init__()
        self.head_list = kwargs.pop("head_list")
        self.use_light_svtr_head = False

        self.gtc_head = "sar"
        assert len(self.head_list) >= 2
        for idx, head_name in enumerate(self.head_list):
            name = list(head_name)[0]
            if name == "SARHead":
                pass

            elif name == "NRTRHead":
                pass
            elif name == "CTCHead":
                # ctc neck
                self.encoder_reshape = Im2Seq(in_channels)
                neck_args = self.head_list[idx][name]["Neck"]
                encoder_type = neck_args.pop("name")
                if encoder_type == "lightsvtr":
                    # v6 safetensors 中 CTC 分支直接命名为 head.encoder/head.head。
                    self.encoder = EncoderWithLightSVTR(in_channels=in_channels, **neck_args)
                    self.head = nn.Linear(self.encoder.out_channels, out_channels_list["CTCLabelDecode"], bias=True)
                    self.out_channels = out_channels_list["CTCLabelDecode"]
                    self.use_light_svtr_head = True
                else:
                    self.ctc_encoder = SequenceEncoder(
                        in_channels=in_channels, encoder_type=encoder_type, **neck_args
                    )
                    # ctc head
                    head_args = self.head_list[idx][name].get("Head", {})
                    if head_args is None:
                        head_args = {}

                    self.ctc_head = CTCHead(
                        in_channels=self.ctc_encoder.out_channels,
                        out_channels=out_channels_list["CTCLabelDecode"],
                        **head_args,
                    )
            else:
                raise NotImplementedError(f"{name} is not supported in MultiHead yet")

    def forward(self, x, data=None):
        """根据配置执行 v6 LightSVTR 或历史 CTC 分支。"""
        if self.use_light_svtr_head:
            ctc_encoder = self.encoder(x)
            ctc_encoder = ctc_encoder.squeeze(dim=2).permute(0, 2, 1)
            predicts = self.head(ctc_encoder)
            if not self.training:
                predicts = F.softmax(predicts, dim=2)
            return predicts
        ctc_encoder = self.ctc_encoder(x)
        return self.ctc_head(ctc_encoder)
