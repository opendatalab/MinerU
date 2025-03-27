import torch
import torch.nn as nn

from pytorchocr.modeling.necks.rnn import Im2Seq, SequenceEncoder
from .rec_nrtr_head import Transformer
from .rec_ctc_head import CTCHead
from .rec_sar_head import SARHead

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
        super().__init__()
        self.head_list = kwargs.pop('head_list')

        self.gtc_head = 'sar'
        assert len(self.head_list) >= 2
        for idx, head_name in enumerate(self.head_list):
            name = list(head_name)[0]
            if name == 'SARHead':
                pass
                # # sar head
                # sar_args = self.head_list[idx][name]
                # self.sar_head = eval(name)(in_channels=in_channels, \
                #                            out_channels=out_channels_list['SARLabelDecode'], **sar_args)
            elif name == 'NRTRHead':
                pass
                # gtc_args = self.head_list[idx][name]
                # max_text_length = gtc_args.get('max_text_length', 25)
                # nrtr_dim = gtc_args.get('nrtr_dim', 256)
                # num_decoder_layers = gtc_args.get('num_decoder_layers', 4)
                # self.before_gtc = nn.Sequential(
                #     nn.Flatten(2), FCTranspose(in_channels, nrtr_dim))
                # self.gtc_head = Transformer(
                #     d_model=nrtr_dim,
                #     nhead=nrtr_dim // 32,
                #     num_encoder_layers=-1,
                #     beam_size=-1,
                #     num_decoder_layers=num_decoder_layers,
                #     max_len=max_text_length,
                #     dim_feedforward=nrtr_dim * 4,
                #     out_channels=out_channels_list['NRTRLabelDecode'])
            elif name == 'CTCHead':
                # ctc neck
                self.encoder_reshape = Im2Seq(in_channels)
                neck_args = self.head_list[idx][name]['Neck']
                encoder_type = neck_args.pop('name')
                self.ctc_encoder = SequenceEncoder(in_channels=in_channels, \
                                                   encoder_type=encoder_type, **neck_args)
                # ctc head
                head_args = self.head_list[idx][name].get('Head', {})
                if head_args is None:
                    head_args = {}
                self.ctc_head = eval(name)(in_channels=self.ctc_encoder.out_channels, \
                                           out_channels=out_channels_list['CTCLabelDecode'], **head_args)
            else:
                raise NotImplementedError(
                    '{} is not supported in MultiHead yet'.format(name))

    def forward(self, x, data=None):
        ctc_encoder = self.ctc_encoder(x)
        ctc_out = self.ctc_head(ctc_encoder)
        head_out = dict()
        head_out['ctc'] = ctc_out
        head_out['res'] = ctc_out
        head_out['ctc_neck'] = ctc_encoder
        # eval mode
        if not self.training:
            return ctc_out
        if self.gtc_head == 'sar':
            sar_out = self.sar_head(x, data[1:])['res']
            head_out['sar'] = sar_out
        else:
            gtc_out = self.gtc_head(self.before_gtc(x), data[1:])['res']
            head_out['nrtr'] = gtc_out
        return head_out
