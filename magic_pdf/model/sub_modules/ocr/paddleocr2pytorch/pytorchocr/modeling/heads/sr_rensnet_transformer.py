# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from:
https://github.com/FudanVI/FudanOCR/blob/main/text-gestalt/loss/transformer_english_decomposition.py
"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def subsequent_mask(size):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = torch.ones(1, size, size, dtype=torch.float32)
    mask_inf = torch.triu(
        torch.full(
            size=[1, size, size], fill_value=-np.inf, dtype=torch.float32),
        diagonal=1)
    mask = mask + mask_inf
    padding_mask = torch.equal(mask, torch.Tensor(1).type(mask.dtype))
    return padding_mask



def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None, attention_map=None):
    d_k = query.shape[-1]
    scores = torch.matmul(query,
                           key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    else:
        pass

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.compress_attention = compress_attention
        self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, attention_map=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, attention_map = attention(
            query,
            key,
            value,
            mask=mask,
            dropout=self.dropout,
            attention_map=attention_map)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x), attention_map


class ResNet(nn.Module):
    def __init__(self, num_in, block, layers):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(num_in, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2, 2), (2, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.layer1_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer1 = self._make_layer(block, 128, 256, layers[0])
        self.layer1_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer1_bn = nn.BatchNorm2d(256)
        self.layer1_relu = nn.ReLU(inplace=True)

        self.layer2_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer2 = self._make_layer(block, 256, 256, layers[1])
        self.layer2_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer2_bn = nn.BatchNorm2d(256)
        self.layer2_relu = nn.ReLU(inplace=True)

        self.layer3_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer3 = self._make_layer(block, 256, 512, layers[2])
        self.layer3_conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.layer3_bn = nn.BatchNorm2d(512)
        self.layer3_relu = nn.ReLU(inplace=True)

        self.layer4_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer4 = self._make_layer(block, 512, 512, layers[3])
        self.layer4_conv2 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.layer4_conv2_bn = nn.BatchNorm2d(1024)
        self.layer4_conv2_relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, inplanes, planes, blocks):

        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 3, 1, 1),
                nn.BatchNorm2d(
                    planes), )
        else:
            downsample = None
        layers = []
        layers.append(block(inplanes, planes, downsample))
        for i in range(1, blocks):
            layers.append(block(planes, planes, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.layer1_pool(x)
        x = self.layer1(x)
        x = self.layer1_conv(x)
        x = self.layer1_bn(x)
        x = self.layer1_relu(x)

        x = self.layer2(x)
        x = self.layer2_conv(x)
        x = self.layer2_bn(x)
        x = self.layer2_relu(x)

        x = self.layer3(x)
        x = self.layer3_conv(x)
        x = self.layer3_bn(x)
        x = self.layer3_relu(x)

        x = self.layer4(x)
        x = self.layer4_conv2(x)
        x = self.layer4_conv2_bn(x)
        x = self.layer4_conv2_relu(x)

        return x


class Bottleneck(nn.Module):
    def __init__(self, input_dim):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, 1)
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(input_dim, input_dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(input_dim)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() *
            (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.proj(x)
        return out


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embed = self.lut(x) * math.sqrt(self.d_model)
        return embed


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.parameter.Parameter(torch.ones(features))
        self.b_2 = nn.parameter.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.mask_multihead = MultiHeadedAttention(
            h=16, d_model=1024, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(1024)

        self.multihead = MultiHeadedAttention(h=16, d_model=1024, dropout=0.1)
        self.mul_layernorm2 = LayerNorm(1024)

        self.pff = PositionwiseFeedForward(1024, 2048)
        self.mul_layernorm3 = LayerNorm(1024)

    def forward(self, text, conv_feature, attention_map=None):
        text_max_length = text.shape[1]
        mask = subsequent_mask(text_max_length)
        result = text
        result = self.mul_layernorm1(result + self.mask_multihead(
            text, text, text, mask=mask)[0])
        b, c, h, w = conv_feature.shape
        conv_feature = conv_feature.view(b, c, h * w).permute(0, 2, 1).contiguous()
        word_image_align, attention_map = self.multihead(
            result,
            conv_feature,
            conv_feature,
            mask=None,
            attention_map=attention_map)
        result = self.mul_layernorm2(result + word_image_align)
        result = self.mul_layernorm3(result + self.pff(result))

        return result, attention_map


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample != None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.cnn = ResNet(num_in=1, block=BasicBlock, layers=[1, 2, 5, 3])

    def forward(self, input):
        conv_result = self.cnn(input)
        return conv_result


class Transformer(nn.Module):
    def __init__(self, in_channels=1, alphabet='0123456789'):
        super(Transformer, self).__init__()
        self.alphabet = alphabet
        word_n_class = self.get_alphabet_len()
        self.embedding_word_with_upperword = Embeddings(512, word_n_class)
        self.pe = PositionalEncoding(dim=512, dropout=0.1, max_len=5000)

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.generator_word_with_upperword = Generator(1024, word_n_class)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_alphabet_len(self):
        return len(self.alphabet)

    def forward(self, image, text_length, text_input, attention_map=None):
        if image.shape[1] == 3:
            R = image[:, 0:1, :, :]
            G = image[:, 1:2, :, :]
            B = image[:, 2:3, :, :]
            image = 0.299 * R + 0.587 * G + 0.114 * B

        conv_feature = self.encoder(image)  # batch, 1024, 8, 32
        max_length = max(text_length)
        text_input = text_input[:, :max_length]

        text_embedding = self.embedding_word_with_upperword(
            text_input)  # batch, text_max_length, 512
        if torch.cuda.is_available():
            postion_embedding = self.pe(
                torch.zeros(text_embedding.shape).cuda()).cuda()
        else:
            postion_embedding = self.pe(
                torch.zeros(text_embedding.shape))  # batch, text_max_length, 512
        text_input_with_pe = torch.cat([text_embedding, postion_embedding], 2)  # batch, text_max_length, 1024
        batch, seq_len, _ = text_input_with_pe.shape

        text_input_with_pe, word_attention_map = self.decoder(
            text_input_with_pe, conv_feature)

        word_decoder_result = self.generator_word_with_upperword(
            text_input_with_pe)

        if self.training:
            total_length = torch.sum(text_length).data
            probs_res = torch.zeros([total_length, self.get_alphabet_len()]).type_as(word_decoder_result.data)
            start = 0

            for index, length in enumerate(text_length):
                length = int(length.numpy())
                probs_res[start:start + length, :] = word_decoder_result[
                    index, 0:0 + length, :]

                start = start + length

            return probs_res, word_attention_map, None
        else:
            return word_decoder_result
