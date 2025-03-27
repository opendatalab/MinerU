from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TableAttentionHead(nn.Module):
    def __init__(self, in_channels, hidden_size, loc_type, in_max_len=488, **kwargs):
        super(TableAttentionHead, self).__init__()
        self.input_size = in_channels[-1]
        self.hidden_size = hidden_size
        self.elem_num = 30
        self.max_text_length = 100
        self.max_elem_length = kwargs.get('max_elem_length', 500)
        self.max_cell_num = 500

        self.structure_attention_cell = AttentionGRUCell(
            self.input_size, hidden_size, self.elem_num, use_gru=False)
        self.structure_generator = nn.Linear(hidden_size, self.elem_num)
        self.loc_type = loc_type
        self.in_max_len = in_max_len

        if self.loc_type == 1:
            self.loc_generator = nn.Linear(hidden_size, 4)
        else:
            if self.in_max_len == 640:
                self.loc_fea_trans = nn.Linear(400, self.max_elem_length + 1)
            elif self.in_max_len == 800:
                self.loc_fea_trans = nn.Linear(625, self.max_elem_length + 1)
            else:
                self.loc_fea_trans = nn.Linear(256, self.max_elem_length + 1)
            self.loc_generator = nn.Linear(self.input_size + hidden_size, 4)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char.type(torch.int64), onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None):
        # if and else branch are both needed when you want to assign a variable
        # if you modify the var in just one branch, then the modification will not work.
        fea = inputs[-1]
        if len(fea.shape) == 3:
            pass
        else:
            last_shape = int(np.prod(fea.shape[2:]))  # gry added
            fea = torch.reshape(fea, [fea.shape[0], fea.shape[1], last_shape])
            # fea = fea.transpose([0, 2, 1])  # (NTC)(batch, width, channels)
            fea = fea.permute(0, 2, 1)
        batch_size = fea.shape[0]

        hidden = torch.zeros((batch_size, self.hidden_size))
        output_hiddens = []
        if self.training and targets is not None:
            raise NotImplementedError
        else:
            temp_elem = torch.zeros([batch_size], dtype=torch.int32)
            structure_probs = None
            loc_preds = None
            elem_onehots = None
            outputs = None
            alpha = None
            max_elem_length = torch.as_tensor(self.max_elem_length)
            i = 0
            while i < max_elem_length + 1:
                elem_onehots = self._char_to_onehot(
                    temp_elem, onehot_dim=self.elem_num)
                (outputs, hidden), alpha = self.structure_attention_cell(
                    hidden, fea, elem_onehots)
                output_hiddens.append(torch.unsqueeze(outputs, dim=1))
                structure_probs_step = self.structure_generator(outputs)
                temp_elem = structure_probs_step.argmax(dim=1, keepdim=False)
                i += 1

            output = torch.cat(output_hiddens, dim=1)
            structure_probs = self.structure_generator(output)
            structure_probs = F.softmax(structure_probs, dim=-1)
            if self.loc_type == 1:
                loc_preds = self.loc_generator(output)
                loc_preds = F.sigmoid(loc_preds)
            else:
                loc_fea = fea.permute(0, 2, 1)
                loc_fea = self.loc_fea_trans(loc_fea)
                loc_fea = loc_fea.permute(0, 2, 1)
                loc_concat = torch.cat([output, loc_fea], dim=2)
                loc_preds = self.loc_generator(loc_concat)
                loc_preds = F.sigmoid(loc_preds)
        return {'structure_probs': structure_probs, 'loc_preds': loc_preds}


class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionGRUCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(
            input_size=input_size + num_embeddings, hidden_size=hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden), dim=1)
        res = torch.add(batch_H_proj, prev_hidden_proj)
        res = torch.tanh(res)
        e = self.score(res)
        alpha = F.softmax(e, dim=1)
        alpha = alpha.permute(0, 2, 1)
        context = torch.squeeze(torch.matmul(alpha, batch_H), dim=1)
        concat_context = torch.cat([context, char_onehots.float()], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return (cur_hidden, cur_hidden), alpha


class AttentionLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(AttentionLSTM, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        self.attention_cell = AttentionLSTMCell(
            in_channels, hidden_size, out_channels, use_gru=False)
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, batch_max_length=25):
        batch_size = inputs.shape[0]
        num_steps = batch_max_length

        hidden = (torch.zeros((batch_size, self.hidden_size)), torch.zeros(
            (batch_size, self.hidden_size)))
        output_hiddens = []

        if targets is not None:
            for i in range(num_steps):
                # one-hot vectors for a i-th char
                char_onehots = self._char_to_onehot(
                    targets[:, i], onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs,
                                                    char_onehots)

                hidden = (hidden[1][0], hidden[1][1])
                output_hiddens.append(torch.unsqueeze(hidden[0], dim=1))
            output = torch.cat(output_hiddens, dim=1)
            probs = self.generator(output)

        else:
            targets = torch.zeros([batch_size], dtype=torch.int32)
            probs = None

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs,
                                                    char_onehots)
                probs_step = self.generator(hidden[0])
                hidden = (hidden[1][0], hidden[1][1])
                if probs is None:
                    probs = torch.unsqueeze(probs_step, dim=1)
                else:
                    probs = torch.cat(
                        [probs, torch.unsqueeze(
                            probs_step, dim=1)], dim=1)

                next_input = probs_step.argmax(dim=1)

                targets = next_input

        return probs


class AttentionLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionLSTMCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        if not use_gru:
            self.rnn = nn.LSTMCell(
                input_size=input_size + num_embeddings, hidden_size=hidden_size)
        else:
            self.rnn = nn.GRUCell(
                input_size=input_size + num_embeddings, hidden_size=hidden_size)

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden[0]), dim=1)
        res = torch.add(batch_H_proj, prev_hidden_proj)
        res = torch.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, dim=1)
        alpha = alpha.permute(0, 2, 1)
        context = torch.squeeze(torch.matmul(alpha, batch_H), dim=1)
        concat_context = torch.cat([context, char_onehots.float()], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)

        return (cur_hidden, cur_hidden), alpha