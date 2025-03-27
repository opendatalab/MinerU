from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchocr.modeling.common import Activation
# import paddle
# from paddle import nn, ParamAttr
# from paddle.nn import functional as F
import numpy as np


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        bn_name = "bn_" + name
        self.bn = nn.BatchNorm2d(
            out_channels, )
        self.act = act
        if act is not None:
            self._act = Activation(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self._act(x)
        return x


class LocalizationNetwork(nn.Module):
    def __init__(self, in_channels, num_fiducial, loc_lr, model_name):
        super(LocalizationNetwork, self).__init__()
        self.F = num_fiducial
        F = num_fiducial
        if model_name == "large":
            num_filters_list = [64, 128, 256, 512]
            fc_dim = 256
        else:
            num_filters_list = [16, 32, 64, 128]
            fc_dim = 64

        # self.block_list = []
        self.block_list = nn.Sequential()
        for fno in range(0, len(num_filters_list)):
            num_filters = num_filters_list[fno]
            name = "loc_conv%d" % fno
            # conv = self.add_sublayer(
            #     name,
            #     ConvBNLayer(
            #         in_channels=in_channels,
            #         out_channels=num_filters,
            #         kernel_size=3,
            #         act='relu',
            #         name=name))
            conv = ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=3,
                    act='relu',
                    name=name)
            # self.block_list.append(conv)
            self.block_list.add_module(name, conv)
            if fno == len(num_filters_list) - 1:
                pool = nn.AdaptiveAvgPool2d(1)
            else:
                # pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
                pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            in_channels = num_filters
            # self.block_list.append(pool)
            self.block_list.add_module('{}_pool'.format(name), pool)
        name = "loc_fc1"
        stdv = 1.0 / math.sqrt(num_filters_list[-1] * 1.0)
        self.fc1 = nn.Linear(
            in_channels,
            fc_dim,
            bias=True,
        )


        # Init fc2 in LocalizationNetwork
        initial_bias = self.get_initial_fiducials()
        initial_bias = initial_bias.reshape(-1)
        name = "loc_fc2"
        self.fc2 = nn.Linear(
            fc_dim,
            F * 2,
            bias=True
        )
        self.out_channels = F * 2

    def forward(self, x):
        """
           Estimating parameters of geometric transformation
           Args:
               image: input
           Return:
               batch_C_prime: the matrix of the geometric transformation
        """
        B = x.shape[0]
        i = 0
        for block in self.block_list:
            x = block(x)
        x = x.squeeze(dim=2).squeeze(dim=2)
        x = self.fc1(x)

        x = F.relu(x)
        x = self.fc2(x)
        x = x.reshape(shape=[-1, self.F, 2])
        return x

    def get_initial_fiducials(self):
        """ see RARE paper Fig. 6 (a) """
        F = self.F
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return initial_bias


class GridGenerator(nn.Module):
    def __init__(self, in_channels, num_fiducial):
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.F = num_fiducial

        name = "ex_fc"
        self.fc = nn.Linear(
            in_channels,
            6,
            bias=True
        )

    def forward(self, batch_C_prime, I_r_size):
        """
        Generate the grid for the grid_sampler.
        Args:
            batch_C_prime: the matrix of the geometric transformation
            I_r_size: the shape of the input image
        Return:
            batch_P_prime: the grid for the grid_sampler
        """
        C = self.build_C_paddle()
        P = self.build_P_paddle(I_r_size)

        inv_delta_C_tensor = self.build_inv_delta_C_paddle(C).type(torch.float32)
        P_hat_tensor = self.build_P_hat_paddle(
            C, torch.as_tensor(P)).type(torch.float32)

        inv_delta_C_tensor.stop_gradient = True
        P_hat_tensor.stop_gradient = True

        batch_C_ex_part_tensor = self.get_expand_tensor(batch_C_prime)

        batch_C_ex_part_tensor.stop_gradient = True

        batch_C_prime_with_zeros = torch.cat(
            [batch_C_prime, batch_C_ex_part_tensor], dim=1)
        inv_delta_C_tensor = inv_delta_C_tensor.to(batch_C_prime_with_zeros.device)
        batch_T = torch.matmul(inv_delta_C_tensor, batch_C_prime_with_zeros)
        P_hat_tensor = P_hat_tensor.to(batch_T.device)
        batch_P_prime = torch.matmul(P_hat_tensor, batch_T)
        return batch_P_prime

    def build_C_paddle(self):
        """ Return coordinates of fiducial points in I_r; C """
        F = self.F
        ctrl_pts_x = torch.linspace(-1.0, 1.0, int(F / 2), dtype=torch.float64)
        ctrl_pts_y_top = -1 * torch.ones([int(F / 2)], dtype=torch.float64)
        ctrl_pts_y_bottom = torch.ones([int(F / 2)], dtype=torch.float64)
        ctrl_pts_top = torch.stack([ctrl_pts_x, ctrl_pts_y_top], dim=1)
        ctrl_pts_bottom = torch.stack([ctrl_pts_x, ctrl_pts_y_bottom], dim=1)
        C = torch.cat([ctrl_pts_top, ctrl_pts_bottom], dim=0)
        return C  # F x 2

    def build_P_paddle(self, I_r_size):
        I_r_height, I_r_width = I_r_size
        I_r_grid_x = (torch.arange(
            -I_r_width, I_r_width, 2, dtype=torch.float64) + 1.0
                      ) / torch.as_tensor(np.array([I_r_width]).astype(np.float64))

        I_r_grid_y = (torch.arange(
            -I_r_height, I_r_height, 2, dtype=torch.float64) + 1.0
                      ) / torch.as_tensor(np.array([I_r_height]).astype(np.float64))

        # P: self.I_r_width x self.I_r_height x 2
        P = torch.stack(torch.meshgrid([I_r_grid_x, I_r_grid_y]), dim=2)
        # P = paddle.transpose(P, perm=[1, 0, 2])
        P = P.permute(1, 0, 2)
        # n (= self.I_r_width x self.I_r_height) x 2
        return P.reshape([-1, 2])

    def build_inv_delta_C_paddle(self, C):
        """ Return inv_delta_C which is needed to calculate T """
        F = self.F
        hat_C = torch.zeros((F, F), dtype=torch.float64)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                if i == j:
                    hat_C[i, j] = 1
                else:
                    r = torch.norm(C[i] - C[j])
                    hat_C[i, j] = r
                    hat_C[j, i] = r
        hat_C = (hat_C**2) * torch.log(hat_C)
        delta_C = torch.cat(  # F+3 x F+3
            [
                torch.cat(
                    [torch.ones(
                        (F, 1), dtype=torch.float64), C, hat_C], dim=1),  # F x F+3
                torch.cat(
                    [
                        torch.zeros(
                            (2, 3), dtype=torch.float64), C.permute(1,0)
                    ],
                    dim=1),  # 2 x F+3
                torch.cat(
                    [
                        torch.zeros(
                            (1, 3), dtype=torch.float64), torch.ones(
                                (1, F), dtype=torch.float64)
                    ],
                    dim=1)  # 1 x F+3
            ],
            dim=0)
        inv_delta_C = torch.inverse(delta_C)
        return inv_delta_C  # F+3 x F+3

    def build_P_hat_paddle(self, C, P):
        F = self.F
        eps = self.eps
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        # P_tile: n x 2 -> n x 1 x 2 -> n x F x 2
        # P_tile = paddle.tile(paddle.unsqueeze(P, axis=1), (1, F, 1))
        P_tile = torch.unsqueeze(P, dim=1).repeat(1, F, 1)
        C_tile = torch.unsqueeze(C, dim=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        # rbf_norm: n x F
        rbf_norm = torch.norm(P_diff, p=2, dim=2, keepdim=False)

        # rbf: n x F
        # rbf = torch.mul(
        #     torch.square(rbf_norm), torch.log(rbf_norm + eps))
        rbf = torch.mul(
            rbf_norm**2, torch.log(rbf_norm + eps))
        P_hat = torch.cat(
            [torch.ones(
                (n, 1), dtype=torch.float64), P, rbf], dim=1)
        return P_hat  # n x F+3

    def get_expand_tensor(self, batch_C_prime):
        B, H, C = batch_C_prime.shape
        batch_C_prime = batch_C_prime.reshape([B, H * C])
        batch_C_ex_part_tensor = self.fc(batch_C_prime)
        batch_C_ex_part_tensor = batch_C_ex_part_tensor.reshape([-1, 3, 2])
        return batch_C_ex_part_tensor


class TPS(nn.Module):
    def __init__(self, in_channels, num_fiducial, loc_lr, model_name):
        super(TPS, self).__init__()
        self.loc_net = LocalizationNetwork(in_channels, num_fiducial, loc_lr,
                                           model_name)
        self.grid_generator = GridGenerator(self.loc_net.out_channels,
                                            num_fiducial)
        self.out_channels = in_channels

    def forward(self, image):
        image.stop_gradient = False
        batch_C_prime = self.loc_net(image)
        batch_P_prime = self.grid_generator(batch_C_prime, image.shape[2:])
        batch_P_prime = batch_P_prime.reshape(
            [-1, image.shape[2], image.shape[3], 2])
        if torch.__version__ < '1.3.0':
            batch_I_r = F.grid_sample(image, grid=batch_P_prime)
        else:
            batch_I_r = F.grid_sample(image, grid=batch_P_prime, align_corners=True)
        return batch_I_r
