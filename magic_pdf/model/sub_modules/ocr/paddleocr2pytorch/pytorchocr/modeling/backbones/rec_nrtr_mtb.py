
import torch
from torch import nn


class MTB(nn.Module):
    def __init__(self, cnn_num, in_channels):
        super(MTB, self).__init__()
        self.block = nn.Sequential()
        self.out_channels = in_channels
        self.cnn_num = cnn_num
        if self.cnn_num == 2:
            for i in range(self.cnn_num):
                self.block.add_module(
                    'conv_{}'.format(i),
                    nn.Conv2d(
                        in_channels=in_channels
                        if i == 0 else 32 * (2**(i - 1)),
                        out_channels=32 * (2**i),
                        kernel_size=3,
                        stride=2,
                        padding=1))
                self.block.add_module('relu_{}'.format(i), nn.ReLU())
                self.block.add_module('bn_{}'.format(i),
                                        nn.BatchNorm2d(32 * (2**i)))


    def forward(self, images):
        x = self.block(images)
        if self.cnn_num == 2:
            # (b, w, h, c)
            x = x.permute(0, 3, 2, 1)
            x_shape = x.shape
            x = torch.reshape(
                x, (x_shape[0], x_shape[1], x_shape[2] * x_shape[3]))
        return x
