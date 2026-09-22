import logging

import torch
import torch.nn as nn
from ldw_nets.nn.unit import tensor3Dto2D, Swish
from ldw_nets.nn.network.resNet import ResNet
from ldw_nets.nn.block import ResidualBlock, ResidualBlock131


__all__ = (
    "Conv3D_2DNet",
)

class Conv3D_2DNet(torch.nn.Module):

    def __init__(self, backbone="resNet18", relu_type="swish"):
        super(Conv3D_2DNet, self).__init__()
        self.frontend_nout = 64

        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1, self.frontend_nout, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False
            ),
            nn.BatchNorm3d(self.frontend_nout),
            Swish(),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
        )

        self.threeD_to_2D_tensor = tensor3Dto2D()

        if backbone=="ResNet18":
            self.trunk = ResNet('ResidualBlock', [2, 2, 2, 2], relu_type=relu_type)
        elif backbone=="ResNet34":
            self.trunk = ResNet('ResidualBlock', [3, 4, 6, 3], relu_type=relu_type)
        elif backbone=="ResNet50":
            self.trunk = ResNet('ResidualBlock131', [3, 4, 6, 3], relu_type=relu_type)
        elif backbone=="ResNet101":
            self.trunk = ResNet('ResidualBlock131', [3, 4, 24, 3], relu_type=relu_type)
        elif backbone=="ResNet152":
            self.trunk = ResNet('ResidualBlock131', [3, 8, 36, 3], relu_type=relu_type)
        else:
            logging.info("unk ResNet in Conv3D_2DResNet!!!!")


    def forward(self, xs_pad):
        xs_pad = xs_pad.transpose(1, 2)  # [B, T, C, H, W] -> [B, C, T, H, W]

        B, C, T, H, W = xs_pad.size()

        xs_pad = self.frontend3D(xs_pad)  # kakaaaaa
        Tnew = xs_pad.shape[2]
        xs_pad = self.threeD_to_2D_tensor(xs_pad)
        xs_pad = self.trunk(xs_pad)
        return xs_pad.view(B, Tnew, xs_pad.size(1))



