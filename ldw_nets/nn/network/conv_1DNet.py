import logging

import torch
import torch.nn as nn
from ldw_nets.nn.network.resNet import ResNet1D



__all__ = (
    "Conv_1DNet",
)

class Conv_1DNet(torch.nn.Module):

    def __init__(self, backbone="resNet18", relu_type="swish", a_upsample_ratio=1):
        super(Conv_1DNet, self).__init__()

        if backbone=="ResNet18":
            self.trunk = ResNet1D('ResidualBlock_1D', [2, 2, 2, 2], relu_type=relu_type)
        elif backbone=="ResNet34":
            self.trunk = ResNet1D('ResidualBlock_1D', [3, 4, 6, 3], relu_type=relu_type)
        elif backbone=="ResNet50":
            self.trunk = ResNet1D('ResidualBlock131_1D', [3, 4, 6, 3], relu_type=relu_type)
        elif backbone=="ResNet101":
            self.trunk = ResNet1D('ResidualBlock131_1D', [3, 4, 24, 3], relu_type=relu_type)
        elif backbone=="ResNet152":
            self.trunk = ResNet1D('ResidualBlock131_1D', [3, 8, 36, 3], relu_type=relu_type)
        else:
            logging.info("unk ResNet in Conv_1DResNet!!!!")


    def forward(self, xs_pad):
        B, T, C = xs_pad.size()
        xs_pad = xs_pad[:, : T // 640 * 640, :]
        xs_pad = xs_pad.transpose(1, 2).contiguous()
        xs_pad = self.trunk(xs_pad)
        return xs_pad.transpose(1, 2).contiguous()



