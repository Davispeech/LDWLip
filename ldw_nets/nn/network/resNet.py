import ast

import torch.nn as nn
from ldw_nets.nn.unit import Swish
from ldw_nets.nn.block import (
    Conv,
    Conv3Dfrontend,
    DownsampleCB,
    DownsampleCB_1D,
    ResidualBlock,
    ResidualBlock131,
    ResidualBlock_1D,
    ResidualBlock131_1D,
    #
    PositionwiseFeedForward,
    PositionalEncoding,
    RelPositionalEncoding,
    embed,
    ConformerLayer,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
    #
    SwinTransformerBlockR,
)


__all__ = (
    "ResNet",
    "ResNet1D",
)


class ResNet1D(nn.Module):
    def __init__(
        self,
        block="ResidualBlock_1D",
        layers="[2, 2, 2, 2]",
        relu_type="swish",
        a_upsample_ratio=1,
    ):
        """__init__.

        :param block: torch.nn.Module, class of blocks.
        :param layers: List, customised layers in each block.
        :param relu_type: str, type of activation function.
        :param a_upsample_ratio: int, The ratio related to the \
            temporal resolution of output features of the frontend. \
            a_upsample_ratio=1 produce features with a fps of 25.
        """
        super(ResNet1D, self).__init__()
        self.inplanes = 64
        self.relu_type = relu_type
        self.downsample_block = DownsampleCB_1D
        self.a_upsample_ratio = a_upsample_ratio
        block = globals()[block]
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=self.inplanes,
            kernel_size=80,
            stride=4,
            padding=38,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(self.inplanes)

        if relu_type == "relu":
            self.relu = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu = nn.PReLU(num_parameters=self.inplanes)
        elif relu_type == "swish":
            self.relu = Swish()

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(
            kernel_size=20 // self.a_upsample_ratio,
            stride=20 // self.a_upsample_ratio,
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        """_make_layer.

        :param block: torch.nn.Module, class of blocks.
        :param planes: int,  number of channels produced by the convolution.
        :param blocks: int, number of layers in a block.
        :param stride: int, size of the convolving kernel.
        """
        block_expansion = 1
        downsample = None
        if stride != 1 or self.inplanes != planes * block_expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes,
                outplanes=planes * block_expansion,
                stride=stride,
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                relu_type=self.relu_type,
            )
        )
        self.inplanes = planes * block_expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    relu_type=self.relu_type,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        """forward.

        :param x: torch.Tensor, input tensor with input size (B, C, T)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x


class ResNet(nn.Module):
    def __init__(
        self,
        block="ResidualBlock",
        layers="[2, 2, 2, 2]",
        relu_type="swish",
    ):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.relu_type = relu_type
        self.downsample_block = DownsampleCB
        block = globals()[block]
        if not isinstance(layers, list):
            layers = ast.literal_eval(layers)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, blocks, stride=1):
        """_make_layer.
        :param block: torch.nn.Module, class of blocks.
        :param planes: int,  number of channels produced by the convolution.
        :param blocks: int, number of layers in a block.
        :param stride: int, size of the convolving kernel.
        """
        block_expansion = 1
        downsample = None
        if stride != 1 or self.inplanes != planes * block_expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes,
                outplanes=planes * block_expansion,
                stride=stride,
            )

        layers=[]
        layers.append(block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type,))
        self.inplanes = planes * block_expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type,))

        return nn.Sequential(*layers)

    def forward(self, x):
        """forward.
        :param x: torch.Tensor, input tensor with input size (B, C, T, H, W).
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


