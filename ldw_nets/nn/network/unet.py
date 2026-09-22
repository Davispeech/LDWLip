import ast

import torch.nn as nn


__all__ = (
    "UNet1D",
)

class UNet1D(nn.Module):
    def __init__(
        self,
        block="ResidualBlock",
        layers="[2, 2, 2, 2]",
        relu_type="swish",
    ):
        super(UNet1D, self).__init__()
        self.inplanes = 64
        self.relu_type = relu_type
        from ldw_nets.nn.block.basicblock import DownsampleCB
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
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes,
                outplanes=planes * block.expansion,
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
        self.inplanes = planes * block.expansion
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
        :param x: torch.Tensor, input tensor with input size (B, C, T, H, W).
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


