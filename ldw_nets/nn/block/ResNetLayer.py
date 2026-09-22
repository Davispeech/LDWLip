import torch.nn as nn
from .basicblock import ResidualBlock

__all__ = (
    "ResNetBlockLayer",
)


class ResNetBlockLayer(nn.Module):
    def __init__(self, block="ResidualBlock", inplanes=64, planes=64, blocks=2, stride=1, relu_type='swish'):
        super(ResNetBlockLayer, self).__init__()
        self.block_expansion = 1  # 你可以根据 block.expansion 来设置
        self.inplanes = inplanes
        block = globals()[block]

        if stride != 1 or inplanes != planes * self.block_expansion:
            from ldw_nets.nn.block import DownsampleCB
            downsample = DownsampleCB(
                inplanes=inplanes,
                outplanes=planes * self.block_expansion,
                stride=stride,
            )
        else:
            downsample = None

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, relu_type=relu_type))
        inplanes = planes * self.block_expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, relu_type=relu_type))

        self.layer = nn.Sequential(*layers)
        self.outplanes = inplanes  # for chaining layers

    def forward(self, x):
        return self.layer(x)

