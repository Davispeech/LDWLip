import torch
from torch import nn
import torch.nn.functional as F


__all__ = (
    "MobileNetV3Large",
    "MobileNetV3LargeFADC",
    "MobileNetV3LargeFDAM",
    "MobileNetV3LargeFDConv",
    "MobileNetV3LargeFreqFusion",
)


def _make_divisible(v, divisor=8, min_value=None):
    # 确保通道数为8的倍数
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6

class SEBlock(nn.Module):
    def __init__(self, in_ch, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch, bias=False),
            HSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MobileBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, exp_size, use_se, use_hs, basemode='Conv'):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

        # 修改这里：HSwish不带inplace参数，ReLU带
        if use_hs:
            activation = HSwish()
        else:
            activation = nn.ReLU(inplace=True)
        if basemode=="FADC" and kernel_size==3:
            from .FADC import AdaptiveDilatedConv
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, exp_size, 1, bias=False),
                nn.BatchNorm2d(exp_size),
                activation,
                AdaptiveDilatedConv(
                    in_channels=exp_size,
                    out_channels=exp_size,
                    kernel_size=kernel_size,
                ),
                nn.BatchNorm2d(exp_size),
                SEBlock(exp_size) if use_se else nn.Identity(),
                activation,
                nn.Conv2d(exp_size, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, exp_size, 1, bias=False),
                nn.BatchNorm2d(exp_size),
                activation,
                nn.Conv2d(exp_size, exp_size, kernel_size, stride, kernel_size // 2, groups=exp_size, bias=False),
                nn.BatchNorm2d(exp_size),
                SEBlock(exp_size) if use_se else nn.Identity(),
                activation,
                nn.Conv2d(exp_size, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            return x + out
        else:
            return out

class MobileNetV3Large(nn.Module):
    def __init__(self, width_mult=1.0, in_channels=1, basemode='Conv'):  # ← 增加 in_channels
        super().__init__()

        cfgs = [
            [3, 16, 16, False, False, 1],
            [3, 64, 24, False, False, 2],
            [3, 72, 24, False, False, 1],
            [5, 72, 40, True, False, 2],
            [5, 120, 40, True, False, 1],
            [5, 120, 40, True, False, 1],
            [3, 240, 80, False, True, 2],
            [3, 200, 80, False, True, 1],
            [3, 184, 80, False, True, 1],
            [3, 184, 80, False, True, 1],
            [3, 480, 112, True, True, 1],
            [3, 672, 112, True, True, 1],
            [5, 672, 160, True, True, 2],
            [5, 960, 160, True, True, 1],
            [5, 960, 160, True, True, 1],
        ]

        input_channel = _make_divisible(16 * width_mult)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, input_channel, 3, stride=2, padding=1, bias=False),  # ← 支持1通道
            nn.BatchNorm2d(input_channel),
            HSwish()
        )

        layers = []
        for k, exp_size, c, use_se, use_hs, s in cfgs:
            output_channel = _make_divisible(c * width_mult)
            exp_size = _make_divisible(exp_size * width_mult)
            layers.append(MobileBottleneck(input_channel, output_channel, k, s, exp_size, use_se, use_hs, basemode))
            input_channel = output_channel
        self.bottlenecks = nn.Sequential(*layers)

        last_exp_size = _make_divisible(960 * width_mult)
        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, last_exp_size, 1, bias=False),
            nn.BatchNorm2d(last_exp_size),
            HSwish()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottlenecks(x)
        x = self.conv_last(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return x
