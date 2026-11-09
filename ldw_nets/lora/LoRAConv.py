import torch
import torch.nn as nn
import math

class LoRAConv(nn.Module):
    def __init__(self, base_conv: nn.Module, r=4, alpha=8, dropout=0.0):
        super().__init__()
        self.base_conv = base_conv
        self.r = r
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)

        in_channels = base_conv.in_channels
        out_channels = base_conv.out_channels
        kernel_size = base_conv.kernel_size
        stride = base_conv.stride
        padding = base_conv.padding
        dilation = base_conv.dilation
        groups = base_conv.groups

        ConvClass = type(base_conv)

        # lora_down: in -> r
        self.lora_down = ConvClass(
            in_channels=in_channels,
            out_channels=r * groups,
            kernel_size=kernel_size,
            stride=stride,        # ⚠ 保持和 base_conv 一致
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )

        # lora_up: r -> out, kernel_size=1 保持空间尺寸
        self.lora_up = ConvClass(
            in_channels=r * groups,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=groups,
            bias=False
        )

        # 初始化
        nn.init.zeros_(self.lora_up.weight)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        self.record_corr = False
        self._corr_base = None
        self._corr_out = None

    def forward(self, x):
        base = self.base_conv(x)
        out = base + self.scale * self.lora_up(self.lora_down(self.dropout(x)))
        if self.record_corr:
            self._corr_base = base.detach()
            self._corr_out = out
        return out
