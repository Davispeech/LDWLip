import torch
import torch.nn as nn
import torch.nn.init as init
import math


__all__ = (
    "LipAuth_Conv3DFront",
    "LipAuth_Conv2DBackbone",
    "LipAuth_SeqEncoder",
)


# ---- 3DConv 前端 ----
class LipAuth_Conv3DFront(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 空间降为 1/4
        )

    def forward(self, x):  # [B, T, C=1, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # -> [B, 1, T, H, W]
        return self.conv3d(x)


# ---- 加深但轻量的 2D CNN ----
class MBConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4):
        super().__init__()
        mid_ch = in_ch * expansion
        self.expand = nn.Conv2d(in_ch, mid_ch, kernel_size=1) if expansion != 1 else nn.Identity()
        self.depthwise = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch)
        self.project = nn.Conv2d(mid_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
        self.shortcut = (in_ch == out_ch)

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.act(x)
        x = self.depthwise(x)
        x = self.act(x)
        x = self.project(x)
        x = self.bn(x)
        if self.shortcut:
            x = x + identity
        return x

class LipAuth_Conv2DBackbone_src(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            MBConvBlock(32, 32),
            nn.MaxPool2d(2),

            MBConvBlock(32, 64),
            MBConvBlock(64, 64),
            nn.MaxPool2d(2),

            MBConvBlock(64, 128),
            MBConvBlock(128, 128),
            nn.AdaptiveAvgPool2d((1, 1))  # 输出 [B*T, 128, 1, 1]
        )

    def forward(self, x):  # [B, C, T, H, W]
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()     # [B, T, C, H, W]
        x = x.view(B * T, C, H, W)                    # [B*T, C, H, W]
        x = self.features(x)                          # [B*T, 128, 1, 1]
        x = x.view(B, T, -1)                          # [B, T, 128]
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class LipAuth_Conv2DBackbone(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(32, 32, num_blocks=1, stride=1)
        self.layer2 = self._make_layer(32, 64, num_blocks=1, stride=2)
        self.layer3 = self._make_layer(64, 128, num_blocks=1, stride=2)
        self.layer4 = self._make_layer(128, 256, num_blocks=1, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 输出 [B*T, 128, 1, 1]

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):  # [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()   # [B, T, C, H, W]
        x = x.view(B * T, C, H, W)                  # [B*T, C, H, W]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)                            # [B*T, 128, 1, 1]
        x = x.view(B, T, -1)                        # [B, T, 128]
        return x


# ---- Transformer 层 ----
class LipAuth_SeqEncoder(nn.Module):
    def __init__(self, d_model=512, max_len=100):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))  # [1, T, d_model]

        self.layer1 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            batch_first=True
        )
        self.layer2 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            batch_first=True
        )

    def forward(self, x):  # [B, T, 128]
        B, T, D = x.shape
        x = x + self.pos_embed[:, :T, :]
        x = self.layer1(x)
        x = self.layer2(x)
        return x

