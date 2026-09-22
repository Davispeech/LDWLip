from torch import nn


__all__ = (
    "TSNetwork",
)


class DepthwiseSeparable3D(nn.Module):
    """3D深度可分离卷积：空间+通道分离"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SpatialTemporalConv(nn.Module):
    """(2+1)D分解卷积：空间2D + 时间1D分离"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # 空间卷积 (H, W)
        self.spatial_conv = nn.Conv3d(in_channels, out_channels,
                                      kernel_size=(1, kernel_size, kernel_size),
                                      stride=(1, stride, stride),
                                      padding=(0, padding, padding))
        # 时间卷积 (T)
        self.temporal_conv = nn.Conv3d(out_channels, out_channels,
                                       kernel_size=(kernel_size, 1, 1),
                                       stride=(stride, 1, 1),
                                       padding=(padding, 0, 0))
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.relu(self.bn(self.temporal_conv(x)))
        return x


class ChannelAttention3D(nn.Module):
    """轻量化3D通道注意力（SE模块变体）"""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class TSNetwork(nn.Module):
    def __init__(self, ):
        super().__init__()
        # 输入尺寸：(T, H, W)，假设为16帧112x112

        # Stage 1: 初始卷积层（高分辨率浅层特征）
        self.conv1 = SpatialTemporalConv(1, 32, kernel_size=3, stride=1, padding=1)
        self.att1 = ChannelAttention3D(32)

        # Stage 2-4: 下采样阶段（逐步减少时空分辨率）
        self.stage2 = self._make_stage(32, 64, stride=2)
        self.stage3 = self._make_stage(64, 128, stride=2)
        self.stage4 = self._make_stage(128, 256, stride=2)

        self.pool = nn.AdaptiveAvgPool3d((512, 1, 1))

        # 初始化权重
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, stride):
        """构建一个下采样阶段（包含残差连接）"""
        return nn.Sequential(
            DepthwiseSeparable3D(in_channels, out_channels, kernel_size=3,
                                 stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            ChannelAttention3D(out_channels),
            DepthwiseSeparable3D(out_channels, out_channels, kernel_size=3,
                                 stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # [15, 1, 77, 88, 88]
        x = self.att1(self.conv1(x))
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)  # torch.Size([15, 256, 10, 11, 11])

        x = x.permute(0, 2, 1, 3, 4)
        x = self.pool(x).squeeze(-1).squeeze(-1)

        return x


