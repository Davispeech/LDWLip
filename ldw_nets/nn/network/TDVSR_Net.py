import torch.nn as nn
import torch.nn.functional as F


__all__ = (
    "SNet",
    "CNet",
    "DNet",
)


class GlobalPool(nn.Module):
    def __init__(self, method='avg', keep_dim=False):
        super().__init__()
        self.method = method
        self.keep_dim = keep_dim

    def forward(self, x):
        if self.method == 'avg':
            x = F.adaptive_avg_pool2d(x, (1, 1))
        elif self.method == 'max':
            x = F.adaptive_max_pool2d(x, (1, 1))
        if not self.keep_dim:
            x = x.view(x.size(0), -1)
        return x


class SNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # [-1, 1, encoder, Conv, [1, 16, 3, 1]]
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # [-1, 1, encoder, Conv, [16, 32, 3, 1]]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # [-1, 1, encoder, MaxPool, [3, 2]]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # [-1, 1, encoder, Conv, [32, 64, 3, 1]]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # [-1, 1, encoder, Conv, [64, 64, 3, 1]]
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # [-1, 1, encoder, MaxPool, [3, 2]]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # [-1, 1, encoder, Conv, [64, 128, 3, 1]]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # [-1, 1, encoder, Conv, [128, 128, 3, 1]]
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # [-1, 1, encoder, MaxPool, [3, 2]]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # [-1, 1, encoder, Conv, [128, 256, 3, 1]]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # [-1, 1, encoder, Conv, [256, 256, 3, 1]]
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # [-1, 1, encoder, MaxPool, [3, 2]]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # [-1, 1, encoder, Conv, [256, 512, 3, 1]]
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # [-1, 1, encoder, Conv, [512, 256, 3, 1]]
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # [-1, 1, encoder, GlobalPool, ['avg', False]]
            GlobalPool(method='avg', keep_dim=False)
        )

    def forward(self, x):  # torch.Size([16, 75, 1, 88, 88])
        return self.layers(x)


class CNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Spatiotemporal Conv Blocks
        self.b1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        self.b2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        self.b3 = nn.Sequential(
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        self.linear = nn.Linear(96, 256)
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # 保持时间维，压缩空间维

    def forward(self, x):  # x: [B, T, C, H, W] = [16, 75, 1, 88, 88]
        x = x.permute(0, 2, 1, 3, 4)  # [16, 1, 75, 88, 88]
        x = self.b1(x)               # -> [16, 64, 75, 22, 22]
        x = self.b2(x)               # -> [16, 128, 75, 11, 11]
        x = self.b3(x)               # -> [16, 96, 75, 5, 5]
        x = self.pool(x)             # -> [16, 256, 75, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # -> [16, 96, 75]
        x = x.permute(0, 2, 1)         # -> [16, 75, 96]
        x = self.linear(x)
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        # Residual connection
        out = out + identity
        out = F.relu(self.conv3(out))
        out = self.conv4(out)
        return out

class DNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        in_channels = 1
        hidden_channels = 32
        gru_hidden_size = 128
        # 3D convolution
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )

        # Transition from 3D to 2D
        # After pooling, squeeze the temporal dimension (assuming input shape is [B, C, D, H, W])
        self.reduce_dim = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)

        # Two 2-layer Conv2D blocks
        self.conv2d_stage1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv2d_stage2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Three 4-layer Conv2D residual blocks
        self.block1 = Conv2DBlock(hidden_channels, hidden_channels, hidden_channels)
        self.block2 = Conv2DBlock(hidden_channels, hidden_channels, hidden_channels)
        self.block3 = Conv2DBlock(hidden_channels, hidden_channels, hidden_channels)

        # Pooling before BiGRU
        self.pool = nn.AdaptiveAvgPool2d((None, 1))  # output shape: [B, C, H, 1]

        # BiGRU
        self.bigru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=gru_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        # torch.Size([16, 75, 1, 88, 88])
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3d(x)  # -> [B, C, D', H', W']
        x = x.mean(dim=2)  # temporal average/squeeze: [B, C, H', W']
        x = self.reduce_dim(x)

        x = self.conv2d_stage1(x)
        x = self.conv2d_stage2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.pool(x)  # -> [B, C, H, 1]
        x = x.squeeze(-1)  # -> [B, C, H]
        x = x.permute(0, 2, 1)  # -> [B, H, C]

        output, _ = self.bigru(x)  # -> [B, H, 2*hidden]
        output = output[:, -1, :]  # Take last time step

        return output  # torch.Size([16, 256])
