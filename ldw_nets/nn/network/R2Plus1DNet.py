import torch
import torch.nn as nn
import torch.nn.functional as F




class LiteR2Plus1DBlock(nn.Module):
    """轻量化 R(2+1)D Block: Depthwise 空间卷积 + 1D 时间卷积"""
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        # Depthwise 空间卷积
        self.spatial_dw = nn.Conv3d(in_ch, in_ch, kernel_size=(1,3,3),
                                    stride=(1,stride,stride), padding=(0,1,1),
                                    groups=in_ch, bias=False)
        self.spatial_pw = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        # 时间卷积
        self.temporal_conv = nn.Conv3d(out_ch, out_ch, kernel_size=(3,1,1),
                                       stride=(stride,1,1), padding=(1,0,0), bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.spatial_dw(x)
        out = self.spatial_pw(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.temporal_conv(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class LiteR2Plus1D18(nn.Module):
    """轻量化 R(2+1)D-18 视频分类"""
    def __init__(self, num_classes=101, input_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,3,3), stride=(1,2,2), padding=(0,1,1))
        )

        # stage 通道缩小
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 256, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=(stride,stride,stride), bias=False),
                nn.BatchNorm3d(out_ch)
            )
        layers = [LiteR2Plus1DBlock(in_ch, out_ch, stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(LiteR2Plus1DBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

# 测试
if __name__ == "__main__":
    model = LiteR2Plus1D18(num_classes=101)
    x = torch.randn(2, 3, 8, 112, 112)  # batch=2, 8帧, 112x112
    y = model(x)
    print(y.shape)  # torch.Size([2, 101])
