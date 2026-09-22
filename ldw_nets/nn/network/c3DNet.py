import torch
from torch import nn
from torch.nn import Conv3d, MaxPool3d, Linear, ReLU, Flatten, Dropout

__all__ = (
    "C3DNet",
)


class C3DNet(nn.Module):
    def __init__(self, ):  # num_classes最终分的类别
        super().__init__()
        self.b1 = nn.Sequential(
            Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            ReLU(),
            MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        self.b2 = nn.Sequential(
            Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            ReLU(),
            MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.b3 = nn.Sequential(
            Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            ReLU(),
            Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            ReLU(),
            MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.b4 = nn.Sequential(
            Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            ReLU(),
            Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            ReLU(),
            MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.b5 = nn.Sequential(
            Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            ReLU(),
            Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            ReLU(),
            MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        )

        self.pool = nn.AdaptiveAvgPool3d((512, 1, 1))

        self.__init_weight()

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # [15, 77, 1, 88, 88]
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)  # torch.Size([15, 512, 4, 3, 3])

        x = x.permute(0, 2, 1, 3, 4)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return x  # torch.Size([15, 77, 512])

    # 网络权重参数初始化
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
