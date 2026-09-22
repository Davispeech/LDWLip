import torch
import torch.nn as nn
import torch.nn.init as init
import math
from .conv3D_2DNet import Conv3D_2DNet


__all__ = (
    "Lip3DA2D",
)


class Lip3DA2D(torch.nn.Module):
    def __init__(self, dropout_p=0.5):
        super(Lip3DA2D, self).__init__()
        # 3D卷积层（保持原样）
        self.visualEncoder = Conv3D_2DNet("ResNet18")

        # GRU层（保留到gru2，删除后续的全连接层和输出层）
        self.gru1 = nn.GRU(512, 256, 1, bidirectional=True)  # 96 * 4 * 8
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)  # 这是最后一层

        # 删除以下部分：
        # self.FC = nn.Linear(512, 27 + 1)
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)
        self._init()

    def _init(self):
        # 初始化GRU层（保持原样）
        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(m.weight_ih_l0[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + 256])
                init.constant_(m.bias_ih_l0[i: i + 256], 0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)

    def forward(self, x):
        x = self.visualEncoder(x)  # torch.Size([15, 77, 512])

        # 调整维度（保持原样）
        x = x.permute(1, 0, 2)

        # GRU处理（保留到gru2）
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()

        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)  # 这是最后一层输出

        x = x.permute(1, 0, 2).contiguous()  # torch.Size([15, 77, 512])

        return x
