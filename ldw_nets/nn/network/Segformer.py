import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerConfig


class SegFormerEncoder_src(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # 1. 加载 SegFormer-B0 encoder
        if pretrained:
            self.model = SegformerModel.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
        else:
            config = SegformerConfig.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
            self.model = SegformerModel(config)

        self.encoder = self.model.encoder  # encoder 对象
        self.config = self.model.config

        # 2. 分类头
        # 取最后一层输出通道数作为分类输入
        hidden_size = self.config.hidden_sizes[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.repeat(1, 3, 1, 1)
        encoder_outputs = self.model.encoder(x)  # 得到多层输出
        # latest layer 特征
        last_feat = encoder_outputs[-1]  # (B, C, H, W)
        out = self.pool(last_feat).flatten(1)  # (B, C)

        return out


class SegFormerEncoder(nn.Module):
    def __init__(self, isFreqFusion=False, pretrained=True, c=64):
        """
        Args:
            isFreqFusion: 是否启用频域融合 (FreqFusion)
            pretrained: 是否加载预训练权重
            c: FreqFusion 时每层特征统一到的通道数
        """
        super().__init__()
        self.isFreqFusion = isFreqFusion

        # 1. 加载 SegFormer-B0
        if pretrained:
            self.model = SegformerModel.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
        else:
            config = SegformerConfig.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
            self.model = SegformerModel(config)

        # 2. 如果要用 FreqFusion，则需要 1x1 conv 将通道统一为 c
        if not isFreqFusion:
            self.encoder = self.model.encoder  # encoder 对象
            self.config = self.model.config

            # 2. 分类头
            # 取最后一层输出通道数作为分类输入
            hidden_size = self.config.hidden_sizes[-1]
            self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        else:
            # 确保输出所有 hidden states
            self.model.config.output_hidden_states = True
            self.config = self.model.config
            from ldw_nets.nn.network.FreqFusion import FreqFusion
            self.conv1x1_1 = nn.Conv2d(self.config.hidden_sizes[0], c, kernel_size=1)
            self.conv1x1_2 = nn.Conv2d(self.config.hidden_sizes[1], c, kernel_size=1)
            self.conv1x1_3 = nn.Conv2d(self.config.hidden_sizes[2], c, kernel_size=1)
            self.conv1x1_4 = nn.Conv2d(self.config.hidden_sizes[3], c, kernel_size=1)

            # 定义 FreqFusion 模块
            self.ff1 = FreqFusion(hr_channels=c, lr_channels=c)
            self.ff2 = FreqFusion(hr_channels=c, lr_channels=2 * c)
            self.ff3 = FreqFusion(hr_channels=c, lr_channels=3 * c)

    def forward(self, x):
        if self.isFreqFusion:
            x = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)

            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)

            outputs = self.model(x)
            hidden_states = outputs.hidden_states  # list，包含4层特征 (B, C, H, W)

            x1, x2, x3, x4 = hidden_states  # ✅ 直接解包

            # 通道统一
            x1 = self.conv1x1_1(x1)
            x2 = self.conv1x1_2(x2)
            x3 = self.conv1x1_3(x3)
            x4 = self.conv1x1_4(x4)

            # 融合
            _, x3, x4_up = self.ff1(hr_feat=x3, lr_feat=x4)
            _, x2, x34_up = self.ff2(hr_feat=x2, lr_feat=torch.cat([x3, x4_up], dim=1))
            _, x1, x234_up = self.ff3(hr_feat=x1, lr_feat=torch.cat([x2, x34_up], dim=1))

            x1234 = torch.cat([x1, x234_up], dim=1)  # (B, 4c, H/4, W/4)
            x_pooled = F.adaptive_avg_pool2d(x1234, output_size=1).flatten(1)
            return x_pooled

        else:
            # x: (B, C, H, W)
            x = x.repeat(1, 3, 1, 1)
            encoder_outputs = self.model.encoder(x)  # 得到多层输出
            # latest layer 特征
            last_feat = encoder_outputs[-1]  # (B, C, H, W)
            out = self.pool(last_feat).flatten(1)  # (B, C)

            return out



# 测试
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegFormerEncoder(isFreqFusion=True, pretrained=True).to(device)
    x = torch.randn(1155, 1, 88, 88).to(device)  # 模拟输入
    y = model(x)
    print(y.shape)  # torch.Size([1155, 256])
