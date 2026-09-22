import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenet import mobilenet_v2



__all__ = (
    "MobileNetV2_035",

)


class MobileNetV2_035(nn.Module):
    def __init__(self):
        super(MobileNetV2_035, self).__init__()
        base_model = mobilenet_v2(width_mult=0.35, pretrained=False)

        # 改第一层卷积输入通道为1，保持其他参数不变
        first_conv = base_model.features[0][0]  # Conv2d(3, 16, kernel=3, stride=2, padding=1)
        new_first_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )

        # 初始化新卷积层权重：把原3通道权重按均值合成1通道权重
        with torch.no_grad():
            new_first_conv.weight[:] = first_conv.weight.mean(dim=1, keepdim=True)
            if first_conv.bias is not None:
                new_first_conv.bias[:] = first_conv.bias

        # 替换base_model.features第一层卷积
        base_model.features[0][0] = new_first_conv

        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.last_channel = base_model.last_channel

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


