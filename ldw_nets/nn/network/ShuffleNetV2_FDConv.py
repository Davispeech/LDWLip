import torch
from torch import nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformerBlock



__all__ = (
    "ShuffleNetV2",
)


# 通道分组 + shuffle
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batchsize, -1, height, width)


class SwinBlockWrapper(nn.Module):
    def __init__(self, input_channels, output_channels, input_resolution=(28,28), num_heads=4, window_size=7):
        super().__init__()
        # Swin Transformer block
        self.swin_block = SwinTransformerBlock(
            dim=input_channels,                # 输入通道
            input_resolution=input_resolution, # 特征图尺寸 HxW
            num_heads=num_heads,               # 注意力头数
            window_size=window_size,           # window size
        )
        # 如果输入输出通道不同，需要1x1 conv调整
        self.channel_adjust = nn.Conv2d(input_channels, output_channels, 1, 1, 0) \
            if input_channels != output_channels else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # -> [B, H, W, C]
        x = self.swin_block(x)
        x = x.permute(0, 3, 1, 2)  # -> [B, C, H, W] 方便后续卷积

        return x


# 基本单元：ShuffleNetV2 block
class ShuffleNetV2Block(nn.Module):
    def __init__(self, inp, outp, stride, basemode="Conv"):
        super(ShuffleNetV2Block, self).__init__()
        self.stride = stride
        branch_features = outp // 2

        if stride == 1:
            assert inp == outp

        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride=stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        # --------- depthwise conv 选择 ---------
        if basemode == "FADC":
            from .FADC import AdaptiveDilatedConv
            depthwise_conv = AdaptiveDilatedConv(in_channels=branch_features,out_channels=branch_features,kernel_size=3)
        elif basemode == "FDConv":
            from .FDConv import FDConv
            depthwise_conv = FDConv(in_channels=branch_features, out_channels=branch_features, kernel_size=3, kernel_num=branch_features)
        else:  # 普通卷积
            depthwise_conv = nn.Conv2d(
                branch_features, branch_features, 3, stride=stride, padding=1,
                groups=branch_features, bias=False
            )
        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if stride > 1 else branch_features, branch_features, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            depthwise_conv,
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            x2 = self.branch2(x2)
            out = torch.cat((x1, x2), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out

# 构建整体网络
class ShuffleNetV2(nn.Module):
    def __init__(self, width_mult=1.0, input_channels=1, basemode='Conv'):
        super(ShuffleNetV2, self).__init__()

        stage_repeats = [4, 8, 4]

        stage_out_channels = {
            0.5: [24, 48, 96, 192, 1024],
            1.0: [24, 116, 232, 464, 1024],
            1.5: [24, 176, 352, 704, 1024],
            2.0: [24, 244, 488, 976, 2048]
        }[width_mult]

        input_channel = stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stages = []
        for idx, numrepeat in enumerate(stage_repeats):
            output_channel = stage_out_channels[idx + 1]
            seq = []
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                if (idx + 2) in (3, 4) and stride==1:
                    if basemode in ["FADC", "FDConv"]:
                        seq.append(ShuffleNetV2Block(input_channel, output_channel, stride, basemode=basemode))
                    elif basemode == "SwinF":
                        seq.append(SwinBlockWrapper(input_channel, output_channel))
                    else:
                        seq.append(ShuffleNetV2Block(input_channel, output_channel, stride))
                else:
                    seq.append(ShuffleNetV2Block(input_channel, output_channel, stride))
                input_channel = output_channel
            stages.append(nn.Sequential(*seq))

        self.stage2, self.stage3, self.stage4 = stages

        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channel, stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(stage_out_channels[-1]),
            nn.ReLU(inplace=True)
        )
        self.globalpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.globalpool(x).view(x.size(0), -1)
        return x
