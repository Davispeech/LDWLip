import torch
from torch import nn
import torch.nn.functional as F


__all__ = (
    "MobileNetV3Large",
    "MobileNetV2",
    "ShuffleNetV2lrw",
    "ShuffleNetV2",
    "EfficientNet",
    "EfficientNetLite0",
    "ResNet18",
    "SqueezeNet",
)


# ------------------------
# Inverted Residual Block
# ------------------------
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            # Pointwise
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        # Depthwise
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        # Pointwise Linear
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# ------------------------
# MobileNetV2
# ------------------------
class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0, in_channels=1):
        super(MobileNetV2, self).__init__()
        # Configuration: t, c, n, s
        # t: expand ratio, c: output channels, n: number of blocks, s: stride
        cfgs = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        self.features = [nn.Conv2d(in_channels, input_channel, 3, stride=2, padding=1, bias=False),
                         nn.BatchNorm2d(input_channel),
                         nn.ReLU6(inplace=True)]

        # Building inverted residual blocks
        for t, c, n, s in cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

        # Final layers
        self.features.append(nn.Conv2d(input_channel, last_channel, 1, bias=False))
        self.features.append(nn.BatchNorm2d(last_channel))
        self.features.append(nn.ReLU6(inplace=True))

        self.features = nn.Sequential(*self.features)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def _make_divisible(v, divisor=8, min_value=None):
    # 确保通道数为8的倍数
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6

class SEBlock(nn.Module):
    def __init__(self, in_ch, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch, bias=False),
            HSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MobileBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, exp_size, use_se, use_hs):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

        # 修改这里：HSwish不带inplace参数，ReLU带
        if use_hs:
            activation = HSwish()
        else:
            activation = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, exp_size, 1, bias=False),
            nn.BatchNorm2d(exp_size),
            activation,
            nn.Conv2d(exp_size, exp_size, kernel_size, stride, kernel_size//2, groups=exp_size, bias=False),
            nn.BatchNorm2d(exp_size),
            SEBlock(exp_size) if use_se else nn.Identity(),
            activation,
            nn.Conv2d(exp_size, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            return x + out
        else:
            return out

class MobileNetV3Large(nn.Module):
    def __init__(self, width_mult=1.0, in_channels=1):  # ← 增加 in_channels
        super().__init__()

        cfgs = [
            [3, 16, 16, False, False, 1],
            [3, 64, 24, False, False, 2],
            [3, 72, 24, False, False, 1],
            [5, 72, 40, True, False, 2],
            [5, 120, 40, True, False, 1],
            [5, 120, 40, True, False, 1],
            [3, 240, 80, False, True, 2],
            [3, 200, 80, False, True, 1],
            [3, 184, 80, False, True, 1],
            [3, 184, 80, False, True, 1],
            [3, 480, 112, True, True, 1],
            [3, 672, 112, True, True, 1],
            [5, 672, 160, True, True, 2],
            [5, 960, 160, True, True, 1],
            [5, 960, 160, True, True, 1],
        ]

        input_channel = _make_divisible(16 * width_mult)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, input_channel, 3, stride=2, padding=1, bias=False),  # ← 支持1通道
            nn.BatchNorm2d(input_channel),
            HSwish()
        )

        layers = []
        for k, exp_size, c, use_se, use_hs, s in cfgs:
            output_channel = _make_divisible(c * width_mult)
            exp_size = _make_divisible(exp_size * width_mult)
            layers.append(MobileBottleneck(input_channel, output_channel, k, s, exp_size, use_se, use_hs))
            input_channel = output_channel
        self.bottlenecks = nn.Sequential(*layers)

        last_exp_size = _make_divisible(960 * width_mult)
        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, last_exp_size, 1, bias=False),
            nn.BatchNorm2d(last_exp_size),
            HSwish()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottlenecks(x)
        x = self.conv_last(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return x


# 通道分组 + shuffle
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batchsize, -1, height, width)

# 基本单元：ShuffleNetV2 block
class ShuffleNetV2Block(nn.Module):
    def __init__(self, inp, outp, stride):
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

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if stride > 1 else branch_features, branch_features, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, 3, stride=stride, padding=1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out

# 构建整体网络
class ShuffleNetV2(nn.Module):
    def __init__(self, width_mult=1.0, input_channels=1):
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


# Swish 激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# SE模块
class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        reduced_channels = in_channels // reduction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

# MBConv block
class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio, stride, kernel_size, se_ratio=0.25):
        super().__init__()
        hidden_dim = in_ch * expand_ratio
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU()
            ]
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ]
        if se_ratio:
            layers.append(SEModule(hidden_dim, reduction=int(1 / se_ratio)))
        layers += [
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# 主体网络
class EfficientNet(nn.Module):
    def __init__(self, width_mult=1.0, input_channels=1, depth_mult=1.0):
        super().__init__()
        base_channels = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU()
        )

        # 构建MBConv Blocks
        cfg = [
                # expand_ratio, channels, repeats, stride, kernel_size
                [1, 16, 1, 1, 3],
                [6, 24, 2, 2, 3],
                [6, 40, 2, 2, 5],
                [6, 80, 3, 2, 3],
                [6, 112, 3, 1, 5],
                [6, 192, 4, 2, 5],
                [6, 320, 1, 1, 3],
            ]
        layers = []
        in_ch = base_channels
        for expand_ratio, out_ch, repeats, stride, k in cfg:
            out_ch = int(out_ch * width_mult)
            repeats = int(repeats * depth_mult)
            for i in range(repeats):
                s = stride if i == 0 else 1
                layers.append(MBConv(in_ch, out_ch, expand_ratio, s, k))
                in_ch = out_ch
        self.blocks = nn.Sequential(*layers)

        head_channels = int(1280 * width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        return x


# -----------------------
# ReLU6 激活函数代替 Swish（更适合移动设备）
# -----------------------
class ReLU6(nn.Module):
    def forward(self, x):
        return F.relu6(x, inplace=True)

# -----------------------
# Squeeze-and-Excitation（可选，lite0 中默认仍保留）
# -----------------------
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SqueezeExcitation, self).__init__()
        reduced = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

# -----------------------
# MBConv Block
# -----------------------
class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, expand_ratio, use_se=True, se_ratio=0.25):
        super(MBConv, self).__init__()
        mid_ch = in_ch * expand_ratio
        self.use_residual = (in_ch == out_ch and stride == 1)
        self.expand = expand_ratio != 1

        layers = []
        if self.expand:
            layers.extend([
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                ReLU6()
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(mid_ch, mid_ch, kernel_size, stride, kernel_size // 2, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            ReLU6()
        ])

        # SE
        if use_se:
            layers.append(SqueezeExcitation(mid_ch, se_ratio))

        # Project
        layers.extend([
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):  # torch.Size([1196, 32, 11, 11])
        if self.use_residual:  # torch.Size([1196, 24, 6, 6])
            return x + self.block(x)
        else:
            return self.block(x)

# -----------------------
# EfficientNet-Lite0
# -----------------------
class EfficientNetLite0(nn.Module):
    def __init__(self, ):
        super(EfficientNetLite0, self).__init__()
        self.cfgs = [
            # t, c, n, s, k, use_se
            [1, 16, 1, 1, 3, False],
            [6, 24, 2, 2, 3, False],
            [6, 40, 2, 2, 5, True],
            [6, 80, 3, 2, 3, False],
            [6, 112, 3, 1, 5, True],
            [6, 192, 4, 2, 5, True],
            [6, 320, 1, 1, 3, True]
        ]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            ReLU6()
        )

        # MBConv blocks
        in_channels = 32
        blocks = []
        for t, c, n, s, k, use_se in self.cfgs:
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(MBConv(in_channels, c, k, stride, t, use_se))
                in_channels = c
        self.blocks = nn.Sequential(*blocks)

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            ReLU6()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)



# -------------------------------------
# BasicBlock: 用于 ResNet-18 / ResNet-34
# -------------------------------------
class BasicBlock(nn.Module):
    expansion = 1  # 输出通道不变

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample  # 对残差连接进行调整
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

# -------------------------------------
# ResNet18 构建函数
# -------------------------------------
class ResNet18(nn.Module):
    def __init__(self, ):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # Stem
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64,  2, stride=1)  # conv2_x
        self.layer2 = self._make_layer(128, 2, stride=2)  # conv3_x
        self.layer3 = self._make_layer(256, 2, stride=2)  # conv4_x
        self.layer4 = self._make_layer(512, 2, stride=2)  # conv5_x

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 初始化参数
        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)


# -----------------------------
# Fire 模块
# -----------------------------
class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels,
                                   kernel_size=3, padding=1)

        self.expand_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return self.expand_activation(torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1))

# -----------------------------
# SqueezeNet v1.1
# -----------------------------
class SqueezeNet(nn.Module):
    def __init__(self, in_channels=1):
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),  # 输出: 64x112x112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 64x56x56

            Fire(64, 16, 64, 64),     # 128x56x56
            Fire(128, 16, 64, 64),    # 128x56x56
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 128x28x28

            Fire(128, 32, 128, 128),  # 256x28x28
            Fire(256, 32, 128, 128),  # 256x28x28
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 256x14x14

            Fire(256, 48, 192, 192),  # 384x14x14
            Fire(384, 48, 192, 192),  # 384x14x14
            Fire(384, 64, 256, 256),  # 512x14x14
            Fire(512, 64, 256, 256),  # 512x14x14
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shufflelrw(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResiduallrw_src(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResiduallrw_src, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shufflelrw(out, 2)


class InvertedResiduallrw(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel, basemode='Conv'):
        super(InvertedResiduallrw, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        self.basemode = basemode
        assert stride in [1, 2]

        oup_inc = oup // 2

        # -------- 封装一个 DWConv 选择器 -------- #
        def DWConv(in_ch, stride):
            if self.basemode=='FDConv':
                from .FDConv import FDConv
                return FDConv(in_channels=in_ch, out_channels=in_ch, kernel_num=8, kernel_size=3, padding=1, bias=True)
            else:
                # 默认 DWConv
                return nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)

        # -------- benchmodel == 1 -------- #
        if self.benchmodel == 1:
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                DWConv(oup_inc, stride),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

        # -------- benchmodel == 2 -------- #
        else:
            self.banch1 = nn.Sequential(
                # dw
                DWConv(inp, stride),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                DWConv(oup_inc, stride),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # 沿通道拼接
        return torch.cat((x, out), 1)

    def forward(self, x):
        if self.benchmodel == 1:
            # 前半部分直连，后半部分经过 banch2
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        else:  # benchmodel == 2
            out = self._concat(self.banch1(x), self.banch2(x))

        # 通道打乱
        return channel_shufflelrw(out, 2)


class ShuffleNetV2lrw(nn.Module):
    def __init__(self, input_size=224, width_mult=2., basemode_in='Conv'):
        super(ShuffleNetV2lrw, self).__init__()

        assert input_size % 32 == 0, "Input size needs to be divisible by 32"

        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise ValueError(
                """Width multiplier should be in [0.5, 1.0, 1.5, 2.0]. Current value: {}""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]

        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                basemode = 'Conv'
                if i == 0:
                    # inp, oup, stride, benchmodel):
                    self.features.append(InvertedResiduallrw(input_channel, output_channel, 2, 2, basemode))
                else:
                    if idxstage == 2:
                        basemode = basemode_in
                    self.features.append(InvertedResiduallrw(input_channel, output_channel, 1, 1, basemode))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size / 32)))

    def forward(self, x):
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, self.stage_out_channels[-1])
        return x
