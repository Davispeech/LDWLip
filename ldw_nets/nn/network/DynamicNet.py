import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F



__all__ = (
    "R2Plus1D10",
    "SqueezeTime",
    "MobileViCLIP",
    "Mobile3DNet",
    "SlowFast",
    "SlowFastLite",
    "X3DNet_M",
    "TimeSformerLite",
    "TimeSformerTiny",
    "TimeSformerTiny",
)


class R2Plus1DBlock(nn.Module):
    """R(2+1)D 基础块：2D 空间卷积 + 1D 时间卷积"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 空间卷积 HxW
        self.spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3),
                                      stride=(1,stride,stride), padding=(0,1,1), bias=False)
        self.spatial_bn = nn.BatchNorm3d(out_channels)
        # 时间卷积 T
        self.temporal_conv = nn.Conv3d(out_channels, out_channels, kernel_size=(3,1,1),
                                       stride=(stride,1,1), padding=(1,0,0), bias=False)
        self.temporal_bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.spatial_conv(x)
        out = self.spatial_bn(out)
        out = self.relu(out)
        out = self.temporal_conv(out)
        out = self.temporal_bn(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class R2Plus1D10(nn.Module):
    """R(2+1)D-10 视频分类网络"""
    def __init__(self, input_channels=1):
        super().__init__()
        # stem
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        )

        # stage 配置，总 block 数 = 10
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=3, stride=2)
        self.layer4 = self._make_layer(128, 256, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=(stride,stride,stride), bias=False),
                nn.BatchNorm3d(out_ch)
            )
        layers = [R2Plus1DBlock(in_ch, out_ch, stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(R2Plus1DBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):  # 输入 [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, C=1, T, H, W]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x



from torchvision.models.resnet import BasicBlock
class SmallResNet10(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(32, 1)
        self.layer2 = self._make_layer(64, 1, stride=2)
        self.layer3 = self._make_layer(128, 1, stride=2)
        self.layer4 = self._make_layer(256, 1, stride=2)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # [B, 256, H', W']

class ChannelTimeLearningBlock(nn.Module):
    def __init__(self, in_channels, fold_div=8):
        super().__init__()
        self.fold = in_channels // fold_div
        self.conv_tfc = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv_ioi = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        x_tfc = F.relu(self.conv_tfc(x))
        x_ioi = F.relu(self.conv_ioi(x))
        return x_tfc + x_ioi

class SqueezeTime(nn.Module):
    def __init__(self, T=75, inputdata_channel=1, fold_div=8):
        super().__init__()
        self.T_target = T
        self.inputdata_channel = inputdata_channel
        self.backbone = SmallResNet10(in_channels=self.T_target * self.inputdata_channel)
        self.ctl_block = ChannelTimeLearningBlock(in_channels=256, fold_div=fold_div)

    def forward(self, x):  # 输入 [B, T, C, H, W]
        B, T, C, H, W = x.shape
        # assert C == self.inputdata_channel, f"Input channel {C} does not match model setting {self.inputdata_channel}"

        # 时间插值到固定长度 T_target
        if T != self.T_target:
            x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            x = F.interpolate(x, size=(self.T_target, H, W), mode='trilinear', align_corners=False)
            x = x.permute(0, 2, 1, 3, 4)  # [B, T_target, C, H, W]

        # reshape 到 [B, C*T_target, H, W] 输入 backbone
        x = x.permute(0, 2, 1, 3, 4).reshape(B, C*self.T_target, H, W)

        x = self.backbone(x)  # [B, 256, H', W']

        # 扩展时间维度为 1
        H_new, W_new = x.shape[2], x.shape[3]
        x = x.view(B, 256, 1, H_new, W_new)

        x = self.ctl_block(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = torch.flatten(x, 1)
        return x  # [B, 256]


from torchvision.models import mobilenet_v3_large
class TemporalRepMixer(nn.Module):
    """RepMixer 风格的时间卷积模块"""
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, D, T]
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class MobileViCLIP(nn.Module):
    def __init__(self, width_mult=0.35, encoder_dim=336, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Backbone: MobileNetV3-Large, width_mult=0.35
        base_model = mobilenet_v3_large(weights=None, width_mult=width_mult)
        base_model.classifier = nn.Identity()  # 去掉最后分类层
        self.backbone = base_model

        # 将 backbone 输出映射到 hidden_dim
        self.proj = nn.Linear(encoder_dim, hidden_dim)  # MobileNetV3-Large 最后一层输出是 576

        # 时间维度建模
        self.temporal_mixer = TemporalRepMixer(hidden_dim)

        # 输出全局池化
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        返回: [B, hidden_dim]
        """
        if x.size(2) == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        B, T, C, H, W = x.shape

        # 每帧送入 backbone
        x = x.view(B*T, C, H, W)             # [B*T, C, H, W]
        features = self.backbone(x)          # [B*T, 576]
        features = self.proj(features)       # [B*T, hidden_dim]
        features = features.view(B, T, self.hidden_dim)  # [B, T, hidden_dim]

        # 时间维度建模
        features = features.permute(0, 2, 1)   # [B, hidden_dim, T]
        features = self.temporal_mixer(features)

        # 全局池化时间维度
        features = self.pool(features).squeeze(-1)  # [B, hidden_dim]
        return features


class Mobile3DBlock(nn.Module):
    def __init__(self, in_c, out_c, expansion_ratio=6, stride=1):
        super().__init__()
        hidden_dim = in_c * expansion_ratio
        self.block = nn.Sequential(
            # 扩展维度
            nn.Conv3d(in_c, hidden_dim, 1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(inplace=True),
            # 深度可分离卷积
            nn.Conv3d(hidden_dim, hidden_dim, (3, 3, 3),
                      stride=(1, stride, stride),  # 时间维度不下采样
                      padding=(1, 1, 1), groups=hidden_dim),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(inplace=True),
            # 投影层
            nn.Conv3d(hidden_dim, out_c, 1, bias=False),
            nn.BatchNorm3d(out_c)
        )
        self.skip = nn.Identity() if (in_c == out_c and stride == 1) else None

    def forward(self, x):
        if self.skip is not None:
            return self.block(x) + self.skip(x)
        return self.block(x)


class Mobile3DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 初始卷积
            nn.Conv3d(1, 16, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU6(inplace=True),

            # 堆叠Mobile3D块
            Mobile3DBlock(16, 64, stride=2),
            Mobile3DBlock(64, 128, stride=2),
            Mobile3DBlock(128, 256, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool3d((512, 1, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # [15, 77, 1, 88, 88]
        x = self.features(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return x  # torch.Size([15, 77, 512])


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, k, s, p):
        super().__init__(
            nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )

class X3DMBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            # Expand
            nn.Conv3d(in_c, in_c * 6, kernel_size=1, bias=False),
            nn.BatchNorm3d(in_c * 6),
            nn.ReLU(inplace=True),

            # Depthwise Conv
            nn.Conv3d(in_c * 6, in_c * 6, kernel_size=3, stride=stride, padding=1,
                      groups=in_c * 6, bias=False),
            nn.BatchNorm3d(in_c * 6),
            nn.ReLU(inplace=True),

            # Projection
            nn.Conv3d(in_c * 6, out_c, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_c),
        )

        self.skip = (in_c == out_c and stride == 1)

    def forward(self, x):
        if self.skip:
            return self.block(x) + x
        else:
            return self.block(x)

class X3DNet_M(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBNReLU(1, 24, k=3, s=(1, 2, 2), p=1)  # [B, 1, T, H, W] → [B, 24, T, H/2, W/2]

        self.stage1 = nn.Sequential(
            X3DMBlock(24, 48, stride=2),  # 时空下采样
            X3DMBlock(48, 48),
        )
        self.stage2 = nn.Sequential(
            X3DMBlock(48, 96, stride=2),
            X3DMBlock(96, 96),
        )
        self.stage3 = nn.Sequential(
            X3DMBlock(96, 192, stride=2),
            X3DMBlock(192, 192),
        )

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 输出 [B, C, 1, 1, 1]
        self.fc = nn.Linear(192, 256)

    def forward(self, x):  # 输入 [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, C=1, T, H, W]
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)  # [B, 192]
        x = self.fc(x)  # [B, 256]
        return x


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

def match_size(x, ref):
    # 三线性插值，使 x 的 (T,H,W) 维度与 ref 对齐
    if x.shape[-3:] != ref.shape[-3:]:
        x = F.interpolate(x, size=ref.shape[-3:], mode='trilinear', align_corners=False)
    return x


# SlowFastR18
class SlowFast(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.alpha = 8     # 时间采样倍数
        self.beta = 0.125  # 通道缩放比例

        # Slow Pathway Stem
        self.slow_conv1 = nn.Conv3d(1, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        # Fast Pathway Stem
        self.fast_conv1 = nn.Conv3d(1, int(64*self.beta), kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False)
        self.fast_bn1 = nn.BatchNorm3d(int(64*self.beta))
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        # ResNet Layers for Slow Pathway (R18: 2 blocks per layer)
        self.slow_res2 = self._make_layer(64, 64, blocks=2, stride=1)
        self.slow_res3 = self._make_layer(64, 128, blocks=2, stride=2)
        self.slow_res4 = self._make_layer(128, 256, blocks=2, stride=2)
        self.slow_res5 = self._make_layer(256, 512, blocks=2, stride=2)

        # ResNet Layers for Fast Pathway
        self.fast_res2 = self._make_layer(int(64*self.beta), int(64*self.beta), blocks=2, stride=1)
        self.fast_res3 = self._make_layer(int(64*self.beta), int(128*self.beta), blocks=2, stride=2)
        self.fast_res4 = self._make_layer(int(128*self.beta), int(256*self.beta), blocks=2, stride=2)
        self.fast_res5 = self._make_layer(int(256*self.beta), int(512*self.beta), blocks=2, stride=2)

        # Lateral connections: fast -> slow (channel mapping by 1x1x1 conv)
        self.lateral_p2 = nn.Conv3d(int(64*self.beta), 64, kernel_size=1, bias=False)
        self.lateral_p3 = nn.Conv3d(int(128*self.beta), 128, kernel_size=1, bias=False)
        self.lateral_p4 = nn.Conv3d(int(256*self.beta), 256, kernel_size=1, bias=False)
        self.lateral_p5 = nn.Conv3d(int(512*self.beta), 512, kernel_size=1, bias=False)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))

    def _make_layer(self, inplanes, planes, blocks, stride):
        layers = []
        layers.append(BasicBlock3D(inplanes, planes, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入形状: (B, T, 1, H, W)
        B, T, C, H, W = x.shape

        # Fast Pathway 输入：保持原始时间分辨率
        fast_input = x.permute(0,2,1,3,4).contiguous()  # (B, 1, T, H, W)

        # Slow Pathway 输入：时间采样间隔alpha倍采样
        slow_input = fast_input[:,:,::self.alpha,:,:]  # (B, 1, T//alpha, H, W)

        # Stem
        s = self.slow_relu(self.slow_bn1(self.slow_conv1(slow_input)))  # (B, 64, T//alpha, H/2, W/2)
        s = self.slow_maxpool(s)                                        # (B, 64, T//alpha, H/4, W/4)

        f = self.fast_relu(self.fast_bn1(self.fast_conv1(fast_input))) # (B, 8, T, H/2, W/2)
        f = self.fast_maxpool(f)                                        # (B, 8, T, H/4, W/4)

        # Layer 2
        s2 = self.slow_res2(s)
        f2 = self.fast_res2(f)
        lateral2 = self.lateral_p2(f2)
        lateral2 = match_size(lateral2, s2)
        s2 = s2 + lateral2

        # Layer 3
        s3 = self.slow_res3(s2)
        f3 = self.fast_res3(f2)
        lateral3 = self.lateral_p3(f3)
        lateral3 = match_size(lateral3, s3)
        s3 = s3 + lateral3

        # Layer 4
        s4 = self.slow_res4(s3)
        f4 = self.fast_res4(f3)
        lateral4 = self.lateral_p4(f4)
        lateral4 = match_size(lateral4, s4)
        s4 = s4 + lateral4

        # Layer 5
        s5 = self.slow_res5(s4)
        f5 = self.fast_res5(f4)
        lateral5 = self.lateral_p5(f5)
        lateral5 = match_size(lateral5, s5)
        s5 = s5 + lateral5

        # 池化 + 分类
        s_pool = self.avgpool(s5).flatten(1)
        f_pool = self.avgpool(f5).flatten(1)
        out = torch.cat([s_pool, f_pool], dim=1)

        return out


class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, activation_layer=nn.ReLU):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm3d(out_planes),
            activation_layer(inplace=True)
        )

class SqueezeExcitation3D(nn.Module):
    def __init__(self, input_c, squeeze_factor=4):
        super().__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv3d(input_c, squeeze_c, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(squeeze_c, input_c, kernel_size=1)
        self.hsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        scale = x.mean((2,3,4), keepdim=True)  # Global avg pool (T,H,W)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hsigmoid(scale)
        return x * scale

class MobileNetV3Block3D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, expand_ratio=4, use_se=True, activation=nn.Hardswish):
        super().__init__()
        hidden_dim = in_planes * expand_ratio
        self.use_res_connect = (stride == 1 and in_planes == out_planes)
        self.expand = nn.Sequential(
            nn.Conv3d(in_planes, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            activation(inplace=True)
        )
        self.depthwise = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            activation(inplace=True)
        )
        self.se = SqueezeExcitation3D(hidden_dim) if use_se else nn.Identity()
        self.project = nn.Sequential(
            nn.Conv3d(hidden_dim, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_planes)
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_res_connect:
            return x + out
        else:
            return out

# SlowFastMobileNetV3
class SlowFastLite(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.alpha = 8     # 时间采样倍数
        self.beta = 0.125  # 通道缩放比例

        # Slow Pathway Stem
        self.slow_conv1 = ConvBNActivation(1, 16, kernel_size=3, stride=2, activation_layer=nn.Hardswish)
        self.slow_pool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        # Fast Pathway Stem
        self.fast_conv1 = ConvBNActivation(1, int(16*self.beta), kernel_size=3, stride=2, activation_layer=nn.Hardswish)
        self.fast_pool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        # MobileNetV3 blocks配置(简化版本)
        self.slow_res2 = self._make_layer(16, 24, 2, stride=2, expand=4, se=True)
        self.slow_res3 = self._make_layer(24, 40, 2, stride=2, expand=4, se=True)
        self.slow_res4 = self._make_layer(40, 80, 2, stride=2, expand=6, se=False)
        self.slow_res5 = self._make_layer(80, 112, 2, stride=1, expand=6, se=True)

        self.fast_res2 = self._make_layer(int(16*self.beta), int(24*self.beta), 2, stride=2, expand=4, se=True)
        self.fast_res3 = self._make_layer(int(24*self.beta), int(40*self.beta), 2, stride=2, expand=4, se=True)
        self.fast_res4 = self._make_layer(int(40*self.beta), int(80*self.beta), 2, stride=2, expand=6, se=False)
        self.fast_res5 = self._make_layer(int(80*self.beta), int(112*self.beta), 2, stride=1, expand=6, se=True)

        # lateral connections 1x1 conv
        self.lateral_p2 = nn.Conv3d(int(24*self.beta), 24, kernel_size=1, bias=False)
        self.lateral_p3 = nn.Conv3d(int(40*self.beta), 40, kernel_size=1, bias=False)
        self.lateral_p4 = nn.Conv3d(int(80*self.beta), 80, kernel_size=1, bias=False)
        self.lateral_p5 = nn.Conv3d(int(112*self.beta), 112, kernel_size=1, bias=False)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))

    def _make_layer(self, in_planes, out_planes, blocks, stride, expand, se):
        layers = []
        layers.append(MobileNetV3Block3D(in_planes, out_planes, stride=stride, expand_ratio=expand, use_se=se))
        for _ in range(1, blocks):
            layers.append(MobileNetV3Block3D(out_planes, out_planes, stride=1, expand_ratio=expand, use_se=se))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入: (B, T, 1, H, W)
        B, T, C, H, W = x.shape

        fast_input = x.permute(0,2,1,3,4).contiguous()         # (B, 1, T, H, W)
        slow_input = fast_input[:,:,::self.alpha,:,:]           # (B, 1, T//alpha, H, W)

        # Stem
        s = self.slow_conv1(slow_input)
        s = self.slow_pool(s)
        f = self.fast_conv1(fast_input)
        f = self.fast_pool(f)

        # Layer 2
        s2 = self.slow_res2(s)
        f2 = self.fast_res2(f)
        lateral2 = self.lateral_p2(f2)
        lateral2 = match_size(lateral2, s2)
        s2 = s2 + lateral2

        # Layer 3
        s3 = self.slow_res3(s2)
        f3 = self.fast_res3(f2)
        lateral3 = self.lateral_p3(f3)
        lateral3 = match_size(lateral3, s3)
        s3 = s3 + lateral3

        # Layer 4
        s4 = self.slow_res4(s3)
        f4 = self.fast_res4(f3)
        lateral4 = self.lateral_p4(f4)
        lateral4 = match_size(lateral4, s4)
        s4 = s4 + lateral4

        # Layer 5
        s5 = self.slow_res5(s4)
        f5 = self.fast_res5(f4)
        lateral5 = self.lateral_p5(f5)
        lateral5 = match_size(lateral5, s5)
        s5 = s5 + lateral5

        s_pool = self.avgpool(s5).flatten(1)
        f_pool = self.avgpool(f5).flatten(1)

        out = torch.cat([s_pool, f_pool], dim=1)

        return out


# Patch + Positional Embedding
class PatchEmbed(nn.Module):
    def __init__(self, img_size=88, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def forward(self, x):  # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, C]
        return x + self.pos_embed


# Lite Temporal Attention Block
class TemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):  # x: [B*T, N_patch, C]
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


# Overall TimeSformer Lite
class TimeSformerLite(nn.Module):
    def __init__(self, img_size=88, patch_size=16, frames=8, embed_dim=192, depth=4):
        super().__init__()
        self.frames = frames
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans=1, embed_dim=embed_dim)
        self.temporal_blocks = nn.Sequential(
            *[TemporalAttention(embed_dim) for _ in range(depth)]
        )

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.patch_embed(x)  # [B*T, N, C]
        x = self.temporal_blocks(x)  # [B*T, N, C]
        x = x.mean(dim=1)  # Global average over patches: [B*T, C]
        x = x.view(B, T, -1)  # [B, T, C]
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_dim=256, image_size=88):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (image_size // patch_size) ** 2

    def forward(self, x):
        # x: (B*T, C, H, W)
        x = self.proj(x)  # (B*T, emb_dim, H', W')
        x = rearrange(x, 'b c h w -> b (h w) c')  # flatten patches
        return x  # (B*T, N_patches, emb_dim)


class TimeAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=6):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, B, T):
        # x: (B*T, N_patches, emb_dim)
        N = x.size(1)
        # reshape for time attention: group by patches across time
        x = rearrange(x, '(b t) n c -> n b t c', b=B, t=T)
        x = rearrange(x, 'n b t c -> (n b) t c')  # (N*B, T, C)
        x_norm = self.norm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + out
        x = rearrange(x, '(n b) t c -> n (b t) c', b=B)
        return x  # (N_patches, B*T, emb_dim)


class SpaceAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=6):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # x: (B*T, N_patches, emb_dim)
        x_norm = self.norm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + out
        return x  # (B*T, N_patches, emb_dim)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim=384, num_heads=6):
        super().__init__()
        self.time_attn = TimeAttention(emb_dim, num_heads)
        self.space_attn = SpaceAttention(emb_dim, num_heads)
        self.norm = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )

    def forward(self, x, B, T):
        # x: (B*T, N_patches, emb_dim)
        # time attention
        # TimeAttention output shape: (N_patches, B*T, emb_dim), rearrange to (B*T, N_patches, emb_dim)
        ta_out = self.time_attn(x, B, T)
        ta_out = rearrange(ta_out, 'n (b t) c -> (b t) n c', b=B, t=T)
        x = x + ta_out

        # space attention
        sa_out = self.space_attn(x)
        x = x + sa_out

        # feed-forward
        x_norm = self.norm(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out

        return x  # (B*T, N_patches, emb_dim)


class TimeSformerTiny(nn.Module):
    def __init__(self, in_channels=1, image_size=88, patch_size=16, emb_dim=256, depth=4, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim, image_size)
        self.num_patches = self.patch_embed.num_patches

        # CLS token: 1 token per video (per batch sample)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        # Positional embedding for cls token + all patches across all frames combined
        # max_frames can be dynamic, so just create a large enough embedding and slice later
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + 1000 * self.num_patches, emb_dim))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.patch_embed(x)  # (B*T, N_patches, emb_dim)

        # reshape to (B, T, N_patches, emb_dim)
        x = rearrange(x, '(b t) n c -> b t n c', b=B, t=T)
        # flatten temporal and patch dimension for transformer input
        x = rearrange(x, 'b t n c -> b (t n) c')

        # prepend cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + T*N_patches, emb_dim)

        # add positional embedding (slice according to sequence length)
        x = x + self.pos_embed[:, :x.size(1), :]

        # flatten batch and sequence for transformer blocks
        # split cls token and tokens for transformer processing
        cls_token, tokens = x[:, :1, :], x[:, 1:, :]
        # reshape tokens to (B*T, N_patches, emb_dim)
        tokens = rearrange(tokens, 'b (t n) c -> (b t) n c', t=T, n=self.num_patches)

        # pass tokens through transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens, B, T)

        # reshape tokens back to (B, T*N_patches, emb_dim)
        tokens = rearrange(tokens, '(b t) n c -> b (t n) c', b=B, t=T)

        # concat cls token back
        x = torch.cat((cls_token, tokens), dim=1)  # (B, 1 + T*N_patches, emb_dim)

        x = self.norm(x)

        return x


