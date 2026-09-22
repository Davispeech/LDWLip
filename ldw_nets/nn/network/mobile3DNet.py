import torch
from torch import nn
import torch.nn.functional as F


__all__ = (
    "Mobile3DNet",
    "EnhancedSlowFast",
    "X3DNet_M",
    "TimeSformerLite",
)


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


# ---- 通道注意力 SE Block ----
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # [B, C, T]
        w = self.se(x)
        return x * w

# ---- 时间建模：轻量 Temporal MLP ----
class TemporalMix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=1)
        )

    def forward(self, x):  # [B, C, T]
        return x + self.mlp(x)

# ---- Slow 路径：3层 Conv3D ----
class LiteSlow(nn.Module):
    def __init__(self, in_ch=3, out_ch=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, 16, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 24, kernel_size=(3,3,3), stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(24, out_ch, kernel_size=(3,3,3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

# ---- Fast 路径：3层 Conv3D ----
class LiteFast(nn.Module):
    def __init__(self, in_ch=3, out_ch=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(6, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(8, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

# ---- 最终网络 ----
class EnhancedSlowFast(nn.Module):
    def __init__(self, num_classes=10, beta=1/8):
        super().__init__()
        slow_out = 32
        fast_out = int(slow_out * beta)

        self.slow = LiteSlow(in_ch=1, out_ch=slow_out)
        self.fast = LiteFast(in_ch=1, out_ch=fast_out)

        self.fuse_conv = nn.Conv1d(slow_out + fast_out, 256, kernel_size=1)
        self.global_se = SEBlock(256)
        self.temporal_mixer = TemporalMix(256)
        self.globalpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # [B, 3, T, H, W]->torch.Size([64, 29, 1, 88, 88])
        x = x.permute(0, 2, 1, 3, 4)
        # Subsample
        slow_in = x[:, :, ::8]   # Slow: 每8帧取1帧
        fast_in = x[:, :, ::2]   # Fast: 每2帧取1帧

        s_feat = self.slow(slow_in)  # [B, 32, T_s, 1, 1]
        f_feat = self.fast(fast_in)  # [B, 4,  T_f, 1, 1]

        s = s_feat.squeeze(-1).squeeze(-1)  # [B, 32, T]
        f = f_feat.squeeze(-1).squeeze(-1)  # [B, 4,  T]

        # 时间维对齐
        if f.shape[2] != s.shape[2]:
            s = F.interpolate(s, size=f.shape[2], mode='nearest')

        x = torch.cat([s, f], dim=1)        # [B, 36, T]

        x = self.fuse_conv(x)               # -> [B, 256, T]
        x = self.global_se(x)               # 通道注意力
        x = self.temporal_mixer(x)          # 时间建模  torch.Size([64, 256, 15])
        x = self.globalpool(x).squeeze(-1)
        return x


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


