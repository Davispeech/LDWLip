"""LDWLip unit."""
# ########################################################
# part1: feature extract units
# part2: Activate Function
# part3: Tensor Operator
# ########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = (
    "GlobalPool",
    "MaxPool",
    "AvgPool",
    "AvgPool1d",
    "AvgPool3d",
    "AdaptiveAvgPool3d",
    "Dropout3d",
    "Swish",
    "View",
    "ViewtoBatch",
    "Concat",
    "Transpose",
    "BatchNorm3d",
    "LayerNorm",
    "tensor3Dto2D",
    "nnLinear",
    "BiGRU",
    "GramMatrix_self",
    "GramMatrix_co",
)

# ########################################################
# part1: feature extract units
# ########################################################

# ########################################################
# part2: Activate Function
# ########################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class View(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        size = [x.size(0)]+self.size  # batch不变
        return x.view(size)


class Concat(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 要拼接的维度

    def forward(self, *inputs):
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = inputs[0]
        return torch.cat(inputs, dim=self.dim)


class Transpose(nn.Module):
    def __init__(self, a=1, b=2):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        x = x.transpose(self.a, self.b)
        return x


class ViewtoBatch(nn.Module):
    def forward(self, x, B):
        size = [B, -1]
        for i in range(1, len(x.size())):
            size.append(x.size()[i])
        x = x.view(size)
        return x


class LayerNorm(torch.nn.LayerNorm):
    def __init__(self, nout=256, dim=-1):
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class BatchNorm3d(nn.Module):
    def __init__(self, nout=256):
        super(BatchNorm3d, self).__init__()
        self.bn = nn.BatchNorm3d(nout)

    def forward(self, x):
        return self.bn(x)


class Dropout3d(nn.Module):
    def __init__(self, convdropout=0.5):
        super(Dropout3d, self).__init__()
        self.Dropout3d = nn.Dropout3d(convdropout)

    def forward(self, x):
        return self.Dropout3d(x)


class GlobalPool(nn.Module):
    def __init__(self, pool_type='avg', keepdim=True):
        """
        全局池化层（支持平均池化和最大池化）

        参数:
            pool_type (str): 池化类型，'avg' 或 'max'，默认为'avg'
            keepdim (bool): 是否保持维度，默认为False
        """
        super(GlobalPool, self).__init__()
        self.pool_type = pool_type.lower()
        self.keepdim = keepdim

        if self.pool_type not in ['avg', 'max']:
            raise ValueError("pool_type must be either 'avg' or 'max'")

    def forward(self, x):
        """
        输入:
            x: 形状为 (B, C, H, W) 的张量
        输出:
            池化后的张量，形状为 (B, C, 1, 1) 或 (B, C)（取决于keepdim）
        """
        if self.pool_type == 'avg':
            return nn.functional.adaptive_avg_pool2d(x, (1, 1)) if self.keepdim else x.mean(dim=[2, 3])
        else:
            return nn.functional.adaptive_max_pool2d(x, (1, 1)) if self.keepdim else x.amax(dim=[2, 3])

    def extra_repr(self):
        return f"pool_type={self.pool_type}, keepdim={self.keepdim}"


class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        """
        MaxPooling 层
        参数:
            kernel_size: 池化窗口大小 (int or tuple)
            stride: 步长 (默认等于kernel_size)
            padding: 填充大小
            dilation: 空洞卷积参数
            return_indices: 是否返回最大值位置索引
            ceil_mode: 是否使用ceil模式计算输出形状
        """
        super().__init__()
        self.maxpool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride if stride is not None else kernel_size,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode
        )

    def forward(self, x):
        return self.maxpool(x)


class AvgPool(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        """
        AvgPooling 层
        参数:
            kernel_size: 池化窗口大小 (int or tuple)
            stride: 步长 (默认等于kernel_size)
            padding: 填充大小
            ceil_mode: 是否使用ceil模式计算输出形状
            count_include_pad: 是否包含padding区域在平均计算中
        """
        super().__init__()
        self.avgpool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride if stride is not None else kernel_size,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad
        )

    def forward(self, x):
        return self.avgpool(x)


class AvgPool1d(nn.Module):
    def __init__(self, ndim=2):
        super().__init__()
        self.ndim = ndim
        self.pool = nn.AdaptiveAvgPool1d(1)  # 全局池化映射成多少维度，例如77→1

    def forward(self, x):  # torch.Size([16, 64, 9, 3, 3])
        if self.ndim == 1:
            x = x.permute(0, 2, 1)
        elif self.ndim == 0:
            x = x.permute(1, 2, 0)
        x = x.float()
        x = self.pool(x).squeeze(-1)
        return x


class AvgPool3d(nn.Module):
    def __init__(self, a, b, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(output_size=(a, b, c))

    def forward(self, x):  # torch.Size([16, 64, 9, 3, 3])
        x = self.pool(x).squeeze(2)
        return x


class AdaptiveAvgPool3d(nn.Module):
    """
    自定义的三维自适应平均池化层
    功能：将任意尺寸的 3D 输入 (depth, height, width) 池化为固定目标尺寸
    Args:
        output_size (int or tuple): 目标输出尺寸，例如 1 或 (1, 1, 1)
    """
    def __init__(self, output_size):
        super(AdaptiveAvgPool3d, self).__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size,) * 3

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入张量，形状为 (batch, channels, depth, height, width)
        Returns:
            Tensor: 池化后的张量，形状为 (batch, channels, *output_size)
        """
        # 获取输入尺寸
        _, _, D, H, W = x.shape
        target_D, target_H, target_W = self.output_size

        # 计算池化核大小和步长
        kernel_D = D // target_D if D % target_D == 0 else D // target_D + 1
        kernel_H = H // target_H if H % target_H == 0 else H // target_H + 1
        kernel_W = W // target_W if W % target_W == 0 else W // target_W + 1

        stride_D = D // target_D
        stride_H = H // target_H
        stride_W = W // target_W

        # 使用平均池化
        x = F.avg_pool3d(
            x,
            kernel_size=(kernel_D, kernel_H, kernel_W),
            stride=(stride_D, stride_H, stride_W),
            padding=0
        )
        return x

    def extra_repr(self):
        return f'output_size={self.output_size}'


class BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        """
        Args:
            input_dim: 输入特征维度 (C)
            hidden_dim: GRU隐藏层维度
            num_layers: GRU层数
            dropout: 层间dropout概率
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,  # 双向GRU
            batch_first=True,  # 输入格式为 (batch, 1, feature)
            dropout=dropout if num_layers > 1 else 0
        )
        # 双向GRU输出维度是 hidden_dim*2
        self.output_dim = hidden_dim * 2

    def forward(self, x):
        """
        Input:  (B, C)
        Output: (B, hidden_dim*2)
        """
        size = x.shape
        # 添加虚拟序列维度 (B, C) -> (B, 1, C)
        if len(size) == 2:
            x = x.unsqueeze(1)  # shape: (batch, 1, input_dim)

        # GRU处理
        output, _ = self.gru(x)  # output shape: (batch, 1, hidden_dim*2)

        # 移除序列维度
        if len(size) == 2:
            return output.squeeze(1)  # shape: (batch, hidden_dim*2)
        else:
            return output

    def extra_repr(self):
        return f"input_dim={self.gru.input_size}, hidden_dim={self.gru.hidden_size}, " \
               f"num_layers={self.gru.num_layers}, bidirectional=True"


class nnLinear(nn.Module):
    def __init__(self,
                 input_dim: int = 265,
                 output_dim: int = 384,
                  ** kwargs):
        """
        特征维度扩展器
        :param input_dim: 输入特征维度 (默认265)
        :param output_dim: 输出特征维度 (默认384)
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim,  ** kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.linear.in_features, \
            f"输入特征维度错误，预期 {self.linear.in_features}，实际 {x.size(-1)}"

        # 前向传播 (自动保持前两维不变)
        return self.linear(x)  # [B, T, 384]


# ########################################################
# part3: Tensor Operator
# ########################################################
# def tensor3Dto2D(x):
#     n_batch, n_channels, s_time, sx, sy = x.shape
#     x = x.transpose(1, 2)
#     return x.reshape(n_batch * s_time, n_channels, sx, sy)
class tensor3Dto2D(nn.Module):
    def forward(self, x):
        n_batch, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2)
        x = x.reshape(n_batch * s_time, n_channels, sx, sy)
        return x


class GramMatrix_self(nn.Module):
    def __init__(self, dim=1):
        """
        Compute Gram matrix along a specified dimension.

        Args:
            dim (int): Dimension along which to compute Gram matrix.
                      dim=1: Compute B × T × T (default).
                      dim=2: Compute B × C × C.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Gram matrix of shape:
                         - (B, T, T) if dim=1.
                         - (B, C, C) if dim=2.
        """
        if self.dim == 1:
            # Compute Gram matrix along time dimension (B × T × T)
            x_T = x.transpose(1, 2)  # (B, C, T)
            gram = torch.matmul(x, x_T)  # (B, T, T)
        elif self.dim == 2:
            # Compute Gram matrix along channel dimension (B × C × C)
            x_T = x.transpose(2, 1)  # (B, T, C)
            gram = torch.matmul(x_T, x)  # (B, C, C)
        else:
            raise ValueError(f"dim must be 1 or 2, got {self.dim}")

        return gram


class GramMatrix_co(nn.Module):
    def __init__(self, dim=1):
        """
        Compute cross-Gram matrix between x1 and x2 along a specified dimension.

        Args:
            dim (int): Dimension along which to compute correlations.
                      dim=1: Compute B × T × T (default).
                      dim=2: Compute B × C × C.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): Input tensor of shape (B, T, C).
            x2 (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Cross-Gram matrix of shape:
                         - (B, T, T) if dim=1 (time-wise correlations).
                         - (B, C, C) if dim=2 (channel-wise correlations).
        """
        if len(x1.shape)==2:
            x1 = x1.unsqueeze(-1)
        if len(x2.shape)==2:
            x2 = x2.unsqueeze(-1)
        if x1.shape != x2.shape:
            raise ValueError(f"x1 and x2 must have the same shape, got {x1.shape} and {x2.shape}")

        if self.dim == 1:
            # Compute time-wise cross-correlations: (B, T, C) @ (B, C, T) -> (B, T, T)
            x2_T = x2.transpose(1, 2)  # (B, C, T)
            gram = torch.matmul(x1, x2_T)
        elif self.dim == 2:
            # Compute channel-wise cross-correlations: (B, C, T) @ (B, T, C) -> (B, C, C)
            x1_T = x1.transpose(1, 2)  # (B, C, T)
            gram = torch.matmul(x1_T, x2)
        else:
            raise ValueError(f"dim must be 1 or 2, got {self.dim}")

        return gram
