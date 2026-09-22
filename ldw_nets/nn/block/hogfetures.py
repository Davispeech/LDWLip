import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
from typing import Tuple, Optional


__all__ = (
    "HOGLayer",
    "HOGFeatureExtractor",
)


class HOGLayer(nn.Module):
    """
    PyTorch 实现的 HOG 特征提取层，支持 batch 输入。
    参考论文：Dalal & Triggs, CVPR 2005
    """

    def __init__(self, nbins=9, cell_size=8, win_size=64, stride=8):
        super().__init__()
        self.nbins = nbins
        self.cell_size = cell_size
        self.win_size = win_size
        self.stride = stride

        # 定义梯度计算核
        self.kernel_x = torch.tensor([[-1, 0, 1]], dtype=torch.float32).view(1, 1, 1, 3)
        self.kernel_y = torch.tensor([[-1], [0], [1]], dtype=torch.float32).view(1, 1, 3, 1)

    def forward(self, x):
        """
        输入: (B, C, H, W) 的 batch 图像 (建议 C=1 或 C=3)
        输出: (B, nbins, H//cell_size, W//cell_size) 的 HOG 特征
        """
        if x.dim() != 4:
            raise ValueError("Input must be a 4D tensor (B, C, H, W)")

        B, C, H, W = x.shape

        # 1. 计算梯度 (Sobel 算子)
        grad_x = F.conv2d(x, self.kernel_x.repeat(C, 1, 1, 1).to(x.device),
                          padding=(0, 1), groups=C)
        grad_y = F.conv2d(x, self.kernel_y.repeat(C, 1, 1, 1).to(x.device),
                          padding=(1, 0), groups=C)

        # 2. 计算梯度幅值和角度 (合并多通道)
        magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2)).sum(dim=1)  # (B, H, W)
        angle = torch.atan2(grad_y.sum(dim=1), grad_x.sum(dim=1))  # (B, H, W)
        angle = (angle + np.pi) % np.pi  # 转换到 [0, pi]

        # 3. 计算每个 cell 的直方图
        bin_size = np.pi / self.nbins
        bin_idx = torch.floor(angle / bin_size).long() % self.nbins  # (B, H, W)

        # 展开为 cell 粒度
        cells = F.unfold(magnitude.unsqueeze(1),
                         kernel_size=self.cell_size,
                         stride=self.stride)  # (B, cell_size*cell_size, num_cells)
        cells = cells.view(B, self.cell_size * self.cell_size, -1)  # (B, cell_area, num_cells)

        bin_idx_cells = F.unfold(bin_idx.float().unsqueeze(1),
                                 kernel_size=self.cell_size,
                                 stride=self.stride).long()  # (B, cell_area, num_cells)

        # 4. 聚合直方图
        hog_feats = torch.zeros(B, self.nbins, cells.shape[-1]).to(x.device)
        for b in range(B):
            hog_feats[b].scatter_add_(0, bin_idx_cells[b], cells[b])

        # 5. 调整形状为 (B, nbins, H_out, W_out)
        H_out = (H - self.cell_size) // self.stride + 1
        W_out = (W - self.cell_size) // self.stride + 1
        hog_feats = hog_feats.view(B, self.nbins, H_out, W_out)

        # 6. L2 归一化 (按 block)
        eps = 1e-6
        hog_feats = F.normalize(hog_feats, p=2, dim=1, eps=eps)

        return hog_feats


class HOGFeatureExtractor(nn.Module):
    """
    完整的 HOG 特征提取器，包含可训练的 MLP
    """

    def __init__(self, img_size=88, nbins=9, cell_size=8, feat_dim=512):
        super().__init__()
        self.hog = HOGLayer(nbins=nbins, cell_size=cell_size)

        # 计算 HOG 特征维度
        hog_out_size = nbins * ((img_size - cell_size) // cell_size + 1) ** 2

        # 可训练的 MLP
        self.mlp = nn.Sequential(
            nn.Linear(hog_out_size, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        输入: (B, C, H, W) 的 batch 图像
        输出: (B, feat_dim) 的特征向量
        """
        # 1. 计算 HOG 特征
        # x -> torch.Size([15, 1, 88, 88])
        hog_feats = self.hog(x)  # (B, nbins, H_out, W_out)  -> torch.Size([15, 9, 11, 11])

        # 2. 展平并送入 MLP
        B = x.shape[0]
        hog_feats = hog_feats.view(B, -1)  # (B, nbins*H_out*W_out)
        return self.mlp(hog_feats)  # (B, feat_dim)


class HOGFeatureExtractor2(nn.Module):
    def __init__(self,
                 cell_size: int = 8,
                 block_size: int = 2,
                 num_bins: int = 9,
                 gamma_correction: bool = True,
                 device: Optional[torch.device] = None):
        """
        HOG 特征提取器

        参数:
            cell_size: 每个cell的像素大小 (默认8x8)
            block_size: 每个block包含的cell数量 (默认2x2)
            num_bins: 方向梯度直方图的bin数量 (默认9)
            gamma_correction: 是否应用gamma校正 (默认True)
            device: 计算设备 (默认None，自动选择)
        """
        super().__init__()
        self.cell_size = cell_size
        self.block_size = block_size
        self.num_bins = num_bins
        self.gamma_correction = gamma_correction
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 预计算梯度方向的cos和sin值
        self.bin_centers = torch.linspace(0, math.pi, num_bins + 1)[:-1].to(self.device)
        self.cos_bins = torch.cos(self.bin_centers).view(1, 1, -1)
        self.sin_bins = torch.sin(self.bin_centers).view(1, 1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取HOG特征

        参数:
            x: 输入图像张量 (B, C, H, W) 或 (C, H, W)

        返回:
            HOG特征张量 (B, num_blocks, num_blocks, num_bins * block_size**2) 或
                       (num_blocks, num_blocks, num_bins * block_size**2)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # 1. 转换为灰度图
        if x.size(1) == 3:
            x = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
            x = x.unsqueeze(1)  # (B, 1, H, W)

        # 2. Gamma校正
        x = torch.clamp(x, min=0)
        if self.gamma_correction:
            x = torch.sqrt(x)

        # 3. 计算梯度
        x = x.to(self.device)
        kernel_x = torch.tensor([[-1, 0, 1]], dtype=torch.float32, device=self.device).view(1, 1, 1, 3)
        kernel_y = torch.tensor([[-1], [0], [1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 1)

        grad_x = F.conv2d(x, kernel_x, padding=(0, 1))
        grad_y = F.conv2d(x, kernel_y, padding=(1, 0))

        # 4. 计算梯度幅值和方向
        magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
        orientation = torch.atan2(grad_y, grad_x)  # 范围 [-π, π]
        orientation = torch.where(orientation < 0, orientation + math.pi, orientation)  # 转换到 [0, π]

        # 5. 计算每个cell的直方图
        B, _, H, W = x.shape
        cell_h = H // self.cell_size
        cell_w = W // self.cell_size

        # 将方向分配到bin
        bin_idx = torch.floor(orientation / math.pi * self.num_bins).long()
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)

        # 创建cell直方图 (B, cell_h, cell_w, num_bins)
        cell_hist = torch.zeros((B, cell_h, cell_w, self.num_bins), device=self.device)

        for b in range(B):
            for i in range(cell_h):
                for j in range(cell_w):
                    y_start = i * self.cell_size
                    y_end = y_start + self.cell_size
                    x_start = j * self.cell_size
                    x_end = x_start + self.cell_size

                    cell_mag = magnitude[b, 0, y_start:y_end, x_start:x_end]
                    cell_bin = bin_idx[b, 0, y_start:y_end, x_start:x_end]

                    for bin_num in range(self.num_bins):
                        cell_hist[b, i, j, bin_num] = torch.sum(cell_mag * (cell_bin == bin_num).float())

        # 6. 归一化block
        block_h = cell_h - self.block_size + 1
        block_w = cell_w - self.block_size + 1
        block_features = torch.zeros((B, block_h, block_w, self.num_bins * self.block_size * self.block_size),
                                     device=self.device)

        for b in range(B):
            for i in range(block_h):
                for j in range(block_w):
                    block = cell_hist[b, i:i + self.block_size, j:j + self.block_size, :]
                    block_flat = block.reshape(-1)
                    eps = 1e-5
                    block_norm = block_flat / (torch.norm(block_flat) + eps)
                    block_features[b, i, j, :] = block_norm

        if x.dim() == 3:
            return block_features.squeeze(0)
        return block_features

    def compute_descriptor_size(self, image_size: Tuple[int, int]) -> Tuple[int, int, int]:
        """
        计算给定图像尺寸的HOG描述符大小

        参数:
            image_size: (height, width) 元组

        返回:
            (block_h, block_w, feature_dim) 元组
        """
        cell_h = image_size[0] // self.cell_size
        cell_w = image_size[1] // self.cell_size
        block_h = cell_h - self.block_size + 1
        block_w = cell_w - self.block_size + 1
        feature_dim = self.num_bins * self.block_size * self.block_size
        return (block_h, block_w, feature_dim)

