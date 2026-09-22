import numpy as np

import torch
from torch import nn


__all__ = (
    "FlowNet",
)


class FlowNet(nn.Module):
    def __init__(self,
                 flow_type='farneback',
                 normalize=True,
                 flow_scaling=20.0):
        """
        光流特征提取器

        参数:
            flow_type: 光流算法类型 ('farneback' 或 'liteflownet')
            normalize: 是否归一化光流到[-1,1]
            flow_scaling: 光流值缩放因子（用于可视化）
        """
        super().__init__()
        self.flow_type = flow_type
        self.normalize = normalize
        self.flow_scaling = flow_scaling

        if flow_type == 'liteflownet':
            # 需要安装liteflownet: pip install liteflownet
            from liteflownet import LiteFlowNet
            self.flownet = LiteFlowNet()
        else:
            # 使用OpenCV的Farneback光流（需要CPU计算）
            self.flownet = None

    def forward(self, x):
        """
        输入:
            x: 视频张量 [B, T, C, H, W] (值范围[0,1])
        输出:
            flow_features: 光流特征 [B, 2*(T-1), H, W]
        """
        B, T, C, H, W = x.shape
        device = x.device

        # 转换为灰度图 (如果输入是RGB)
        if C == 3:
            x_gray = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]  # [B, T, H, W]
        else:
            x_gray = x.squeeze(2)  # [B, T, H, W]

        # 初始化光流输出张量
        flow_features = torch.zeros(B, 2 * (T - 1), H, W, device=device)

        for b in range(B):
            for t in range(T - 1):
                # 获取相邻帧
                img1 = x_gray[b, t].cpu().numpy()  # [H, W]
                img2 = x_gray[b, t + 1].cpu().numpy()  # [H, W]

                # 计算光流
                if self.flow_type == 'farneback':
                    import cv2
                    flow = cv2.calcOpticalFlowFarneback(
                        img1, img2, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2,
                        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
                    )  # [H, W, 2] (dx, dy)
                else:
                    flow = self.flownet(img1, img2)  # [H, W, 2]

                # 转换为Tensor并缩放
                flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).to(device)  # [2, H, W]

                if self.normalize:
                    # 归一化到[-1,1]
                    max_val = torch.max(torch.abs(flow_tensor)) + 1e-5
                    flow_tensor = flow_tensor / max_val
                else:
                    # 按固定因子缩放
                    flow_tensor = flow_tensor / self.flow_scaling

                # 存储结果
                flow_features[b, 2 * t:2 * t + 2] = flow_tensor

        return flow_features

    def visualize_flow(self, flow):
        """
        可视化光流 (返回RGB图像)
        输入: flow [2, H, W] (dx, dy)
        输出: flow_rgb [3, H, W]
        """
        import cv2
        flow_np = flow.permute(1, 2, 0).cpu().numpy()
        hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return torch.from_numpy(flow_rgb).permute(2, 0, 1).float() / 255.0
