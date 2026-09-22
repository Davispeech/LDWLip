import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


__all__ = (
    "LipAuthOutput",
)


class LipAuthOutput(nn.Module):
    def __init__(self, root_dir: str, support_path: str, id_list: List[str], save_layerFeatures: List[float], auth_type="eval"):
        super().__init__()
        """
        初始化 LipAuthOutput 类
        :param support_path: 注册用户特征存储目录
        :param id_list: 注册用户ID列表
        :param threshold: 认证相似度阈值
        """
        self.root_dir = root_dir
        # 'data/lipauth_support/VSR_M1/5_grid_multitask_test'

        if root_dir not in support_path:
            self.support_path = root_dir + "/" +support_path
        else:
            self.support_path = support_path
        self.id_list = id_list  # [1,2,20,22]
        # [['3', 'ctc', '0.75'], ['5', 'cos', '0.75']]
        self.save_layerFeatures = save_layerFeatures

        # 确保目录存在
        if auth_type == "demo":
            self.registered_features = self.load_registered_VSA()

    
