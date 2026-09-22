import logging
import numpy as np
from typing import Any, List, Tuple, Dict
from distutils.version import LooseVersion
import six
import torch
import torch.nn as nn
import torch.nn.functional as F


from ldw_nets.nn.unit import LayerNorm
from ldw_nets.nn.block import PositionalEncoding, PositionwiseFeedForward, MultiHeadedAttention
from ldw_nets.utils import repeat
from ldw_nets.utils import subsequent_mask

# from ldw_nets.lm.ctc_scores import CTCPrefixScorer
# from ldw_nets.lm.utils import BatchBeamSearch, get_model_conf, dynamic_import_lm, torch_load
from ldw_nets.scorer_interface import BatchScorerInterface
# from ldw_nets.scorers.length_bonus import LengthBonus


__all__ = (
    "ClassifierOutput",
)


class ClassifierOutput(nn.Module):
    """
    501 类分类后处理
    输入: logits  [B, 501]
    输出: dict{
        'pred_idx'  : [B]        top-1 类别索引
        'prob'      : [B, 501]   概率
        'top5_idx'  : [B, 5]     top-5 索引
        'top5_prob' : [B, 5]     top-5 概率
    }
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, topk: int = 5) -> Dict[str, torch.Tensor]:
        prob = torch.softmax(logits, dim=1)              # [B, 501]
        pred_idx = logits.argmax(dim=1)                  # [B]
        top5_prob, top5_idx = prob.topk(topk, dim=1)     # [B, 5]

        return {
            'pred_idx'  : pred_idx,
            'prob'      : prob,
            'top5_idx'  : top5_idx,
            'top5_prob' : top5_prob,
        }

