"""LDWLip unit."""
# ########################################################
# part1: feature extract units
# part2: Activate Function
# part3: Tensor Operator
# ########################################################
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = (
    "Adapter",
)


class Adapter(nn.Module):
    """
    Bottleneck Adapter for Transformer/Conformer blocks.
    Can be used intra-block or post-encoder.

    x -> Linear(d_model, r) -> ReLU/GELU -> Linear(r, d_model) -> x + output
    """
    def __init__(self, d_model, r=64, dropout=0.0, activation='relu'):
        """
        Args:
            d_model (int): Input and output dimension
            r (int): Bottleneck dimension
            dropout (float): Dropout rate
            activation (str): 'relu' or 'gelu'
        """
        super(Adapter, self).__init__()
        self.down_proj = nn.Linear(d_model, r)
        self.up_proj = nn.Linear(r, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")

    def forward(self, x):
        """
        x: [batch, seq_len, d_model] or [batch, d_model]
        """
        z = self.down_proj(x)
        z = self.activation(z)
        z = self.dropout(z)
        z = self.up_proj(z)
        # residual connection
        return x + z
