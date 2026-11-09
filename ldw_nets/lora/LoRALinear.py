import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA wrapper for a Linear layer.
    Only adds a low-rank delta (B @ A) to the original weight.
    """
    def __init__(self, linear: nn.Linear, r=4, alpha=8, dropout=0.0):
        super().__init__()
        self.linear = linear  # 原始 Linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.scale = alpha / r

        # LoRA 低秩参数
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        # 初始化 B 为 0 保证初始 forward 与原 Linear 一致
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.scale * self.lora_B(self.lora_A(self.lora_dropout(x)))


class LoRAMoE(nn.Module):
    def __init__(self, linear: nn.Linear, num_experts=4, r=4, alpha=8, dropout=0.0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.num_experts = num_experts

        # 每个专家都是 LoRALinear，传入同一个原 Linear
        self.experts = nn.ModuleList([
            LoRALinear(linear, r=r+2*i, alpha=alpha+2*i, dropout=dropout)
            for i in range(num_experts)
        ])
        self.gate = nn.Linear(self.in_features, num_experts)

    def forward(self, x):
        """
        x: [B, T, C]
        """
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)  # [B*T, C]

        # gate 权重
        gate_scores = F.softmax(self.gate(x_flat), dim=-1).unsqueeze(-1)  # [B*T, num_experts, 1]

        # 每个专家输出
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)  # [B*T, num_experts, out_features]

        # gate 加权
        out_flat = torch.sum(expert_outputs * gate_scores, dim=1)  # [B*T, out_features]

        # reshape 回 [B, T, out_features]
        out = out_flat.view(B, T, self.out_features)
        return out


class _SameRankLoRA(nn.Module):
    """Low-rank residual used by every paper expert. Rank does not grow with the expert index."""

    def __init__(self, in_features, out_features, r, alpha, dropout=0.0):
        super().__init__()
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.scale * self.lora_B(self.lora_A(self.lora_dropout(x)))


class LoRAMoEPaper(nn.Module):
    """Paper LoRA-MoE: K same-rank experts and one utterance-level router.

    The original linear stays frozen. The router is named lora_gate so the
    existing train/save rule, which keeps only names containing lora_, both
    updates it and writes it into the adapter checkpoint.
    """

    def __init__(self, linear: nn.Linear, num_experts=3, r=4, alpha=8, dropout=0.0):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.num_experts = num_experts
        step = int(os.environ.get("EAAI_MOE_RANK_STEP", "0") or "0")
        self.experts = nn.ModuleList([
            _SameRankLoRA(
                self.in_features, self.out_features, r + step * i, alpha + step * i, dropout)
            for i in range(num_experts)
        ])
        if step and not getattr(LoRAMoEPaper, "_rank_logged", False):
            ranks = [r + step * i for i in range(num_experts)]
            alphas = [alpha + step * i for i in range(num_experts)]
            print("LoRAMoEPaper expert ranks %s alphas %s" % (ranks, alphas), flush=True)
            LoRAMoEPaper._rank_logged = True
        self.lora_gate = nn.Linear(self.in_features, num_experts)

    def forward(self, x):
        """x: [B, T, C]. One routing weight per utterance, shared across frames.

        _stage is None for the original joint training. The Table 8 run sets it:
        0 trains expert 0 on every clip; 1 freezes that expert and updates one of
        the remaining experts per batch, chosen by _slot.
        """
        pooled = x.mean(dim=1)
        gates = F.softmax(self.lora_gate(pooled), dim=-1)
        stage = getattr(self, "_stage", None)
        active = getattr(self, "_active", None)
        if stage == 0 and active is None:
            return self.linear(x) + self.experts[0](x)
        residual = 0
        if active is None and stage == 1 and self.num_experts > 1:
            active = 1 + (int(getattr(self, "_slot", 0)) % (self.num_experts - 1))
        if active is None:
            active_ids = None
        elif isinstance(active, int):
            active_ids = (active,)
        else:
            active_ids = tuple(active)
        for index, expert in enumerate(self.experts):
            expert_out = expert(x)
            if active_ids is not None and index not in active_ids:
                expert_out = expert_out.detach()
            residual = residual + gates[:, index].view(-1, 1, 1) * expert_out
        return self.linear(x) + residual



