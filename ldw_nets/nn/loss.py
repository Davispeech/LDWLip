import logging
from distutils.version import LooseVersion
from ldw_nets.utils import to_device

import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = (
    "loss_ctc",
    "loss_att",
    "loss_total",
    "LossRelation",
    "nnCrossEntropy",
    "nnFocalLoss",
    "ArcFaceLoss",
    "loss_CrossEntropy",
    "loss_SoftLabelCE",
    "loss_KLDiv",
    "loss_CosineSimilarity",
    "loss_MSE",
    "loss_L1",
    "loss_rc",
)


class loss_total(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        # self.layer_ids = layer_ids

    def forward(self, xs):
        if len(xs) != len(self.weights):
            logging.info("Loss Num Error!!!")
        loss_total_ = []
        for i in range(len(xs)):
            loss_total_.append(xs[i]*self.weights[i])
        return sum(loss_total_)


class loss_rc(nn.Module):
    def __init__(self, weights_r, weights_c):
        super().__init__()
        self.weights = [weights_r, weights_c]

    def forward(self, loss_csc_r, loss_csc_c):
        rc = self.weights[0] * loss_csc_r + self.weights[1] * loss_csc_c
        return rc


class LossRelation(nn.Module):
    def __init__(self, matrix='all-ones', distance='mse', normalize=True):
        """
        Compute loss between input x and a target matrix (all-ones or identity).

        Args:
            matrix (str): Target matrix type. Options: 'all-ones', 'identity'.
            distance (str): Distance metric. Options: 'mse', 'mae'.
            normalize (bool): If True, normalize input x and target matrix to [0, 1] range.
        """
        super().__init__()
        self.matrix = matrix
        self.distance = distance
        self.normalize = normalize

        # Initialize loss function
        if distance == 'mse':
            self.loss_fn = nn.MSELoss()
        elif distance == 'mae':
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"distance must be 'mse' or 'mae', got {distance}")

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, T) or (B, C, C).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        B, dim1, dim2 = x.shape
        if dim1 != dim2:
            raise ValueError(f"Input x must be square in last two dimensions, got shape {x.shape}")

        # Generate target matrix
        if self.matrix == 'all-ones':
            target = torch.ones_like(x)
        elif self.matrix == 'identity':
            target = torch.eye(dim1, device=x.device).unsqueeze(0).expand(B, -1, -1)
        else:
            raise ValueError(f"matrix must be 'all-ones' or 'identity', got {self.matrix}")

        # Normalize x and target to [0, 1] range if enabled
        if self.normalize:
            x = self._normalize(x)

        # Compute loss
        loss = self.loss_fn(x, target)
        return loss

    def _normalize(self, x):
        """Normalize input tensor to [0, 1] range."""
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min + 1e-8)  # Avoid division by zero


class nnCrossEntropy_src(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, out, labels):
        target = labels.view(-1)
        loss = self.loss_fn(out, target)
        return loss


class nnCrossEntropy(nn.Module):
    def __init__(self, ignore_index=-1, gamma=2.0):
        """
        gamma: Focal Loss 调节参数，越大越强调难分类样本
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, out, labels):
        """
        out: [B, num_classes]
        labels: [B]
        """
        target = labels.view(-1)
        logits = out.view(-1, out.size(-1))
        # 计算每个样本的原始交叉熵
        ce = self.ce_loss(logits, target)  # [B]

        # 计算 softmax 概率
        pt = torch.exp(-ce)  # 正确类别的预测概率
        # Focal Loss: 难分类样本（pt小）权重更大
        focal_weight = (1 - pt) ** self.gamma
        loss = (focal_weight * ce).mean()
        return loss


class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.randn(embedding_size, num_classes))
        nn.init.xavier_uniform_(self.W)

        # 预计算常数
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(torch.pi) - m)
        self.mm = torch.sin(torch.tensor(torch.pi) - m) * m

    def forward(self, x, labels):
        labels = labels.squeeze()
        # L2 normalize
        x = F.normalize(x, dim=1)
        # W = F.normalize(self.W, dim=0)
        W = F.normalize(self.W.to(x.device), dim=0)

        cos_theta = torch.matmul(x, W)  # [B, num_classes]

        cos_theta = cos_theta.clamp(-1, 1)

        # 计算 sinθ，避免 acos
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))

        # cos(theta + m) = cosθ * cos(m) - sinθ * sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # 保证数值稳定 (ArcFace trick)
        cond = cos_theta - self.th
        cos_theta_m = torch.where(cond > 0, cos_theta_m, cos_theta - self.mm)

        # 构造最终 logits
        one_hot = F.one_hot(labels, num_classes=cos_theta.size(1)).float().to(x.device)
        logits = cos_theta * (1 - one_hot) + cos_theta_m * one_hot
        logits *= self.s

        loss = F.cross_entropy(logits, labels)
        return loss


class nnFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 可以是None、float 或 list/tensor（每类权重）
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, labels):
        """
        inputs: (N, C) logits，未经过softmax
        targets: (N,) 类别索引
        """
        targets = labels.view(-1)
        # 去除 ignore_index 的位置
        if self.ignore_index >= 0:
            valid_mask = (targets != self.ignore_index)
            inputs = inputs[valid_mask]
            targets = targets[valid_mask]

        log_probs = F.log_softmax(inputs, dim=1)  # (N, C)
        probs = torch.exp(log_probs)              # (N, C)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        pt = (probs * targets_one_hot).sum(dim=1)      # (N,)
        log_pt = (log_probs * targets_one_hot).sum(dim=1)

        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                alpha = torch.tensor(self.alpha).to(inputs.device)[targets]  # 每个样本的α
            else:
                alpha = self.alpha
        else:
            alpha = 1.0

        loss = -alpha * (1 - pt) ** self.gamma * log_pt  # focal loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 'none'


class loss_CrossEntropy(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-ignore_index)

    def forward(self, out, labels):
        out = out.view(-1, out.shape[-1])
        labels = labels.reshape(-1)
        loss = self.loss_fn(out, labels)
        return loss


class loss_SoftLabelCE(nn.Module):
    def __init__(self, temperature: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, student_logits, teacher_logits):  # 修正拼写错误
        B, T, _ = student_logits.shape
        # 教师概率分布（无需梯度）
        teacher_probs = F.softmax(teacher_logits.detach() / self.temperature, dim=-1)
        # 学生对数概率（数值稳定）
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        # 计算交叉熵损失
        loss = -torch.sum(teacher_probs * student_log_probs) / (B * T)
        return loss


class loss_KLDiv(nn.Module):
    def __init__(self, temperature: float = 3.0, eps: float = 1e-8):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.temperature = temperature
        self.eps = eps
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor) -> torch.Tensor:
        # 温度缩放
        scaled_student = student_logits / self.temperature
        scaled_teacher = teacher_logits / self.temperature

        # 教师概率分布（数值稳定处理）
        teacher_probs = F.softmax(scaled_teacher.detach(), dim=-1)
        # 显式指定设备，避免跨设备问题
        eps_tensor = torch.tensor(self.eps, device=teacher_probs.device)
        teacher_probs = teacher_probs.clamp(min=eps_tensor)

        # 学生对数概率（无需额外clamp）
        student_log_probs = F.log_softmax(scaled_student, dim=-1)

        # 计算KL散度损失
        loss = self.kl_loss(student_log_probs, teacher_probs)
        return loss * (self.temperature ** 2)


class loss_CosineSimilarity(nn.Module):
    def __init__(self, dim=1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps  # 用于数值稳定性

    def forward(self, x, x_aux):
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(x_aux, x, dim=self.dim)
        # 转换为损失：最大化余弦相似度等价于最小化负的相似度
        loss = -cos_sim.mean()

        return loss


class loss_MSE(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, x, x_aux):
        loss = self.loss_fn(x, x_aux)
        return loss


class loss_L1(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, x, x_aux):
        loss = self.loss_fn(x, x_aux)
        return loss


class loss_ctc(torch.nn.Module):
    """CTC module
    :param int odim: dimension of outputs
    :param int eprojs: number of encoder projection units
    :param float dropout_rate: dropout rate (0.0 ~ 1.0)
    :param str ctc_type: builtin or warpctc
    :param bool reduce: reduce the CTC loss into a scalar
    """
    def __init__(self, odim, eprojs, dropout_rate, ctc_type="builtin", reduce=True):
        super().__init__()
        self.loss = None
        self.probs = None  # for visualization

        # In case of Pytorch >= 1.7.0, CTC will be always builtin
        self.ctc_type = (
            ctc_type
            if LooseVersion(torch.__version__) < LooseVersion("1.7.0")
            else "builtin"
        )

        if ctc_type != self.ctc_type:
            logging.debug(f"CTC was set to {self.ctc_type} due to PyTorch version.")

        if self.ctc_type == "builtin":
            reduction_type = "sum" if reduce else "none"
            self.ctc_loss = torch.nn.CTCLoss(
                reduction=reduction_type, zero_infinity=True
            )
        elif self.ctc_type == "cudnnctc":
            reduction_type = "sum" if reduce else "none"
            self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)
        elif self.ctc_type == "warpctc":
            import warpctc_pytorch as warp_ctc
            self.ctc_loss = warp_ctc.CTCLoss(size_average=True, reduce=reduce)
        else:
            raise ValueError(
                'ctc_type must be "builtin" or "warpctc": {}'.format(self.ctc_type)
            )
        self.ignore_id = -1
        self.reduce = reduce

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen):
        if self.ctc_type in ["builtin", "cudnnctc"]:
            th_pred = th_pred.log_softmax(2)
            # Use the deterministic CuDNN implementation of CTC loss to avoid
            #  [issue#17798](https://github.com/pytorch/pytorch/issues/17798)
            with torch.backends.cudnn.flags(deterministic=True):
                loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)

            # Batch-size average
            loss = loss / th_pred.size(1)
            return loss
        elif self.ctc_type == "warpctc":
            return self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
        elif self.ctc_type == "gtnctc":
            targets = [t.tolist() for t in th_target]
            log_probs = torch.nn.functional.log_softmax(th_pred, dim=2)
            return self.ctc_loss(log_probs, targets, th_ilen, 0, "none")
        else:
            raise NotImplementedError

    def forward(self, ys_hat, hlens, ys_pad):
        """CTC forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad:
            batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        """
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        # zero padding for hs
        if self.ctc_type == "builtin":
            olens = to_device(ys_hat, torch.LongTensor([len(s) for s in ys]))
            hlens = hlens.long()
            ys_pad = torch.cat(ys)  # without this the code breaks for asr_mix
            self.loss = self.loss_fn(ys_hat, ys_pad, hlens, olens)
        else:
            logging.info("loss_ctc type error!!!!!!!")

        # get length info
        if self.reduce:
            # NOTE: sum() is needed to keep consistency
            # since warpctc return as tensor w/ shape (1,)
            # but builtin return as tensor w/o shape (scalar).
            self.loss = self.loss.sum()
            # logging.debug("ctc loss:" + str(float(self.loss)))
        return self.loss


class loss_att(nn.Module):  # LabelSmoothingLoss
    """Label-smoothing loss.

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(
        self,
        size,
        padding_idx,
        smoothing,
        normalize_length=False,
        criterion=nn.KLDivLoss(reduction="none"),
    ):
        """Construct an LabelSmoothingLoss object."""
        super(loss_att, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target):
        """Compute loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target:
            target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom

