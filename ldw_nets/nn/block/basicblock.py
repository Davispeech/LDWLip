import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ldw_nets.nn.unit import Swish


__all__ = (
    "Conv",  # CBS
    "Conv3D",
    "Conv3DBlock",
    "Conv3Dfrontend",
    "SpatioTemporalConv",
    "DownsampleCB",  # downsample_basic_block
    "DownsampleCB_1D",
    "ResidualBlock",
    "ResidualBlock131",
    "ResidualBlock_1D",
    "ResidualBlock131_1D",
    "ConvDictBlock",
)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv3D(nn.Module):
    """Standard 3D convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # 默认激活函数
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        初始化3D卷积层
        参数:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小 (int或tuple)
            s: 步长 (int或tuple)
            p: 填充 (自动计算或指定)
            g: 分组卷积组数
            d: 空洞率
            act: 激活函数 (True=默认, False=None, 或指定nn.Module)
        """
        super().__init__()
        self.conv = nn.Conv3d(c1, c2, k, s, self.autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm3d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):  # (B, C_in, D, H, W)
        """应用卷积、批归一化和激活函数"""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """跳过批归一化的前向传播"""
        return self.act(self.conv(x))

    @staticmethod
    def autopad(k, p=None, d=1):
        """
        自动计算填充大小以保持空间维度
        参数:
            k: 卷积核大小 (int或tuple)
            p: 填充值 (None则自动计算)
            d: 空洞率
        返回:
            填充值 (tuple)
        """
        if p is None:
            if isinstance(k, int):
                k = (k, k, k)
            p = (k[0] // 2 * d, k[1] // 2 * d, k[2] // 2 * d)
        return p


class Conv3DBlock(torch.nn.Module):
    def __init__(self, nin=64, nout=64, frames_3d=3, kenerl_3d=3):
        super(Conv3DBlock, self).__init__()
        self.Conv3Dblock = nn.Sequential(
            nn.Conv3d(
                nin, nout, (frames_3d, kenerl_3d, kenerl_3d), (1, 2, 2), (2, 3, 3), bias=False
            ),
            nn.BatchNorm3d(nout),
            Swish(),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
        )
        self.in_channels =nin

    def forward(self, xs_pad):
        if self.in_channels != xs_pad.shape[1]:
            xs_pad = xs_pad.transpose(1, 2)  # [B, T, C, H, W] -> [B, C, T, H, W]
        # B, C, T, H, W = x.shape

        xs_pad = self.Conv3Dblock(xs_pad)
        return xs_pad


# self.frontend3D = nn.Sequential(
#             nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
#             nn.BatchNorm3d(self.frontend_nout),
#             frontend_relu,
#             nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))



class Conv3Dfrontend(torch.nn.Module):
    def __init__(self, frontend_nout=64, frames_3d=5, kenerl_3d=7, relu_type='swish'):
        super(Conv3Dfrontend, self).__init__()
        self.frontend_nout = frontend_nout

        if relu_type == 'relu':
            frontend_relu = nn.ReLU(True)
        elif relu_type == 'prelu':
            frontend_relu = nn.PReLU(self.frontend_nout)
        elif relu_type == 'swish':
            frontend_relu = Swish()
        else:
            raise ValueError("frontend_relu???")

        self.Conv3Dblock = nn.Sequential(
            nn.Conv3d(
                1, self.frontend_nout, (frames_3d, kenerl_3d, kenerl_3d), (1, 2, 2), (2, 3, 3), bias=False
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
        )

    def forward(self, xs_pad):
        xs_pad = xs_pad.transpose(1, 2)  # [B, T, C, H, W] -> [B, C, T, H, W]
        xs_pad = self.Conv3Dblock(xs_pad)
        return xs_pad


class SpatioTemporalConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, spatial_kernel_size=3, temporal_kernel_size=3):
        """
        动态时空分离卷积
        Args:
            in_channels: 输入通道数（与时间维度无关）
            out_channels: 输出通道数
            spatial_kernel_size: 空间卷积核大小（默认3）
            temporal_kernel_size: 时间卷积核大小（默认3）
        """
        super(SpatioTemporalConv, self).__init__()
        self.in_channels = in_channels
        # 空间卷积 (1×kh×kw)
        self.spatial_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, spatial_kernel_size, spatial_kernel_size),
            padding=(0, spatial_kernel_size // 2, spatial_kernel_size // 2),
            bias=False
        )

        # 时间卷积 (kt×1×1) - 使用分组卷积实现通道独立的时间建模
        self.temporal_conv = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(temporal_kernel_size, 1, 1),
            padding=(temporal_kernel_size // 2, 0, 0),
            groups=out_channels,  # 深度可分离卷积
            bias=True
        )

        # 自适应时间维度处理
        self.temporal_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # 保留时间维度

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.in_channels != x.shape[1]:
            x = x.transpose(1, 2)  # [B, T, C, H, W] -> [B, C, T, H, W]
        # B, C, T, H, W = x.shape

        # 空间特征提取 (保持时间维度不变)
        x = self.spatial_conv(x)  # [B, Cout, T, H, W]

        # 时间特征提取（动态适应任意T）
        x = self.temporal_conv(x)  # [B, Cout, T, H, W]
        return x


def DownsampleCB_1D(inplanes, outplanes, stride):
    """downsample_basic_block.
    :param inplanes: int, number of channels in the input sequence.
    :param outplanes: int, number of channels produced by the convolution.
    :param stride: int, size of the convolving kernel.
    """
    return nn.Sequential(
        nn.Conv1d(
            inplanes,
            outplanes,
            kernel_size=1,
            stride=stride,
            bias=False,
        ),
        nn.BatchNorm1d(outplanes),
    )


def DownsampleCB(inplanes, outplanes, stride):
    """downsample_basic_block.
    :param inplanes: int, number of channels in the input sequence.
    :param outplanes: int, number of channels produced by the convolution.
    :param stride: int, size of the convolving kernel.
    """
    return nn.Sequential(
        nn.Conv2d(
            inplanes,
            outplanes,
            kernel_size=1,
            stride=stride,
            bias=False,
        ),
        nn.BatchNorm2d(outplanes),
    )


class ResidualBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        relu_type="swish",
    ):
        # c1, c2, num_heads, num_layers, window_size=8
        # inplanes,planes,stride=1,downsample=None,relu_type="swish",
        """__init__.
        :param inplanes: int, number of channels in the input sequence.
        :param planes: int,  number of channels produced by the convolution.
        :param stride: int, size of the convolving kernel.
        :param downsample: boolean, if True, the temporal resolution is downsampled.
        :param relu_type: str, type of activation function.
        """
        super(ResidualBlock, self).__init__()

        assert relu_type in ["relu", "prelu", "swish"]
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,)
        self.bn1 = nn.BatchNorm2d(planes)

        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        elif relu_type == "swish":
            self.relu1 = Swish()
            self.relu2 = Swish()
        else:
            raise NotImplementedError
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """forward.
        :param x: torch.Tensor, input tensor with input size (B, C, T, H, W).
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        return out


class ResidualBlock131(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        relu_type="swish",
    ):
        """__init__.
        :param inplanes: int, number of channels in the input sequence.
        :param planes: int,  number of channels produced by the convolution.
        :param stride: int, size of the convolving kernel.
        :param downsample: boolean, if True, the temporal resolution is downsampled.
        :param relu_type: str, type of activation function.
        """
        super(ResidualBlock131, self).__init__()

        assert relu_type in ["relu", "prelu", "swish"]
        planes = int(planes/4)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False,)
        self.bn1 = nn.BatchNorm2d(planes)

        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
            self.relu3 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
            self.relu3 = nn.PReLU(num_parameters=planes*4)
        elif relu_type == "swish":
            self.relu1 = Swish()
            self.relu2 = Swish()
            self.relu3 = Swish()
        else:
            raise NotImplementedError
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, stride=1, padding=0, bias=False, )
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """forward.
        :param x: torch.Tensor, input tensor with input size (B, C, T, H, W).
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu3(out)
        return out


class ResidualBlock_1D(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        relu_type="swish",
    ):
        # c1, c2, num_heads, num_layers, window_size=8
        # inplanes,planes,stride=1,downsample=None,relu_type="swish",
        """__init__.
        :param inplanes: int, number of channels in the input sequence.
        :param planes: int,  number of channels produced by the convolution.
        :param stride: int, size of the convolving kernel.
        :param downsample: boolean, if True, the temporal resolution is downsampled.
        :param relu_type: str, type of activation function.
        """
        super(ResidualBlock_1D, self).__init__()

        assert relu_type in ["relu", "prelu", "swish"]
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,)
        self.bn1 = nn.BatchNorm1d(planes)

        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        elif relu_type == "swish":
            self.relu1 = Swish()
            self.relu2 = Swish()
        else:
            raise NotImplementedError
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, )
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """forward.
        :param x: torch.Tensor, input tensor with input size (B, C, T, H, W).
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        return out


class ResidualBlock131_1D(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        relu_type="swish",
    ):
        """__init__.
        :param inplanes: int, number of channels in the input sequence.
        :param planes: int,  number of channels produced by the convolution.
        :param stride: int, size of the convolving kernel.
        :param downsample: boolean, if True, the temporal resolution is downsampled.
        :param relu_type: str, type of activation function.
        """
        super(ResidualBlock131_1D, self).__init__()

        assert relu_type in ["relu", "prelu", "swish"]
        planes = int(planes/4)
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False,)
        self.bn1 = nn.BatchNorm1d(planes)

        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
            self.relu3 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
            self.relu3 = nn.PReLU(num_parameters=planes*4)
        elif relu_type == "swish":
            self.relu1 = Swish()
            self.relu2 = Swish()
            self.relu3 = Swish()
        else:
            raise NotImplementedError
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, )
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes*4, kernel_size=1, stride=1, padding=0, bias=False, )
        self.bn3 = nn.BatchNorm1d(planes*4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """forward.
        :param x: torch.Tensor, input tensor with input size (B, C, T, H, W).
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu3(out)
        return out


class elasnet_prox(nn.Module):
    r"""Applies the elastic net proximal operator,
    NOTS: it will degenerate to ell1_prox if mu=0.0

    The elastic net proximal operator function is given as the following function
    \argmin_{x} \lambda ||x||_1 + \mu /2 ||x||_2^2 + 0.5 ||x - input||_2^2

    Args:
      lambd: the :math:`\lambda` value on the ell_1 penalty term. Default: 0.5
      mu:    the :math:`\mu` value on the ell_2 penalty term. Default: 0.0

    Shape:
      - Input: :math:`(N, *)` where `*` means, any number of additional
        dimensions
      - Output: :math:`(N, *)`, same shape as the input

    """

    def __init__(self, lambd=0.5, mu=0.0):
        super(elasnet_prox, self).__init__()
        self.lambd = lambd
        self.scaling_mu = 1.0 / (1.0 + mu)

    def forward(self, input):
        return F.softshrink(input * self.scaling_mu, self.lambd * self.scaling_mu)

    def extra_repr_(self):
        return '{} {}'.format(self.lambd, self.scaling_mu)


class ConvDictBlock(nn.Module):
    # c = argmin_c lmbd * ||c||_1  +  mu/2 * ||c||_2^2 + 1 / 2 * ||x - weight (@conv) c||_2^2
    def __init__(self, in_channels=1, out_channels=3, lmbd=0.1, tensor_size=16, n_steps=50):
        super(ConvDictBlock, self).__init__()
        kernel_size = 3
        step_size = 0.1
        self.size = tensor_size
        self.mu = 1.0
        self.lmbd = lmbd  # LAMBDA
        self.adapt_lambda = False
        self.n_dict = 1
        self.stride = 1
        self.kernel_size = (kernel_size, kernel_size)
        self.padding = 1
        self.padding_mode = "constant"
        assert self.padding_mode in ['constant', 'reflect', 'replicate', 'circular']
        self.groups = 1
        self.n_steps = n_steps
        self.conv_transpose_output_padding = 0 if self.stride == 1 else 1
        self.w_norm = True
        self.non_negative = True
        self.v_max = None
        self.v_max_error = 0.
        self.xsize = None
        self.zsize = None
        self.lmbd_ = None
        self.square_noise = True

        self.ADAPTIVELAMBDA = False

        # n_variables = 1 if share_weight else self.n_steps
        self.weight = nn.Parameter(torch.Tensor(out_channels, self.n_dict * in_channels, kernel_size, kernel_size))

        self.conv = nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight)

        # variables that are needed for ISTA/FISTA
        self.nonlinear = elasnet_prox(self.lmbd * step_size, self.mu * step_size)

        self.register_buffer('step_size', torch.tensor(step_size, dtype=torch.float))

    def fista_forward(self, x, lmbd):
        self.lmbd = lmbd
        if self.adapt_lambda:
            self.nonlinear = elasnet_prox(self.lmbd * 0.1, self.mu * 0.1)

        # self.c_error = []
        for i in range(self.n_steps):

            weight = self.weight
            step_size = self.step_size

            if i == 0:
                c_pre = 0.

                c = step_size * F.conv2d(x.repeat(1, self.n_dict, 1, 1), weight, bias=None, stride=self.stride,
                                         padding=self.padding)

                # print(f"c.device: {c.device}")
                # print(self.nonlinear)
                c = self.nonlinear(c)
            elif i == 1:
                c_pre = c
                # weight = self.normalize(weight)
                xp = F.conv_transpose2d(c, weight, bias=None, stride=self.stride, padding=self.padding,
                                        output_padding=self.conv_transpose_output_padding)
                r = x.repeat(1, self.n_dict, 1, 1) - xp

                if self.square_noise:
                    gra = F.conv2d(r, weight, bias=None, stride=self.stride, padding=self.padding)
                else:

                    w = r.view(r.size(0), -1)
                    normw = w.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12).expand_as(w).detach()
                    w = (w / normw).view(r.size())

                    gra = F.conv2d(w, weight, bias=None, stride=self.stride, padding=self.padding) * 0.5

                c = c + step_size * gra
                c = self.nonlinear(c)
                t = (math.sqrt(5.0) + 1.0) / 2.0
            else:
                t_pre = t
                t = (math.sqrt(1.0 + 4.0 * t_pre * t_pre) + 1) / 2.0
                a = (t_pre + t - 1.0) / t * c + (1.0 - t_pre) / t * c_pre
                c_pre = c
                xp = F.conv_transpose2d(c, weight, bias=None, stride=self.stride, padding=self.padding,
                                        output_padding=self.conv_transpose_output_padding)
                r = x.repeat(1, self.n_dict, 1, 1) - xp

                if self.square_noise:
                    gra = F.conv2d(r, weight, bias=None, stride=self.stride, padding=self.padding)
                else:

                    w = r.view(r.size(0), -1)
                    normw = w.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12).expand_as(w).detach()
                    w = (w / normw).view(r.size())

                    gra = F.conv2d(w, weight, bias=None, stride=self.stride, padding=self.padding) * 0.5

                c = a + step_size * gra
                c = self.nonlinear(c)

            if self.non_negative:
                c = F.relu(c)

        return c, weight

    def forward(self, x):
        if isinstance(x, tuple):
            x, out_1 = x
        else:
            out_1 = False
        x_shape = x.shape
        x = x.view(x.shape[0], x.shape[1], -1)
        b, t, c = x.shape
        x = x.view(b * t, int(c / (self.size * self.size)), self.size, self.size)

        if self.xsize is None:
            self.xsize = (x.size(-3), x.size(-2), x.size(-1))
        else:
            assert self.xsize[-3] == x.size(-3) and self.xsize[-2] == x.size(-2) and self.xsize[-1] == x.size(-1)

        if self.w_norm:
            self.normalize_weight()

        # self.c_error = []
        c, weight = self.fista_forward(x, self.lmbd)

        # Compute loss
        xp = F.conv_transpose2d(c, weight, bias=None, stride=self.stride, padding=self.padding,
                                output_padding=self.conv_transpose_output_padding)
        r = x.repeat(1, self.n_dict, 1, 1) - xp
        r_loss = torch.sum(torch.pow(r, 2)) / self.n_dict
        c_loss = self.lmbd * torch.sum(torch.abs(c)) + self.mu / 2. * torch.sum(torch.pow(c, 2))

        if self.zsize is None:
            self.zsize = (c.size(-3), c.size(-2), c.size(-1))
            # print(self.zsize)
        else:
            assert self.zsize[-3] == c.size(-3) and self.zsize[-2] == c.size(-2) and self.zsize[-1] == c.size(-1)

        if self.lmbd_ is None and self.ADAPTIVELAMBDA:
            self.lmbd_ = self.lmbd * self.xsize[-3] * self.xsize[-2] * self.xsize[-1] / (self.zsize[-3] * self.zsize[-2] * self.zsize[-1])
            self.lmbd = self.lmbd_
            print("new lmbd: ", self.lmbd)

        out = self.conv(c)
        out = out.view(x_shape)

        if isinstance(out_1, bool):
            return out, (r_loss, c_loss)
        else:
            return (out, out_1), (r_loss, c_loss)

    def normalize_weight(self):
        with torch.no_grad():
            w = self.weight.view(self.weight.size(0), -1)
            normw = w.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12).expand_as(w)
            w = (w / normw).view(self.weight.size())
            self.weight.data = w.data

