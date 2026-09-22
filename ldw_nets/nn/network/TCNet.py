import logging

import torch
import torch.nn as nn
from ldw_nets.nn.unit import Swish
from collections import OrderedDict



__all__ = (
    "TCN",
    "MTCN",
    "DenseTCN",
)


def _average_batch(x, lengths, B):
    return torch.stack([torch.mean(x[index][:,0:i], 1) for index, i in enumerate(lengths)],0 )


class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size // 2:-self.chomp_size // 2].contiguous()
        else:
            return x[:, :, :-self.chomp_size].contiguous()


class ConvBatchChompRelu(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type, dwpw=False):
        super(ConvBatchChompRelu, self).__init__()
        self.dwpw = dwpw
        if dwpw:
            self.conv = nn.Sequential(
                # -- dw
                nn.Conv1d(n_inputs, n_inputs, kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=n_inputs, bias=False),
                nn.BatchNorm1d(n_inputs),
                Chomp1d(padding, True),
                nn.PReLU(num_parameters=n_inputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),
                # -- pw
                nn.Conv1d(n_inputs, n_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                  stride=stride, padding=padding, dilation=dilation)
            self.batchnorm = nn.BatchNorm1d(n_outputs)
            self.chomp = Chomp1d(padding, True)
            self.non_lin = nn.PReLU(
                num_parameters=n_outputs) if relu_type == 'prelu' else Swish() if relu_type == 'swish' else nn.ReLU()

    def forward(self, x):
        if self.dwpw:
            return self.conv(x)
        else:
            out = self.conv(x)
            out = self.batchnorm(out)
            out = self.chomp(out)
            return self.non_lin(out)


# --------- MULTI-BRANCH VERSION ---------------
class MultibranchTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation, padding, dropout=0.2,
                 relu_type='relu', dwpw=False):
        super(MultibranchTemporalBlock, self).__init__()

        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"

        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = ConvBatchChompRelu(n_inputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type,
                                      dwpw=dwpw)
            setattr(self, 'cbcr0_{}'.format(k_idx), cbcr)
        self.dropout0 = nn.Dropout(dropout)

        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = ConvBatchChompRelu(n_outputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type,
                                      dwpw=dwpw)
            setattr(self, 'cbcr1_{}'.format(k_idx), cbcr)
        self.dropout1 = nn.Dropout(dropout)

        # downsample?
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if (n_inputs // self.num_kernels) != n_outputs else None

        # final relu
        if relu_type == 'relu':
            self.relu_final = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu_final = nn.PReLU(num_parameters=n_outputs)
        elif relu_type == 'swish':
            self.relu_final = Swish()

    def forward(self, x):

        # first multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, 'cbcr0_{}'.format(k_idx))
            outputs.append(branch_convs(x))
        out0 = torch.cat(outputs, 1)
        out0 = self.dropout0(out0)

        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, 'cbcr1_{}'.format(k_idx))
            outputs.append(branch_convs(out0))
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1(out1)

        # downsample?
        res = x if self.downsample is None else self.downsample(x)

        return self.relu_final(out1 + res)


class MultibranchTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.2, relu_type='relu', dwpw=False):
        super(MultibranchTemporalConvNet, self).__init__()

        self.ksizes = kernel_size

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            padding = [(s - 1) * dilation_size for s in self.ksizes]
            layers.append(MultibranchTemporalBlock(in_channels, out_channels, self.ksizes,
                                                   stride=1, dilation=dilation_size, padding=padding, dropout=dropout,
                                                   relu_type=relu_type,
                                                   dwpw=dwpw))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    # --------------------------------


class MTCN(nn.Module):
    def __init__(self, input_size=512, num_classes=500, kernel_size=[3,5,7], dropout=0.2, relu_type="swish", dwpw=False):
        super(MTCN, self).__init__()

        # num_channels = 256 * 3 * 1 * 4
        num_channels = [256 * len([3, 5, 7]) * 1] * 4

        self.kernel_sizes = kernel_size
        self.num_kernels = len(self.kernel_sizes)

        self.mb_ms_tcn = MultibranchTemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout, relu_type=relu_type, dwpw=dwpw)
        # self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        xtrans = x.transpose(1, 2)
        out = self.mb_ms_tcn(xtrans)
        out = self.consensus_func(out, lengths, B)
        return out  # self.tcn_output(out)


# --------------- STANDARD VERSION (SINGLE BRANCH) ------------------------
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2,
                 symm_chomp=False, no_padding=False, relu_type='relu', dwpw=False):
        super(TemporalBlock, self).__init__()

        self.no_padding = no_padding
        if self.no_padding:
            downsample_chomp_size = 2 * padding - 4
            padding = 1  # hack-ish thing so that we can use 3 layers

        if dwpw:
            self.net = nn.Sequential(
                # -- first conv set within block
                # -- dw
                nn.Conv1d(n_inputs, n_inputs, kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=n_inputs, bias=False),
                nn.BatchNorm1d(n_inputs),
                Chomp1d(padding, True),
                nn.PReLU(
                    num_parameters=n_inputs) if relu_type == 'prelu' else Swish() if relu_type == 'swish' else nn.ReLU(
                    inplace=True),
                # -- pw
                nn.Conv1d(n_inputs, n_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(
                    num_parameters=n_outputs) if relu_type == 'prelu' else Swish() if relu_type == 'swish' else nn.ReLU(
                    inplace=True),
                nn.Dropout(dropout),
                # -- second conv set within block
                # -- dw
                nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=n_outputs, bias=False),
                nn.BatchNorm1d(n_outputs),
                Chomp1d(padding, True),
                nn.PReLU(
                    num_parameters=n_outputs) if relu_type == 'prelu' else Swish() if relu_type == 'swish' else nn.ReLU(
                    inplace=True),
                # -- pw
                nn.Conv1d(n_outputs, n_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(
                    num_parameters=n_outputs) if relu_type == 'prelu' else Swish() if relu_type == 'swish' else nn.ReLU(
                    inplace=True),
                nn.Dropout(dropout),
            )
        else:
            self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation)
            self.batchnorm1 = nn.BatchNorm1d(n_outputs)
            self.chomp1 = Chomp1d(padding, symm_chomp) if not self.no_padding else None
            if relu_type == 'relu':
                self.relu1 = nn.ReLU()
            elif relu_type == 'prelu':
                self.relu1 = nn.PReLU(num_parameters=n_outputs)
            elif relu_type == 'swish':
                self.relu1 = Swish()
            self.dropout1 = nn.Dropout(dropout)

            self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation)
            self.batchnorm2 = nn.BatchNorm1d(n_outputs)
            self.chomp2 = Chomp1d(padding, symm_chomp) if not self.no_padding else None
            if relu_type == 'relu':
                self.relu2 = nn.ReLU()
            elif relu_type == 'prelu':
                self.relu2 = nn.PReLU(num_parameters=n_outputs)
            elif relu_type == 'swish':
                self.relu2 = Swish()
            self.dropout2 = nn.Dropout(dropout)

            if self.no_padding:
                self.net = nn.Sequential(self.conv1, self.batchnorm1, self.relu1, self.dropout1,
                                         self.conv2, self.batchnorm2, self.relu2, self.dropout2)
            else:
                self.net = nn.Sequential(self.conv1, self.batchnorm1, self.chomp1, self.relu1, self.dropout1,
                                         self.conv2, self.batchnorm2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        if self.no_padding:
            self.downsample_chomp = Chomp1d(downsample_chomp_size, True)
        if relu_type == 'relu':
            self.relu = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu = nn.PReLU(num_parameters=n_outputs)
        elif relu_type == 'swish':
            self.relu = Swish()

    def forward(self, x):
        out = self.net(x)
        if self.no_padding:
            x = self.downsample_chomp(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2, relu_type='relu', dwpw=False):
        super(TemporalConvNet, self).__init__()
        self.ksize = tcn_options['kernel_size'][0] if isinstance(tcn_options['kernel_size'], list) else tcn_options[
            'kernel_size']
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, self.ksize, stride=1, dilation=dilation_size,
                                        padding=(self.ksize - 1) * dilation_size, dropout=dropout, symm_chomp=True,
                                        no_padding=False, relu_type=relu_type, dwpw=dwpw))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    """Implements Temporal Convolutional Network (TCN)
    __https://arxiv.org/pdf/1803.01271.pdf
    """

    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(TCN, self).__init__()
        self.tcn_trunk = TemporalConvNet(input_size, num_channels, dropout=dropout, tcn_options=tcn_options, relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

        self.has_aux_losses = False

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func( x, lengths, B )
        return self.tcn_output(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            Swish(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, T = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class TemporalConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type):
        super(TemporalConvLayer, self).__init__()
        self.net = nn.Sequential(
                nn.Conv1d( n_inputs, n_outputs, kernel_size,
                           stride=stride, padding=padding, dilation=dilation),
                nn.BatchNorm1d(n_outputs),
                Chomp1d(padding, True),
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else Swish() if relu_type == 'swish' else nn.ReLU(),)

    def forward(self, x):
        return self.net(x)


class _ConvBatchChompRelu(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size_set, stride, dilation, dropout, relu_type, se_module=False):
        super(_ConvBatchChompRelu, self).__init__()

        self.num_kernels = len( kernel_size_set )
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"

        for k_idx,k in enumerate( kernel_size_set ):
            if se_module:
                setattr( self, 'cbcr0_se_{}'.format(k_idx), SELayer( n_inputs, reduction=16))
            cbcr = TemporalConvLayer( n_inputs, self.n_outputs_branch, k, stride, dilation, (k-1)*dilation, relu_type)
            setattr( self,'cbcr0_{}'.format(k_idx), cbcr )
        self.dropout0 = nn.Dropout(dropout)
        for k_idx,k in enumerate( kernel_size_set ):
            cbcr = TemporalConvLayer( n_outputs, self.n_outputs_branch, k, stride, dilation, (k-1)*dilation, relu_type)
            setattr( self,'cbcr1_{}'.format(k_idx), cbcr )
        self.dropout1 = nn.Dropout(dropout)

        self.se_module = se_module
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        # final relu
        if relu_type == 'relu':
            self.relu_final = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu_final = nn.PReLU(num_parameters=n_outputs)
        elif relu_type == 'swish':
            self.relu_final = Swish()

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        x = torch.cat(inputs, 1)
        outputs = []
        for k_idx in range( self.num_kernels ):
            if self.se_module:
                branch_se = getattr(self,'cbcr0_se_{}'.format(k_idx))
            branch_convs = getattr(self,'cbcr0_{}'.format(k_idx))
            if self.se_module:
                outputs.append( branch_convs(branch_se(x)))
            else:
                outputs.append( branch_convs(x) )
        out0 = torch.cat(outputs, 1)
        out0 = self.dropout0( out0 )
        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range( self.num_kernels ):
            branch_convs = getattr(self,'cbcr1_{}'.format(k_idx))
            outputs.append( branch_convs(out0) )
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1( out1 )
        # downsample?
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_final(out1 + res)

    def forward(self, input):
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input
        bottleneck_output = self.bn_function(prev_features)
        return bottleneck_output


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, relu_type):
        super(_Transition, self).__init__()
        self.add_module('conv', nn.Conv1d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('norm', nn.BatchNorm1d(num_output_features))
        if relu_type == 'relu':
            self.add_module('relu', nn.ReLU())
        elif relu_type == 'prelu':
            self.add_module('prelu', nn.PReLU(num_parameters=num_output_features))
        elif relu_type == 'swish':
            self.add_module('swish', Swish())


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__( self, num_layers, num_input_features, growth_rate,
                  kernel_size_set, dilation_size_set,
                  dropout, relu_type, squeeze_excitation,
                  ):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            dilation_size = dilation_size_set[i%len(dilation_size_set)]
            layer = _ConvBatchChompRelu(
                n_inputs=num_input_features + i * growth_rate,
                n_outputs=growth_rate,
                kernel_size_set=kernel_size_set,
                stride=1,
                dilation=dilation_size,
                dropout=dropout,
                relu_type=relu_type,
                se_module=squeeze_excitation,
                )

            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseTemporalConvNet(nn.Module):
    def __init__(self, block_config, growth_rate_set, input_size, reduced_size,
                 kernel_size_set, dilation_size_set,
                 dropout=0.2, relu_type='prelu',
                 squeeze_excitation=False,
                 ):
        super(DenseTemporalConvNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([]))

        trans = _Transition(num_input_features=input_size,
                            num_output_features=reduced_size,
                            relu_type='prelu')
        self.features.add_module('transition%d' % (0), trans)
        num_features = reduced_size

        for i, num_layers in enumerate(block_config):

            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate_set[i],
                kernel_size_set=kernel_size_set,
                dilation_size_set=dilation_size_set,
                dropout=dropout,
                relu_type=relu_type,
                squeeze_excitation=squeeze_excitation,
                )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate_set[i]

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=reduced_size,
                                    relu_type=relu_type)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = reduced_size

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))


    def forward(self, x):
        features = self.features(x)
        return features


class DenseTCN(nn.Module):
    def __init__( self, block_config=[3,3,3,3], growth_rate_set=[384,384,384,384], input_size=512, reduced_size=512, kernel_size_set=[3,5,7], dilation_size_set=[1,2,5],
                  dropout=0.2, relu_type="swish",
                  squeeze_excitation=True,
        ):
        super(DenseTCN, self).__init__()

        # num_features = reduced_size + block_config[-1]*growth_rate_set[-1]
        self.tcn_trunk = DenseTemporalConvNet(block_config, growth_rate_set, input_size, reduced_size,
                                          kernel_size_set, dilation_size_set,
                                          dropout=dropout, relu_type=relu_type,
                                          squeeze_excitation=squeeze_excitation,
                                          )
        # self.tcn_output = nn.Linear(num_features, num_classes)
        self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func(x, lengths, B)  # torch.Size([15, 1664])
        return x  # self.tcn_output(x)

