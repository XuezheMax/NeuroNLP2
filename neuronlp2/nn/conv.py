__author__ = 'max'

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import Conv1d, Module
import torch.nn.functional as F
from .utils import _single, _pair, _triple

__all__ = [
    "ConvTimeStep1d",
]


class _ConvTimeStepNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvTimeStepNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class ConvTimeStep1d(_ConvTimeStepNd):
    """
    CNN with time step at axis=1.
    The input shape should be [batch_size, n-step, in_channels, input_length].
    The output shape should be [batch_size, n-step, out_channels,

    Parameters
    ----------
    in_channels : int
        The number of channels of input.
        The shape of the input tensor should be
        ``(batch_size, n-step, in_channels, input_length)``.

    out_channels : int
        The number of learnable convolutional filters this layer has.

    kernel_size : int or iterable of int
        An integer or a 1-element tuple specifying the size of the kernels.

    stride : int or iterable of int
        An integer or a 1-element tuple specifying the stride of the
        convolution operation which controls the stride for the cross-correlation.

    padding : int or tuple
        Zero-padding added to both sides of the input.

    dilation : int or tuple
        Spacing between kernel elements (see `torch.nn.Conv1d` for details).

    groups : int
        Number of blocked connections from input channels to output channels.

    bias : bool
        If True, adds a learnable bias to the output.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(ConvTimeStep1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias)

    def forward(self, input):
        # [batch, n-step, in_channels, input_length]
        input_size = input.size()

        batch_size = input_size[0]
        time_steps = input_size[1]

        # [batch * n-step, in_channels, input_length]
        new_input_size = (batch_size * time_steps, input_size[2], input_size[3])

        # [batch * n-step, out_channels, output_length]
        output = F.conv1d(input.view(new_input_size), self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)

        # [batch * n-step, out_channels, output_length]
        output_size = output.size()
        # [batch, n-step, out_channels, output_length]
        output_size = (batch_size, time_steps, output_size[1], output_size[2])
        return output.view(output_size)
