__author__ = 'max'

from torch.nn import MaxPool1d
from ..utils import _single
import torch.nn.functional as F

__all__ = [
    "MaxPoolTimeStep1d",
]


class MaxPoolTimeStep1d(MaxPool1d):
    """Applies a 1D max pooling over an input signal composed of several input
    planes.
    The input shape should be [batch_size, n-step, in_channels, input_length].
    Args:
        kernel_size: the size of the window to take a max over
            If equal to 0, the pooling region is across the whole length of the input.
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if True, will return the max indices along with the outputs.
                        Useful when Unpooling later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
    """

    def __init__(self, kernel_size=0, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(MaxPoolTimeStep1d, self).__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, input):
        # [batch, n-step, in_channels, output_length]
        input_size = input.size()

        batch_size, time_steps, in_channels, in_length = input_size

        # [batch * n-step, in_channels, input_length]
        new_input_size = (batch_size * time_steps, in_channels, in_length)

        # [batch * n-step, in_channels, output_length]
        if self.kernel_size:
            output = F.max_pool1d(input.view(new_input_size), self.kernel_size, self.stride,
                                  self.padding, self.dilation, self.ceil_mode,
                                  self.return_indices)
        else:
            output = F.max_pool1d(input.view(new_input_size), in_length, in_length,
                                  self.padding, self.dilation, self.ceil_mode,
                                  self.return_indices)

        # [batch * n-step, in_channels, output_length]
        output_size = output.size()
        # [batch, n-step, in_channels, output_length]
        output_size = (batch_size, time_steps, output_size[1], output_size[2])
        return output.view(output_size)
