__author__ = 'max'

from torch.nn import MaxPool1d
from .utils import _single

__all__ = [
    "MaxPoolTimeStep1d",
]


class MaxPoolTimeStep1d(MaxPool1d):
    """
    Pool with time step at axis=1.
    The input shape should be [batch_size, n-step, in_channels, input_length].
    Parameters
    ----------
    kernel_size : integer or iterable
        The length of the pooling region. If an iterable, it should have a single element.

    stride : integer, iterable or ``None``
        stride: the stride of the window. Default value is :attr:`kernel_size`.

    padding : integer or iterable
        The number of elements to be added to the input on each side.
        Must be less than stride.

    dilation : int or tuple
        Spacing between kernel elements (see `torch.nn.Conv1d` for details).

    return_indices : bool
        if True, will return the max indices along with the outputs. Useful when Unpooling later.

    ceil_mode : bool
        when True, will use `ceil` instead of `floor` to compute the output shape.
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(MaxPoolTimeStep1d, self).__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, input):
        # [batch, n-step, in_channels, output_length]
        input_size = input.size()

        batch_size = input_size[0]
        time_steps = input_size[1]

        # [batch * n-step, in_channels, input_length]
        new_input_size = (batch_size * time_steps, input_size[2], input_size[3])

        # [batch * n-step, in_channels, output_length]
        output = super(MaxPoolTimeStep1d, self).forward(input.view(new_input_size))

        # [batch * n-step, in_channels, output_length]
        output_size = output.size()
        # [batch, n-step, in_channels, output_length]
        output_size = (batch_size, time_steps, output_size[1], output_size[2])
        return output.view(output_size)
