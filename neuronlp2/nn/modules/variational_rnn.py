__author__ = 'max'

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F


class VarRNNCellBase(nn.Module):
    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def reset_noise(self, batch_size):
        """
        Should be overriden by all subclasses.
        Args:
            batch_size: (int) batch size of input.
        """
        raise NotImplementedError


class VarRNNCell(VarRNNCellBase):
    r"""An Elman RNN cell with tanh non-linearity and variational dropout.

    .. math::

        h' = \tanh(w_{ih} * x + b_{ih}  +  w_{hh} * (h * \gamma) + b_{hh})

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'tanh'
        p: (float, optional): the drop probability. Default: 0.5

    Inputs: input, hidden
        - **input** (batch, input_size): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.

    Outputs: h'
        - **h'** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(input_size x hidden_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(hidden_size)`

    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh", p=0.5):
        super(VarRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(hidden_size))
            self.bias_hh = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.noise = None

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def reset_noise(self, batch_size):
        if self.training and self.p:
            noise = self.weight_hh.data.new(batch_size, self.hidden_size).bernoulli_(1 - self.p)
            self.noise = Variable(noise)

    def forward(self, input, hx):
        if self.nonlinearity == "tanh":
            func = self._backend.RNNTanhCell
        elif self.nonlinearity == "relu":
            func = self._backend.RNNReLUCell
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

        return func(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )