__author__ = 'max'

import torch
from torch.nn import functional as F


def VarRNNReLUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, noise=None):
    if noise is not None:
        hidden = hidden * noise
    hy = F.relu(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))
    return hy


def VarRNNTanhCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, noise=None):
    if noise is not None:
        hidden = hidden * noise
    hy = F.tanh(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))
    return hy
