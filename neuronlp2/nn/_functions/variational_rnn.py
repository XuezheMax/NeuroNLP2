__author__ = 'max'

import torch
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend
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


def VarLSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, noise=None):
    # if input.is_cuda:
    #     igates = F.linear(input, w_ih)
    #     hgates = F.linear(hidden[0], w_hh)
    #     state = fusedBackend.LSTMFused.apply
    #     return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

    hx, cx = hidden
    if noise is not None:
        hx = hx * noise
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy


def VarGRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, noise=None):
    # if input.is_cuda:
    #     gi = F.linear(input, w_ih)
    #     gh = F.linear(hidden, w_hh)
    #     state = fusedBackend.GRUFused.apply
    #     return state(gi, gh, hidden) if b_ih is None else state(gi, gh, hidden, b_ih, b_hh)

    hx = hidden if noise is None else hidden * noise
    gi = F.linear(input, w_ih, b_ih)
    gh = F.linear(hx, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = F.sigmoid(i_r + h_r)
    inputgate = F.sigmoid(i_i + h_i)
    newgate = F.tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy
