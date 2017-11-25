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
    if input.is_cuda:
        igates = F.linear(input, w_ih)
        hgates = F.linear(hidden[0], w_hh) if noise is None else F.linear(hidden[0] * noise, w_hh)
        state = fusedBackend.LSTMFused()
        return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

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
    hx = hidden if noise is None else hidden * noise
    if input.is_cuda:
        gi = F.linear(input, w_ih)
        gh = F.linear(hx, w_hh)
        state = fusedBackend.GRUFused()
        return state(gi, gh, hidden) if b_ih is None else state(gi, gh, hidden, b_ih, b_hh)

    gi = F.linear(input, w_ih, b_ih)
    gh = F.linear(hx, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = F.sigmoid(i_r + h_r)
    inputgate = F.sigmoid(i_i + h_i)
    newgate = F.tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy


def VarMaskedRecurrent(reverse=False):
    def forward(input, hidden, cell, mask):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            if mask is None or mask[i].data.min() > 0.5:
                hidden = cell(input[i], hidden)
            elif mask[i].data.max() > 0.5:
                hidden_next = cell(input[i], hidden)
                # hack to handle LSTM
                if isinstance(hidden, tuple):
                    hx, cx = hidden
                    hp1, cp1 = hidden_next
                    hidden = (hx + (hp1 - hx) * mask[i], cx + (cp1 - cx) * mask[i])
                else:
                    hidden = hidden + (hidden_next - hidden) * mask[i]
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    return forward


def StackedRNN(inners, num_layers, lstm=False, dropout=0, train=True):

    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, cells, mask):
        assert(len(cells) == total_layers)
        next_hidden = []

        if lstm:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j
                print(num_layers, num_directions)
                print("i, j: %d %d" % (i, j))
                print(l)
                raw_input()
                hy, output = inner(input, hidden[l], cells[l], mask)
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, input.dim() - 1)

            if dropout != 0 and i < num_layers - 1:
                # [length, batch, dim] --> [dim, batch, length] --> [length, batch, dim]
                input = F.dropout2d(input.transpose(0, 2), p=dropout, training=train, inplace=False).transpose(0, 2)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(total_layers, *next_hidden[0].size())

        return next_hidden, input

    return forward


def AutogradVarMaskedRNN(num_layers=1, batch_first=False, dropout=0, train=True, bidirectional=False, lstm=False):

    rec_factory = VarMaskedRecurrent

    if bidirectional:
        layer = (rec_factory(), rec_factory(reverse=True))
    else:
        layer = (rec_factory(),)

    func = StackedRNN(layer,
                      num_layers,
                      lstm=lstm,
                      dropout=dropout,
                      train=train)

    def forward(input, cells, hidden, mask):
        if batch_first:
            input = input.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        nexth, output = func(input, hidden, cells, mask)

        if batch_first:
            output = output.transpose(0, 1)

        return output, nexth

    return forward
