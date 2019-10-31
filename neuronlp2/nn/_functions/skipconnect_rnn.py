__author__ = 'max'

import torch
from torch.nn import functional as F


def SkipConnectRNNReLUCell(input, hidden, hidden_skip, w_ih, w_hh, b_ih=None, b_hh=None, noise_in=None, noise_hidden=None, noise_skip=None):
    if noise_in is not None:
        input = input * noise_in

    hidden = torch.cat([hidden, hidden_skip], dim=1)
    if noise_hidden is not None:
        hidden = hidden * noise_hidden

    hy = F.relu(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))
    return hy


def SkipConnectRNNTanhCell(input, hidden, hidden_skip, w_ih, w_hh, b_ih=None, b_hh=None, noise_in=None, noise_hidden=None):
    if noise_in is not None:
        input = input * noise_in

    hidden = torch.cat([hidden, hidden_skip], dim=1)
    if noise_hidden is not None:
        hidden = hidden * noise_hidden

    hy = torch.tanh(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))
    return hy


def SkipConnectLSTMCell(input, hidden, hidden_skip, w_ih, w_hh, b_ih=None, b_hh=None, noise_in=None, noise_hidden=None):
    input = input.expand(4, *input.size()) if noise_in is None else input.unsqueeze(0) * noise_in

    hx, cx = hidden
    hx = torch.cat([hx, hidden_skip], dim=1)
    hx = hx.expand(4, *hx.size()) if noise_hidden is None else hx.unsqueeze(0) * noise_hidden

    gates = torch.baddbmm(b_ih.unsqueeze(1), input, w_ih) + torch.baddbmm(b_hh.unsqueeze(1), hx, w_hh)

    ingate, forgetgate, cellgate, outgate = gates

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def SkipConnectFastLSTMCell(input, hidden, hidden_skip, w_ih, w_hh, b_ih=None, b_hh=None, noise_in=None, noise_hidden=None):
    if noise_in is not None:
        input = input * noise_in

    hx, cx = hidden
    hx = torch.cat([hx, hidden_skip], dim=1)
    if noise_hidden is not None:
        hx = hx * noise_hidden

    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def SkipConnectGRUCell(input, hidden, hidden_skip, w_ih, w_hh, b_ih=None, b_hh=None, noise_in=None, noise_hidden=None):
    input = input.expand(3, *input.size()) if noise_in is None else input.unsqueeze(0) * noise_in
    hx = torch.cat([hidden, hidden_skip], dim=1)
    hx = hx.expand(3, *hx.size()) if noise_hidden is None else hx.unsqueeze(0) * noise_hidden

    gi = torch.baddbmm(b_ih.unsqueeze(1), input, w_ih)
    gh = torch.baddbmm(b_hh.unsqueeze(1), hx, w_hh)
    i_r, i_i, i_n = gi
    h_r, h_i, h_n = gh

    resetgate = torch.sigmoid(i_r + h_r)
    inputgate = torch.sigmoid(i_i + h_i)
    newgate = torch.tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy


def SkipConnectFastGRUCell(input, hidden, hidden_skip, w_ih, w_hh, b_ih=None, b_hh=None, noise_in=None, noise_hidden=None):
    if noise_in is not None:
        input = input * noise_in

    hx = torch.cat([hidden, hidden_skip], dim=1)
    if noise_hidden is not None:
        hx = hx * noise_hidden

    gi = F.linear(input, w_ih, b_ih)
    gh = F.linear(hx, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = torch.sigmoid(i_r + h_r)
    inputgate = torch.sigmoid(i_i + h_i)
    newgate = torch.tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy


def SkipConnectRecurrent(reverse=False):
    def forward(input, skip_connect, hidden, cell, mask):
        # hack to handle LSTM
        h0 = hidden[0] if isinstance(hidden, tuple) else hidden
        # [length + 1, batch, hidden_size]
        output = input.new_zeros(input.size(0) + 1, *h0.size()) + h0
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        # create batch index
        batch_index = torch.arange(0, h0.size(0)).type_as(skip_connect)
        for i in steps:
            if mask is None or mask[i].data.min() > 0.5:
                hidden_skip = output[skip_connect[i], batch_index]
                hidden = cell(input[i], hidden, hidden_skip)
            elif mask[i].data.max() > 0.5:
                hidden_skip = output[skip_connect[i], batch_index]
                hidden_next = cell(input[i], hidden, hidden_skip)
                # hack to handle LSTM
                if isinstance(hidden, tuple):
                    hx, cx = hidden
                    hp1, cp1 = hidden_next
                    hidden = (hx + (hp1 - hx) * mask[i], cx + (cp1 - cx) * mask[i])
                else:
                    hidden = hidden + (hidden_next - hidden) * mask[i]
            # hack to handle LSTM
            if reverse:
                output[i] = hidden[0] if isinstance(hidden, tuple) else hidden
            else:
                output[i + 1] = hidden[0] if isinstance(hidden, tuple) else hidden

        if reverse:
            # remove last position
            output = output[:-1]
        else:
            # remove position 0
            output = output[1:]

        return hidden, output

    return forward


def StackedRNN(inners, num_layers, lstm=False):
    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def reverse_skip_connection(skip_connect):
        # TODO reverse skip connection for bidirectional rnn.
        return skip_connect

    def forward(input, skip_connect, hidden, cells, mask):
        assert (len(cells) == total_layers)
        next_hidden = []

        skip_connect_forward = skip_connect
        skip_connec_backward = reverse_skip_connection(skip_connect) if num_directions == 2 else None

        if lstm:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j
                skip_connect = skip_connect_forward if j == 0 else skip_connec_backward
                hy, output = inner(input, skip_connect, hidden[l], cells[l], mask)
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, input.dim() - 1)

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


def AutogradSkipConnectRNN(num_layers=1, batch_first=False, bidirectional=False, lstm=False):
    rec_factory = SkipConnectRecurrent

    if bidirectional:
        layer = (rec_factory(), rec_factory(reverse=True))
    else:
        layer = (rec_factory(),)

    func = StackedRNN(layer,
                      num_layers,
                      lstm=lstm)

    def forward(input, skip_connect, cells, hidden, mask):
        if batch_first:
            input = input.transpose(0, 1)
            skip_connect = skip_connect.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        nexth, output = func(input, skip_connect, hidden, cells, mask)

        if batch_first:
            output = output.transpose(0, 1)

        return output, nexth

    return forward


def SkipConnectStep():
    def forward(input, hidden, hidden_skip, cell, mask):
        if mask is None or mask.data.min() > 0.5:
            hidden = cell(input, hidden, hidden_skip)
        elif mask.data.max() > 0.5:
            hidden_next = cell(input, hidden, hidden_skip)
            # hack to handle LSTM
            if isinstance(hidden, tuple):
                hx, cx = hidden
                hp1, cp1 = hidden_next
                hidden = (hx + (hp1 - hx) * mask, cx + (cp1 - cx) * mask)
            else:
                hidden = hidden + (hidden_next - hidden) * mask
        # hack to handle LSTM
        output = hidden[0] if isinstance(hidden, tuple) else hidden

        return hidden, output

    return forward


def StackedStep(layer, num_layers, lstm=False):
    def forward(input, hidden, hidden_skip, cells, mask):
        assert (len(cells) == num_layers)
        next_hidden = []

        if lstm:
            hidden = list(zip(*hidden))

        for l in range(num_layers):
            hy, output = layer(input, hidden[l], hidden_skip[l], cells[l], mask)
            next_hidden.append(hy)
            input = output

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(num_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(num_layers, *next_c[0].size())
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(num_layers, *next_hidden[0].size())

        return next_hidden, input

    return forward


def AutogradSkipConnectStep(num_layers=1, lstm=False):
    layer = SkipConnectStep()

    func = StackedStep(layer,
                       num_layers,
                       lstm=lstm)

    def forward(input, cells, hidden, hidden_skip, mask):
        nexth, output = func(input, hidden, hidden_skip, cells, mask)
        return output, nexth

    return forward
