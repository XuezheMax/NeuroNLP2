import collections
from itertools import repeat
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def prepare_rnn_seq(rnn_input, lengths, masks=None, batch_first=False):
    def check_decreasing(lengths):
        lens, order = torch.sort(lengths, dim=0, descending=True)
        if torch.ne(lens, lengths).sum() == 0:
            return None
        else:
            _, rev_order = torch.sort(order)
            return lens, Variable(order), Variable(rev_order)

    check_res = check_decreasing(lengths)

    if check_res is None:
        lens = lengths
    else:
        lens, order, rev_order = check_res
        batch_dim = 0 if batch_first else 1
        rnn_input = rnn_input.index_select(batch_dim, order)
