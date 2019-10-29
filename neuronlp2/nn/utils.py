import collections
from itertools import repeat
import torch.nn as nn


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


def freeze_embedding(embedding):
    assert isinstance(embedding, nn.Embedding), "input should be an Embedding module."
    embedding.weight.detach_()
