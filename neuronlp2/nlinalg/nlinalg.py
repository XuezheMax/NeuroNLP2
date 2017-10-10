__author__ = 'max'

import numpy
import torch

def logdet(x):
    """

    Args:
        x: 2D positive semidefinite matrix.

    Returns: log determinant of x

    """
    u_chol = torch.potrf(x)
    return torch.sum(torch.log(u_chol.diag())) * 2


def logsumexp(x, dim=None):
    """

    Args:
        x: A pytorch tensor (any dimension will do)
        dim: int or None, over which to perform the summation. `None`, the
             default, performs over all axes.

    Returns: The result of the log(sum(exp(...))) operation.

    """
    if dim is None:
        xmax = x.max()
        xmax_ = x.max()
        return xmax_ + numpy.log(torch.exp(x - xmax).sum())
    else:
        xmax, _ = x.max(dim, keepdim=True)
        xmax_, _ = x.max(dim)
        return xmax_ + torch.log(torch.exp(x - xmax).sum(dim))
