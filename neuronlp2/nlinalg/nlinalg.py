__author__ = 'max'

import numpy
import torch
from torch.autograd.function import Function


class Potrf(Function):
    """
    cf. Iain Murray (2016); arXiv 1602.07527
    """

    @staticmethod
    def forward(ctx, a, upper=True):
        ctx.upper = upper
        fact = torch.potrf(a, upper)
        ctx.save_for_backward(fact)
        return fact

    @staticmethod
    def phi(A):
        """
        Return lower triangle of A and halve the diagonal.
        """
        B = A.tril()

        B = B - 0.5 * torch.diag(torch.diag(B))

        return B

    @staticmethod
    def backward(ctx, grad_output):
        L, = ctx.saved_variables

        if ctx.upper:
            L = L.t()
            grad_output = grad_output.t()

        # make sure not to double-count variation, since
        # only half of output matrix is unique
        Lbar = grad_output.tril()

        P = Potrf.phi(torch.mm(L.t(), Lbar))
        S = torch.gesv(P + P.t(), L.t())[0]
        S = torch.gesv(S.t(), L.t())[0]
        S = Potrf.phi(S)

        return S, None


def potrf(x, upper=True):
    return Potrf.apply(x, upper)


def logdet(x):
    """

    Args:
        x: 2D positive semidefinite matrix.

    Returns: log determinant of x

    """
    # TODO for pytorch 2.0.4, use inside potrf for variable.
    # u_chol = x.potrf()
    u_chol = potrf(x)
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
