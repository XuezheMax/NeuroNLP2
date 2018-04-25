__author__ = 'max'

import torch


def assign_tensor(tensor, val):
    """
    copy val to tensor
    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        val: an n-dimensional torch.Tensor to fill the tensor with

    Returns:

    """
    with torch.no_grad():
        return tensor.copy_(val)
