__author__ = 'max'

import torch


def decode_Viterbi(energies, masks, leading_symbolic):
    """
    decode best sequence of labels with Viterbi algorithm.
    :param energies: energies: torch 4D tensor
        energies of each edge. the shape is [batch_size, n_steps, n_steps, num_labels],
        where the summy root is at index 0.
    :param masks: torch 2D tensor
        masks in the shape [batch_size, n_steps].
    :param leading_symbolic: int
        number of symbolic dependency types leading in type alphabets (set it to 0 if you are not sure)
    :return: torch 2D tensor
        decoding in shape [batch, n_steps]
    """

    assert energies.dim() == 4
    assert masks.dim() == 2

    # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
    # For convenience, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
    energies_shuffled = energies.permute(1, 0, 2, 3)

    # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
    # also remove the first #symbolic rows and columns.
    # now the shape of energies_shuffled is [n_time_steps, b_batch, t, t] where t = num_labels - #symbolic - 1.
    energies_shuffled = energies_shuffled[:, :, leading_symbolic:-1, leading_symbolic:-1]

    length, batch_size, num_label, _ = energies_shuffled.size()

    pi = torch.zeros([length, batch_size, num_label])
    pointer = torch.IntTensor(length, batch_size, num_label).zero_()
    pi[0] = energies_shuffled[0, :, -1. :]
    pointer[0] = -1
    for t in range(1, length):
        pi_prev = pi[t - 1].view(batch_size, num_label, 1)
        pi[t], pointer[t] = torch.max(energies_shuffled[t] + pi_prev, dim=1)

    back_pointer = torch.IntTensor(length, batch_size).zero_()
    _, back_pointer[-1] = torch.max(pi[-1], dim=1)
    for t in reversed(range(length - 1)):
        pointer_last = pointer[t + 1]
        back_pointer[t] = pointer_last[torch.arange(0, batch_size), back_pointer[t + 1]]

    return back_pointer + leading_symbolic