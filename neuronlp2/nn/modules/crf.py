__author__ = 'max'

import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from neuronlp2.nlinalg import logsumexp, logdet


class ChainCRF(nn.Module):
    def __init__(self, input_size, num_labels, bias=True, **kwargs):
        '''

        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            bias: bool
                if apply bias parameter.
            **kwargs:
        '''
        super(ChainCRF, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels + 1
        self.pad_label_id = num_labels

        # transition weight tensor
        self.W_t = Parameter(torch.Tensor(input_size, self.num_labels, self.num_labels))
        # state weight tensor
        self.W_s = Parameter(torch.Tensor(input_size, self.num_labels))
        if bias:
            self.b = Parameter(torch.Tensor(self.num_labels, self.num_labels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.input_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        '''

        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]

        Returns: Tensor
            the energy tensor with shape = [batch, length, num_label, num_label]

        '''
        # compute out_s by tensor dot: [batch, length, input_size] * [input_size, num_label, num_label]
        # the output should be [batch, length, num_label,  num_label]
        batch, length, _ = input.size()
        out_t = torch.matmul(input, self.W_t.view(self.input_size, self.num_labels * self.num_labels))
        out_t = out_t.view(batch, length, self.num_labels, self.num_labels)

        # compute out_s by tensor dot [batch, length, input_size] * [input_size, num_label]
        # this out_s should be [batch, length, num_label]
        out_s = torch.matmul(input, self.W_s)

        output = out_t + out_s.view(batch, length, 1, self.num_labels)

        if self.b is not None:
            output += self.b

        if mask is not None:
            output = output * mask.view(mask.size() + (1, 1))

        return output

    def loss(self, input, target, mask=None):
        '''

        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            target: Tensor
                the tensor of target labels with shape [batch, length]
            mask:Tensor or None
                the mask tensor with shape = [batch, length]

        Returns: Tensor
                A 1D tensor for minus log likelihood loss
        '''
        batch, length, _ = input.size()
        energy = self.forward(input, mask=mask)
        # shape = [length, batch, num_label, num_label]
        energy_transpose = energy.transpose(0, 1)
        # shape = [length, batch]
        target_transpose = target.transpose(0, 1)
        # shape = [length, batch, 1]
        mask_transpose = None
        if mask is not None:
            mask_transpose = mask.view(mask.size() + (1, )).transpose(0, 1)


        # shape = [batch, num_label]
        partition = None

        if input.is_cuda:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long().cuda()
            prev_label = torch.cuda.LongTensor(batch).zero_() - 1
            tgt_energy = Variable(torch.zeros(batch)).cuda()
        else:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long()
            prev_label = torch.LongTensor(batch).zero_() - 1
            tgt_energy = Variable(torch.zeros(batch))

        for t in range(length):
            # shape = [batch, num_label, num_label]
            curr_energy = energy_transpose[t]
            if t == 0:
                partition = curr_energy[:, -1, :].contiguous()
            else:
                # shape = [batch, num_label]
                partition_new = logsumexp(curr_energy + partition.view(partition.size() + (1,)), dim=1)
                if mask_transpose is None:
                    partition = partition_new
                else:
                    mask_t = mask_transpose[t]
                    partition = mask_t * partition_new + (1 - mask_t) * partition

            tgt_energy += curr_energy[batch_index, prev_label, target_transpose[t].data]
            prev_label = target_transpose[t].data

        return logsumexp(partition, dim=1) - tgt_energy

    def decode(self, input, mask=None, leading_symbolic=0):
        """

        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            leading_symbolic: nt
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: Tensor
            decoding results in shape [batch, length]

        """

        energy = self.forward(input, mask=mask)

        # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
        # For convenience, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
        energy_transpose = energy.transpose(0, 1)

        # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
        # also remove the first #symbolic rows and columns.
        # now the shape of energies_shuffled is [n_time_steps, b_batch, t, t] where t = num_labels - #symbolic - 1.
        energy_transpose = energy_transpose[:, :, leading_symbolic:-1, leading_symbolic:-1]

        length, batch_size, num_label, _ = energy_transpose.size()

        pi = torch.zeros([length, batch_size, num_label])
        pointer = torch.IntTensor(length, batch_size, num_label).zero_()
        pi[0] = energy_transpose[0, :, -1. :]
        pointer[0] = -1
        for t in range(1, length):
            pi_prev = pi[t - 1].view(batch_size, num_label, 1)
            pi[t], pointer[t] = torch.max(energy_transpose[t] + pi_prev, dim=1)

        back_pointer = torch.IntTensor(length, batch_size).zero_()
        _, back_pointer[-1] = torch.max(pi[-1], dim=1)
        for t in reversed(range(length - 1)):
            pointer_last = pointer[t + 1]
            back_pointer[t] = pointer_last[torch.arange(0, batch_size), back_pointer[t + 1]]

        return back_pointer + leading_symbolic
