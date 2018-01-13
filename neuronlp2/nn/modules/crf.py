__author__ = 'max'

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from neuronlp2.nlinalg import logsumexp, logdet
from neuronlp2.tasks import parser
from .attention import BiAAttention


class ChainCRF(nn.Module):
    def __init__(self, input_size, num_labels, bigram=True, **kwargs):
        '''

        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            bigram: bool
                if apply bi-gram parameter.
            **kwargs:
        '''
        super(ChainCRF, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels + 1
        self.pad_label_id = num_labels
        self.bigram = bigram


        # state weight tensor
        self.state_nn = nn.Linear(input_size, self.num_labels)
        if bigram:
            # transition weight tensor
            self.trans_nn = nn.Linear(input_size, self.num_labels * self.num_labels)
            self.register_parameter('trans_matrix', None)
        else:
            self.trans_nn = None
            self.trans_matrix = Parameter(torch.Tensor(self.num_labels, self.num_labels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant(self.state_nn.bias, 0.)
        if self.bigram:
            nn.init.xavier_uniform(self.trans_nn.weight)
            nn.init.constant(self.trans_nn.bias, 0.)
        else:
            nn.init.normal(self.trans_matrix)
        # if not self.bigram:
        #     nn.init.normal(self.trans_matrix)

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
        batch, length, _ = input.size()

        # compute out_s by tensor dot [batch, length, input_size] * [input_size, num_label]
        # thus out_s should be [batch, length, num_label] --> [batch, length, num_label, 1]
        out_s = self.state_nn(input).unsqueeze(2)

        if self.bigram:
            # compute out_s by tensor dot: [batch, length, input_size] * [input_size, num_label * num_label]
            # the output should be [batch, length, num_label,  num_label]
            out_t = self.trans_nn(input).view(batch, length, self.num_labels, self.num_labels)
            output = out_t + out_s
        else:
            # [batch, length, num_label, num_label]
            output = self.trans_matrix + out_s

        if mask is not None:
            output = output * mask.unsqueeze(2).unsqueeze(3)

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
            mask_transpose = mask.unsqueeze(2).transpose(0, 1)


        # shape = [batch, num_label]
        partition = None

        if input.is_cuda:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long().cuda()
            prev_label = torch.cuda.LongTensor(batch).fill_(self.num_labels - 1)
            tgt_energy = Variable(torch.zeros(batch)).cuda()
        else:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long()
            prev_label = torch.LongTensor(batch).fill_(self.num_labels - 1)
            tgt_energy = Variable(torch.zeros(batch))

        for t in range(length):
            # shape = [batch, num_label, num_label]
            curr_energy = energy_transpose[t]
            if t == 0:
                partition = curr_energy[:, -1, :]
            else:
                # shape = [batch, num_label]
                partition_new = logsumexp(curr_energy + partition.unsqueeze(2), dim=1)
                if mask_transpose is None:
                    partition = partition_new
                else:
                    mask_t = mask_transpose[t]
                    partition = partition + (partition_new - partition) * mask_t
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

        energy = self.forward(input, mask=mask).data

        # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
        # For convenience, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
        energy_transpose = energy.transpose(0, 1)

        # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
        # also remove the first #symbolic rows and columns.
        # now the shape of energies_shuffled is [n_time_steps, b_batch, t, t] where t = num_labels - #symbolic - 1.
        energy_transpose = energy_transpose[:, :, leading_symbolic:-1, leading_symbolic:-1]

        length, batch_size, num_label, _ = energy_transpose.size()

        if input.is_cuda:
            batch_index = torch.arange(0, batch_size).long().cuda()
            pi = torch.zeros([length, batch_size, num_label, 1]).cuda()
            pointer = torch.cuda.LongTensor(length, batch_size, num_label).zero_()
            back_pointer = torch.cuda.LongTensor(length, batch_size).zero_()
        else:
            batch_index = torch.arange(0, batch_size).long()
            pi = torch.zeros([length, batch_size, num_label, 1])
            pointer = torch.LongTensor(length, batch_size, num_label).zero_()
            back_pointer = torch.LongTensor(length, batch_size).zero_()

        pi[0] = energy[:, 0, -1, leading_symbolic:-1]
        pointer[0] = -1
        for t in range(1, length):
            pi_prev = pi[t - 1]
            pi[t], pointer[t] = torch.max(energy_transpose[t] + pi_prev, dim=1)

        _, back_pointer[-1] = torch.max(pi[-1], dim=1)
        for t in reversed(range(length - 1)):
            pointer_last = pointer[t + 1]
            back_pointer[t] = pointer_last[batch_index, back_pointer[t + 1]]

        return back_pointer.transpose(0, 1) + leading_symbolic


class TreeCRF(nn.Module):
    '''
    Tree CRF layer.
    '''
    def __init__(self, input_size, num_labels, biaffine=True, **kwargs):
        '''

        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        '''
        super(TreeCRF, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels
        self.attention = BiAAttention(input_size, input_size, num_labels, biaffine=biaffine)

    def forward(self, input_h, input_c, mask=None):
        '''

        Args:
            input_h: Tensor
                the head input tensor with shape = [batch, length, input_size]
            input_c: Tensor
                the child input tensor with shape = [batch, length, input_size]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            lengths: Tensor or None
                the length tensor with shape = [batch]

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]

        '''
        batch, length, _ = input_h.size()
        # [batch, num_labels, length, length]
        output = self.attention(input_h, input_c, mask_d=mask, mask_e=mask)
        # set diagonal elements to -inf
        output = output + Variable(torch.diag(output.data.new(length).fill_(-np.inf)))
        return output

    def loss(self, input_h, input_c, heads, types, mask=None, lengths=None):
        '''

        Args:
            input_h: Tensor
                the head input tensor with shape = [batch, length, input_size]
            input_c: Tensor
                the child input tensor with shape = [batch, length, input_size]
            target: Tensor
                the tensor of target labels with shape [batch, length]
            mask:Tensor or None
                the mask tensor with shape = [batch, length]
            lengths: tensor or list of int
                the length of each input shape = [batch]

        Returns: Tensor
                A 1D tensor for minus log likelihood loss
        '''
        batch, length, _ = input_h.size()
        energy = self.forward(input_h, input_c, mask=mask)
        # [batch, num_labels, length, length]
        A = torch.exp(energy)
        # mask out invalid positions
        if mask is not None:
            A = A * mask.unsqueeze(1).unsqueeze(3) * mask.unsqueeze(1).unsqueeze(2)

        # sum along the label axis [batch, length, length]
        A = A.sum(dim=1)
        # get D [batch, 1, length]
        D = A.sum(dim=1, keepdim=True)

        # make sure L is positive-defined
        rtol = 1e-4
        atol = 1e-6
        D += D * rtol + atol

        # [batch, length, length]
        D = Variable(A.data.new(A.size()).zero_()) + D
        # zeros out all elements except diagonal.
        D = D * Variable(torch.eye(length)).type_as(D)

        # compute laplacian matrix
        # [batch, length, length]
        L = D - A

        # compute lengths
        if lengths is None:
            if mask is None:
                lengths = [length for _ in range(batch)]
            else:
                lengths = mask.data.sum(dim=1).long()

        # compute partition Z(x) [batch]
        z = Variable(energy.data.new(batch))
        for b in range(batch):
            Lx = L[b, 1:lengths[b], 1:lengths[b]]
            # print(torch.log(torch.eig(Lx.data)[0]))
            z[b] = logdet(Lx)

        # first create index matrix [length, batch]
        # index = torch.zeros(length, batch) + torch.arange(0, length).view(length, 1)
        index = torch.arange(0, length).view(length, 1).expand(length, batch)
        index = index.type_as(energy.data).long()
        batch_index = torch.arange(0, batch).type_as(energy.data).long()
        # compute target energy [length-1, batch]
        tgt_energy = energy[batch_index, types.data.t(), heads.data.t(), index][1:]
        # sum over dim=0 shape = [batch]
        tgt_energy = tgt_energy.sum(dim=0)

        return z - tgt_energy

