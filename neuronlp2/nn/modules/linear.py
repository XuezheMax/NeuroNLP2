__author__ = 'max'

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class BiLinear(nn.Module):
    '''
    Bi-linear layer
    '''
    def __init__(self, left_features, right_features, out_features, bias=True):
        '''

        Args:
            left_features: size of left input
            right_features: size of right input
            out_features: size of output
            bias: If set to False, the layer will not learn an additive bias.
                Default: True
        '''
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.W_l = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.W_r = Parameter(torch.Tensor(self.out_features, self.left_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.W_l)
        nn.init.xavier_uniform(self.W_r)
        nn.init.constant(self.bias, 0.)
        nn.init.xavier_uniform(self.U)

    def forward(self, input_left, input_right):
        '''

        Args:
            input_left: Tensor
                the left input tensor with shape = [batch1, batch2, ..., left_features]
            input_right: Tensor
                the right input tensor with shape = [batch1, batch2, ..., right_features]

        Returns:

        '''

        left_size = input_left.size()
        right_size = input_right.size()
        assert left_size[:-1] == right_size[:-1], \
            "batch size of left and right inputs mis-match: (%s, %s)" % (left_size[:-1], right_size[:-1])
        batch = int(np.prod(left_size[:-1]))

        # convert left and right input to matrices [batch, left_features], [batch, right_features]
        input_left = input_left.view(batch, self.left_features)
        input_right = input_right.view(batch, self.right_features)

        # output [batch, out_features]
        output = F.bilinear(input_left, input_right, self.U, self.bias)
        output = output + F.linear(input_left, self.W_l, None) + F.linear(input_right, self.W_r, None)
        # convert back to [batch1, batch2, ..., out_features]
        return output.view(left_size[:-1] + (self.out_features, ))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'in1_features=' + str(self.left_features) \
               + ', in2_features=' + str(self.right_features) \
               + ', out_features=' + str(self.out_features) + ')'
