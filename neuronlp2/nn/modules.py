__author__ = 'max'

from overrides import overrides
from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class BiLinear(nn.Module):
    """
    Bi-linear layer
    """
    def __init__(self, left_features, right_features, out_features, bias=True):
        """

        Args:
            left_features: size of left input
            right_features: size of right input
            out_features: size of output
            bias: If set to False, the layer will not learn an additive bias.
                Default: True
        """
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.weight_left = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.weight_right = Parameter(torch.Tensor(self.out_features, self.right_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_left)
        nn.init.xavier_uniform_(self.weight_right)
        nn.init.constant_(self.bias, 0.)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left, input_right):
        """

        Args:
            input_left: Tensor
                the left input tensor with shape = [batch1, batch2, ..., left_features]
            input_right: Tensor
                the right input tensor with shape = [batch1, batch2, ..., right_features]

        Returns:

        """

        batch_size = input_left.size()[:-1]
        batch = int(np.prod(batch_size))

        # convert left and right input to matrices [batch, left_features], [batch, right_features]
        input_left = input_left.view(batch, self.left_features)
        input_right = input_right.view(batch, self.right_features)

        # output [batch, out_features]
        output = F.bilinear(input_left, input_right, self.U, self.bias)
        output = output + F.linear(input_left, self.weight_left, None) + F.linear(input_right, self.weight_right, None)
        # convert back to [batch1, batch2, ..., out_features]
        return output.view(batch_size + (self.out_features, ))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'left_features=' + str(self.left_features) \
               + ', right_features=' + str(self.right_features) \
               + ', out_features=' + str(self.out_features) + ')'


class BiAffine(nn.Module):
    '''
    Bi-Affine energy layer.
    '''

    def __init__(self, key_dim, query_dim):
        '''

        Args:
            key_dim: int
                the dimension of the key.
            query_dim: int
                the dimension of the query.

        '''
        super(BiAffine, self).__init__()
        self.key_dim = key_dim
        self.query_dim = query_dim

        self.q_weight = Parameter(torch.Tensor(self.query_dim))
        self.key_weight = Parameter(torch.Tensor(self.key_dim))
        self.b = Parameter(torch.Tensor(1))
        self.U = Parameter(torch.Tensor(self.query_dim, self.key_dim))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.query_dim)
        nn.init.uniform_(self.q_weight, -bound, bound)
        bound = 1 / math.sqrt(self.key_dim)
        nn.init.uniform_(self.key_weight, -bound, bound)
        nn.init.constant_(self.b, 0.)
        nn.init.xavier_uniform_(self.U)

    def forward(self, query, key, mask_query=None, mask_key=None):
        """

        Args:
            query: Tensor
                the decoder input tensor with shape = [batch, length_query, query_dim]
            key: Tensor
                the child input tensor with shape = [batch, length_key, key_dim]
            mask_query: Tensor or None
                the mask tensor for decoder with shape = [batch, length_query]
            mask_key: Tensor or None
                the mask tensor for encoder with shape = [batch, length_key]

        Returns: Tensor
            the energy tensor with shape = [batch, length_query, length_key]

        """
        # output shape [batch, length_query, length_key]
        # compute bi-affine part
        # [batch, length_query, query_dim] * [query_dim, key_dim]
        # output shape [batch, length_query, key_dim]
        output = torch.matmul(query, self.U)
        # [batch, length_query, key_dim] * [batch, key_dim, length_key]
        # output shape [batch, length_query, length_key]
        output = torch.matmul(output, key.transpose(1, 2))

        # compute query part: [query_dim] * [batch, query_dim, length_query]
        # the output shape is [batch, length_query, 1]
        out_q = torch.matmul(self.q_weight, query.transpose(1, 2)).unsqueeze(2)
        # compute decoder part: [key_dim] * [batch, key_dim, length_key]
        # the output shape is [batch, 1, length_key]
        out_k = torch.matmul(self.key_weight, key.transpose(1, 2)).unsqueeze(1)

        output = output + out_q + out_k + self.b

        if mask_query is not None:
            output = output * mask_query.unsqueeze(2)
        if mask_key is not None:
            output = output * mask_key.unsqueeze(1)
        return output

    @overrides
    def extra_repr(self):
        s = '{key_dim}, {query_dim}'
        return s.format(**self.__dict__)


class CharCNN(nn.Module):
    """
    CNN layers for characters
    """
    def __init__(self, num_layers, in_channels, out_channels, hidden_channels=None, activation='elu'):
        super(CharCNN, self).__init__()
        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            ACT = nn.ELU
        else:
            ACT = nn.Tanh
        layers = list()
        for i in range(num_layers - 1):
            layers.append(('conv{}'.format(i), nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)))
            layers.append(('act{}'.format(i), ACT()))
            in_channels = hidden_channels
        layers.append(('conv_top', nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)))
        layers.append(('act_top', ACT()))
        self.act = ACT
        self.net = nn.Sequential(OrderedDict(layers))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.)
            else:
                assert isinstance(layer, self.act)

    def forward(self, char):
        """

        Args:
            char: Tensor
                the input tensor of character [batch, sent_length, char_length, in_channels]

        Returns: Tensor
            output character encoding with shape [batch, sent_length, in_channels]

        """
        # [batch, sent_length, char_length, in_channels]
        char_size = char.size()
        # first transform to [batch * sent_length, char_length, in_channels]
        # then transpose to [batch * sent_length, in_channels, char_length]
        char = char.view(-1, char_size[2], char_size[3]).transpose(1, 2)
        # [batch * sent_length, out_channels, char_length]
        char = self.net(char).max(dim=2)[0]
        # [batch, sent_length, out_channels]
        return char.view(char_size[0], char_size[1], -1)
