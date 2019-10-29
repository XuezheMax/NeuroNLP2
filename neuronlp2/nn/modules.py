__author__ = 'max'

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


class CharCNN(nn.Module):
    """
    CNN layers for characters
    """
    def __init__(self, in_channels, hidden_channels):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0.)

        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0.)

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
        char = F.elu(self.conv1(char), inplace=True)
        # [batch * sent_length, in_channels, char_length]
        char = self.conv2(char).max(dim=2)[0]
        # [batch, sent_length, in_channels]
        return char.view(char_size[0], char_size[1], -1)
