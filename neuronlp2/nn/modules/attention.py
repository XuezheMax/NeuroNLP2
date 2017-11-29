__author__ = 'max'

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class BiAAttention(nn.Module):
    '''
    Bi-Affine layer.
    '''

    def __init__(self, input_size_encoder, input_size_decoder, num_labels, biaffine=False, **kwargs):
        '''

        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        '''
        super(BiAAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.biaffine = biaffine

        self.W_d = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder))
        self.W_e = Parameter(torch.Tensor(self.num_labels, self.input_size_encoder))
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder, self.input_size_encoder))
        else:
            self.U = None
            self.register_parameter('biaffine_tensor', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.W_d)
        nn.init.xavier_uniform(self.W_e)
        nn.init.constant(self.b, 0.)
        if self.biaffine:
            nn.init.xavier_uniform(self.U)

    def forward(self, input_d, input_e, mask_d=None, mask_e=None):
        '''

        Args:
            input_d: Tensor
                the decoder input tensor with shape = [batch, length_decoder, input_size]
            input_e: Tensor
                the child input tensor with shape = [batch, length_encoder, input_size]
            mask_d: Tensor or None
                the mask tensor for decoder with shape = [batch, length_decoder]
            mask_e: Tensor or None
                the mask tensor for encoder with shape = [batch, length_encoder]
            lengths: Tensor or None
                the length tensor with shape = [batch]

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]

        '''
        assert input_d.size(0) == input_e.size(0), 'batch sizes of encoder and decoder are requires to be equal.'
        batch, length_decoder, _ = input_d.size()
        _, length_encoder, _ = input_e.size()

        # compute decoder part: [num_label, input_size_decoder] * [batch, input_size_decoder, length_decoder]
        # the output shape is [batch, num_label, length_decoder]
        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2)).unsqueeze(3)
        # compute decoder part: [num_label, input_size_encoder] * [batch, input_size_encoder, length_encoder]
        # the output shape is [batch, num_label, length_encoder]
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2)).unsqueeze(2)

        # output shape [batch, num_label, length_decoder, length_encoder]
        if self.biaffine:
            # compute bi-affine part
            # [batch, 1, length_decoder, input_size_decoder] * [num_labels, input_size_decoder, input_size_encoder]
            # output shape [batch, num_label, length_decoder, input_size_encoder]
            output = torch.matmul(input_d.unsqueeze(1), self.U)
            # [batch, num_label, length_decoder, input_size_encoder] * [batch, 1, input_size_encoder, length_encoder]
            # output shape [batch, num_label, length_decoder, length_encoder]
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))

            output = output + out_d + out_e + self.b
        else:
            output = out_d + out_d + self.b

        if mask_d is not None:
            output = output * mask_d.unsqueeze(1).unsqueeze(3) * mask_e.unsqueeze(1).unsqueeze(2)

        return output
