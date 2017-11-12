__author__ = 'max'

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nn import TreeCRF, VarMaskedGRU, VarMaskedRNN, VarMaskedLSTM
from ..nn import Embedding
from ..nn import utils


class BiRecurrentConvTreeCRF(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space,
                 embedd_word=None, embedd_pos=None, embedd_char=None,
                 p_in=0.2, p_rnn=0.5, biaffine=False):

        super(BiRecurrentConvTreeCRF, self).__init__()

        self.word_embedd = Embedding(num_words, word_dim, init_embedding=embedd_word)
        self.char_embedd = Embedding(num_chars, char_dim, init_embedding=embedd_char)
        self.pos_embedd = Embedding(num_pos, pos_dim, init_embedding=embedd_pos)
        self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1)
        self.dropout_in = nn.Dropout(p=p_in)
        self.dropout_rnn = nn.Dropout(p_rnn)

        if rnn_mode == 'RNN':
            RNN = nn.RNN
        elif rnn_mode == 'LSTM':
            RNN = nn.LSTM
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        self.rnn = RNN(word_dim + num_filters, hidden_size, num_layers=num_layers,
                       batch_first=True, bidirectional=True, dropout=p_rnn)

        out_dim = hidden_size * 2
        self.dense_h = nn.Linear(out_dim, tag_space)
        self.dense_c = nn.Linear(out_dim, tag_space)

        self.crf = TreeCRF(tag_space, num_labels, biaffine=biaffine)

    def _get_rnn_output(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        # hack length from mask
        # we do not hack mask from length for special reasons.
        # Thus, always provide mask if it is necessary.
        if length is None and mask is not None:
            length = mask.data.sum(dim=1).long()

        # [batch, length, word_dim]
        word = self.word_embedd(input_word)
        # [batch, length, pos_dim]
        pos = self.pos_embedd(input_pos)

        # [batch, length, char_length, char_dim]
        char = self.char_embedd(input_char)
        char_size = char.size()
        # first transform to [batch *length, char_length, char_dim]
        # then transpose to [batch * length, char_dim, char_length]
        char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
        # put into cnn [batch*length, char_filters, char_length]
        # then put into maxpooling [batch * length, char_filters]
        char, _ = self.conv1d(char).max(dim=2)
        # reshape to [batch, length, char_filters]
        char = char.view(char_size[0], char_size[1], -1)

        # concatenate word and char [batch, length, word_dim+char_filter]
        input = F.tanh(torch.cat([word, char, pos], dim=2))
        # apply dropout
        input = self.dropout_in(input)
        # prepare packed_sequence
        if length is not None:
            seq_input, hx, rev_order, mask = utils.prepare_rnn_seq(input, length, hx=hx, masks=mask, batch_first=True)
            seq_output, hn = self.rnn(seq_input, hx=hx)
            output, hn = utils.recover_rnn_seq(seq_output, rev_order, hx=hn, batch_first=True)
        else:
            # output from rnn [batch, length, hidden_size]
            output, hn = self.rnn(input, hx=hx)
        output = self.dropout_rnn(output)

        # output size [batch, length, tag_space]
        output_h = F.elu(self.dense_h(output))
        output_c = F.elu(self.dense_c(output))

        return (output_h, output_c), hn, mask, length

    def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(input_word, input_char, input_pos,
                                                       mask=mask, length=length, hx=hx)
        # [batch, length, num_label,  num_label]
        return self.crf(output[0], output[1], mask=mask), mask

    def loss(self, input_word, input_char, input_pos, heads, types, mask=None, length=None, hx=None):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(input_word, input_char, input_pos,
                                                       mask=mask, length=length, hx=hx)

        if length is not None:
            max_len = length.max()
            heads = heads[:, :max_len].contiguous()
            types = types[:, :max_len].contiguous()

        if mask is not None:
            mask = mask.contiguous()

        return self.crf.loss(output[0], output[1], heads, types, mask=mask, length=length).sum() / heads.size(0)

    def decode(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, leading_symbolic=0):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(input_word, input_char, input_pos,
                                                       mask=mask, length=length, hx=hx)
        if mask is not None:
            mask = mask.contiguous()
        return self.crf.decode(output[0], output[1], mask=mask, leading_symbolic=leading_symbolic)
