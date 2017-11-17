__author__ = 'max'

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from ..nn import TreeCRF, VarMaskedGRU, VarMaskedRNN, VarMaskedLSTM
from ..nn import Embedding
from ..nn import utils
from ..nn import Attention
from neuronlp2.tasks import parser


class BiRecurrentConvBiAffine(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None,
                 p_in=0.2, p_rnn=0.5, biaffine=False):
        super(BiRecurrentConvBiAffine, self).__init__()

        self.word_embedd = Embedding(num_words, word_dim, init_embedding=embedd_word)
        self.char_embedd = Embedding(num_chars, char_dim, init_embedding=embedd_char)
        self.pos_embedd = Embedding(num_pos, pos_dim, init_embedding=embedd_pos)
        self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1)
        self.dropout_in = nn.Dropout(p=p_in)
        self.dropout_rnn = nn.Dropout(p_rnn)
        self.num_labels = num_labels

        if rnn_mode == 'RNN':
            RNN = nn.RNN
        elif rnn_mode == 'LSTM':
            RNN = nn.LSTM
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        self.rnn = RNN(word_dim + num_filters + pos_dim, hidden_size, num_layers=num_layers,
                       batch_first=True, bidirectional=True, dropout=p_rnn)

        out_dim = hidden_size * 2
        self.dense_h = nn.Linear(out_dim, tag_space)
        self.dense_c = nn.Linear(out_dim, tag_space)
        self.attention = Attention(tag_space, tag_space, self.num_labels, biaffine=biaffine)
        self.logsoftmax = nn.LogSoftmax()

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
        input = F.elu(torch.cat([word, char, pos], dim=2))
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
        # [batch, num_label, length, length]
        mask = mask.contiguous()
        return self.attention(output[0], output[1], mask_d=mask, mask_e=mask), mask, length

    def loss(self, input_word, input_char, input_pos, heads, types,
             mask=None, length=None, hx=None, leading_symbolic=0, predict=False):
        # output shape [batch, num_labels, length, length]
        output, mask, length = self.forward(input_word, input_char, input_pos, mask=mask, length=length, hx=hx)
        batch, _, max_len, _ = output.size()

        if length is not None and heads.size(1) != mask.size(1):
            heads = heads[:, :max_len]
            types = types[:, :max_len]

        # mask invalid position to -inf for log_softmax
        minus_inf = -1e8
        if mask is not None:
            minus_mask = (1 - mask) * minus_inf
            output = output + minus_mask.view(batch, 1, max_len, 1) + minus_mask.view(batch, 1, 1, max_len)

        # TODO for Pytorch 2.0.4, need to set dim=1 for log_softmax or use softmax then take log
        # first convert output to [num_labels * max_len, batch, max_len] for log_softmax computation.
        # then convert back to [batch, num_labels, max_len, max_len]
        loss = self.logsoftmax(output.view(batch, self.num_labels * max_len, max_len).transpose(0, 1))
        loss = loss.transpose(0, 1).contiguous().view(batch, self.num_labels, max_len, max_len)

        # mask invalid position to 0 for sum loss
        if mask is not None:
            loss = loss * mask.view(batch, 1, max_len, 1) * mask.view(batch, 1, 1, max_len)
            # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
            num = mask.sum() - batch
        else:
            # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
            num = float(max_len - 1) * batch

        # first create index matrix [length, batch]
        index = torch.zeros(max_len, batch) + torch.arange(0, max_len).view(max_len, 1)
        index = index.type_as(loss.data).long()
        batch_index = torch.arange(0, batch).type_as(loss.data).long()
        # [length-1, batch]
        loss = loss[batch_index, types.data.t(), heads.data.t(), index][1:]

        if predict:
            # set diagonal elements to -inf
            output = output + Variable(torch.diag(output.data.new(max_len).fill_(-np.inf)))
            # compute naive predictions. First remove the first #leading_symbolic types.
            # then convert to [batch, num_types * length, length]
            # predition shape = [batch, length]
            output_reduce = output[:, leading_symbolic:].contiguous()
            output_reduce = output_reduce.view(batch, (self.num_labels - leading_symbolic) * max_len, max_len)
            _, preds = output_reduce.max(dim=1)
            types_pred = preds / max_len + leading_symbolic
            heads_preds = preds % max_len
            return -loss.sum() / num, (heads_preds, types_pred)
        else:
            return -loss.sum() / num

    def decode(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, leading_symbolic=0):
        output, _, _ = self.forward(input_word, input_char, input_pos, mask=mask, length=length, hx=hx)
        batch, _, max_len, _ = output.size()
        # set diagonal elements to -inf
        output = output + Variable(torch.diag(output.data.new(max_len).fill_(-np.inf)))
        # compute naive predictions. First remove the first #leading_symbolic types.
        # then convert to [batch, num_types * length, length]
        # predition shape = [batch, length]
        output_reduce = output[:, leading_symbolic:].contiguous()
        output_reduce = output_reduce.view(batch, (self.num_labels - leading_symbolic) * max_len, max_len)
        _, preds = output_reduce.max(dim=1)
        types_pred = preds / max_len + leading_symbolic
        heads_preds = preds % max_len
        return heads_preds.data.cpu().numpy(), types_pred.data.cpu().numpy()

    def decode_mst(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, leading_symbolic=0):
        '''
        Args:
            input_word: Tensor
                the word input tensor with shape = [batch, length]
            input_char: Tensor
                the character input tensor with shape = [batch, length, char_length]
            input_pos: Tensor
                the pos input tensor with shape = [batch, length]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            length: Tensor or None
                the length tensor with shape = [batch]
            hx: Tensor or None
                the initial states of RNN
            leading_symbolic: int
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: (Tensor, Tensor)
                predicted heads and types.

        '''
        # energy shape [batch, num_labels, length, length]
        energy, mask, length = self.forward(input_word, input_char, input_pos, mask=mask, length=length, hx=hx)
        batch, _, max_len, _ = energy.size()
        # compute lengths
        if length is None:
            if mask is None:
                length = [max_len for _ in range(batch)]
            else:
                length = mask.data.sum(dim=1).long().cpu().numpy()

        return parser.decode_MST(energy.data.cpu().numpy(), length, leading_symbolic)


class BiVarRecurrentConvBiAffine(BiRecurrentConvBiAffine):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None,
                 p_in=0.2, p_rnn=0.5, biaffine=False):
        super(BiVarRecurrentConvBiAffine, self).__init__(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                                                         num_filters, kernel_size, rnn_mode, hidden_size, num_layers,
                                                         num_labels, tag_space,
                                                         embedd_word=embedd_word, embedd_char=embedd_char,
                                                         embedd_pos=embedd_pos,
                                                         p_in=p_in, p_rnn=p_rnn, biaffine=biaffine)
        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_rnn = nn.Dropout2d(p_rnn)

        if rnn_mode == 'RNN':
            RNN = VarMaskedRNN
        elif rnn_mode == 'LSTM':
            RNN = VarMaskedLSTM
        elif rnn_mode == 'GRU':
            RNN = VarMaskedGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        self.rnn = RNN(word_dim + num_filters, hidden_size, num_layers=num_layers,
                       batch_first=True, bidirectional=True, dropout=p_rnn)

    def _get_rnn_output(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        # [batch, length, word_dim]
        word = self.word_embedd(input_word)

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
        input = F.elu(torch.cat([word, char], dim=2))
        # apply dropout
        # [batch, length, dim] --> [batch, dim, length] --> [batch, length, dim]
        input = self.dropout_in(input.transpose(1, 2)).transpose(1, 2)
        # output from rnn [batch, length, hidden_size]
        output, hn = self.rnn(input, mask, hx=hx)

        output = self.dropout_rnn(output.transpose(1, 2)).transpose(1, 2)

        # output size [batch, length, tag_space]
        output_h = F.elu(self.dense_h(output))
        output_c = F.elu(self.dense_c(output))

        return (output_h, output_c), hn, mask, length


class BiRecurrentConvTreeCRF(BiRecurrentConvBiAffine):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None,
                 p_in=0.2, p_rnn=0.5, biaffine=False):

        super(BiRecurrentConvTreeCRF, self).__init__(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                                                     num_filters, kernel_size, rnn_mode, hidden_size, num_layers,
                                                     num_labels, tag_space,
                                                     embedd_word=embedd_word, embedd_char=embedd_char,
                                                     embedd_pos=embedd_pos,
                                                     p_in=p_in, p_rnn=p_rnn, biaffine=biaffine)

        self.crf = TreeCRF(tag_space, num_labels, biaffine=biaffine)
        self.attention = None
        self.logsoftmax = None

    def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(input_word, input_char, input_pos,
                                                       mask=mask, length=length, hx=hx)
        # [batch, num_label, length, length]
        return self.crf(output[0], output[1], mask=mask), mask, length

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

        return self.crf.loss(output[0], output[1], heads, types, mask=mask, lengths=length).mean()

