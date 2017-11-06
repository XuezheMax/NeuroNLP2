__author__ = 'max'

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nn import ChainCRF, VarMaskedGRU, VarMaskedRNN, VarMaskedLSTM
from ..nn import Embedding
from ..nn import utils


class BiRecurrentConv(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5):
        super(BiRecurrentConv, self).__init__()

        self.word_embedd = Embedding(num_words, word_dim, init_embedding=embedd_word)
        self.char_embedd = Embedding(num_chars, char_dim, init_embedding=embedd_char)
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

        self.dense = nn.Linear(hidden_size * 2, num_labels)

        self.logsoftmax = nn.LogSoftmax()
        self.nll_loss = nn.NLLLoss(size_average=False)

    def forward(self, input_word, input_char, mask=None, length=None, hx=None):
        # hack length from mask
        # we do not hack mask from length for special reasons.
        # Thus, always provide mask if it is necessary.
        if length is None and mask is not None:
            length = mask.data.sum(dim=1).long()

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
        input = F.tanh(torch.cat([word, char], dim=2))
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
        # [batch, length, num_labels]
        return self.dense(self.dropout_rnn(output)), hn, mask

    def loss(self, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0):
        # hack length from mask
        # we do not hack mask from length for special reasons.
        # Thus, always provide mask if it is necessary.
        if length is None and mask is not None:
            length = mask.data.sum(dim=1).long()
        # [batch, length, num_labels]
        output, _, mask = self.forward(input_word, input_char, mask=mask, length=length, hx=hx)
        # preds = [batch, length]
        _, preds = torch.max(output[:, :, leading_symbolic:], dim=2)
        preds += leading_symbolic

        output_size = output.size()
        # [batch * length, num_labels]
        output_size = (output_size[0] * output_size[1], output_size[2])
        output = output.view(output_size)

        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]
            target = target.contiguous()

        if mask is not None:
            return self.nll_loss(self.logsoftmax(output) * mask.contiguous().view(output_size[0], 1),
                                 target.view(-1)) / mask.sum(), \
                   (torch.eq(preds, target).type_as(mask) * mask).sum(), preds
        else:
            num = output_size[0] * output_size[1]
            return self.nll_loss(self.logsoftmax(output), target.view(-1)) / num, \
                   (torch.eq(preds, target).type_as(output)).sum(), preds


class BiRecurrentConvCRF(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5):
        super(BiRecurrentConvCRF, self).__init__()

        self.word_embedd = Embedding(num_words, word_dim, init_embedding=embedd_word)
        self.char_embedd = Embedding(num_chars, char_dim, init_embedding=embedd_char)
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

        self.crf = ChainCRF(hidden_size * 2, num_labels)

    def _get_rnn_output(self, input_word, input_char, mask=None, length=None, hx=None):
        # hack length from mask
        # we do not hack mask from length for special reasons.
        # Thus, always provide mask if it is necessary.
        if length is None and mask is not None:
            length = mask.data.sum(dim=1).long()

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
        input = F.tanh(torch.cat([word, char], dim=2))
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
        return self.dropout_rnn(output), hn, mask, length

    def forward(self, input_word, input_char, mask=None, length=None, hx=None):
        # output from rnn [batch, length, hidden_size]
        output, _, mask, length = self._get_rnn_output(input_word, input_char, mask=mask, length=length, hx=hx)
        # [batch, length, num_label,  num_label]
        return self.crf(output, mask=mask), mask

    def loss(self, input_word, input_char, target, mask=None, length=None, hx=None):
        # output from rnn [batch, length, hidden_size]
        output, _, mask, length = self._get_rnn_output(input_word, input_char, mask=mask, length=length, hx=hx)

        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]
            target = target.contiguous()

        # [batch, length, num_label,  num_label]
        return self.crf.loss(output, target, mask=mask.contiguous()).sum() / target.size(0)

    def decode(self, input_word, input_char, target=None, mask=None, length=None, hx=None, leading_symbolic=0):
        # output from rnn [batch, length, hidden_size]
        output, _, mask, length = self._get_rnn_output(input_word, input_char, mask=mask, length=length, hx=hx)
        if target is None:
            return self.crf.decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]
            target = target.contiguous()

        preds = self.crf.decode(output, mask=mask.contiguous(), leading_symbolic=leading_symbolic)
        if mask is None:
            return preds, torch.eq(preds, target.data).float().sum()
        else:
            return preds, (torch.eq(preds, target.data).float() * mask).sum()

class BiVarRecurrentConv(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5):
        super(BiVarRecurrentConv, self).__init__()

        self.word_embedd = Embedding(num_words, word_dim, init_embedding=embedd_word)
        self.char_embedd = Embedding(num_chars, char_dim, init_embedding=embedd_char)
        self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1)
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

        self.dense = nn.Linear(hidden_size * 2, num_labels)

        self.logsoftmax = nn.LogSoftmax()
        self.nll_loss = nn.NLLLoss(size_average=False)

    def forward(self, input_word, input_char, mask=None, hx=None):
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
        input = F.tanh(torch.cat([word, char], dim=2))
        # apply dropout
        # [batch, length, dim] --> [batch, dim, length] --> [batch, length, dim]
        input = self.dropout_in(input.transpose(1, 2)).transpose(1, 2)
        # output from rnn [batch, length, hidden_size]
        output, _ = self.rnn(input, mask, hx=hx)
        # [batch, length, num_labels]
        return self.dense(self.dropout_rnn(output.transpose(1, 2)).transpose(1, 2))

    def loss(self, input_word, input_char, target, mask=None, hx=None, leading_symbolic=0):
        # [batch, length, num_labels]
        output = self.forward(input_word, input_char, mask=mask, hx=hx)
        # preds = [batch, length]
        _, preds = torch.max(output[:, :, leading_symbolic:], dim=2)
        preds += leading_symbolic

        output_size = output.size()
        # [batch * length, num_labels]
        output_size = (output_size[0] * output_size[1], output_size[2])
        output = output.view(output_size)
        if mask is not None:
            return self.nll_loss(self.logsoftmax(output) * mask.view(output_size[0], 1),
                                 target.view(-1)) / mask.sum(), \
               (torch.eq(preds, target).type_as(mask) * mask).sum(), preds
        else:
            num = output_size[0] * output_size[1]
            return self.nll_loss(self.logsoftmax(output), target.view(-1)) / num, \
                   (torch.eq(preds, target).type_as(output)).sum(), preds