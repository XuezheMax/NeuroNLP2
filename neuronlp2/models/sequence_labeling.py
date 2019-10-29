__author__ = 'max'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from neuronlp2.nn import ChainCRF, VarMaskedGRU, VarMaskedRNN, VarMaskedLSTM, CharCNN


class BiRecurrentConv(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, rnn_mode, hidden_size, num_layers, num_labels,
                 embedd_word=None, embedd_char=None, p_in=0.33, p_out=0.5, p_rnn=(0.5, 0.5)):
        super(BiRecurrentConv, self).__init__()

        self.word_embed = nn.Embedding(num_words, word_dim, _weight=embedd_word, padding_idx=1)
        self.char_embed = nn.Embedding(num_chars, char_dim, _weight=embedd_char, padding_idx=1)
        self.char_cnn = CharCNN(char_dim, char_dim * 4)
        # dropout word
        self.dropout_in = nn.Dropout2d(p=p_in)
        # standard dropout
        self.dropout_rnn_in = nn.Dropout(p=p_rnn[0])
        self.dropout_out = nn.Dropout(p_out)

        if rnn_mode == 'RNN':
            RNN = nn.RNN
        elif rnn_mode == 'LSTM':
            RNN = nn.LSTM
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        self.rnn = RNN(word_dim + char_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn[1])

        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.readout = nn.Linear(hidden_size, num_labels)
        self.criterion = nn.CrossEntropyLoss(reduction='none')


        self.reset_parameters(embedd_word, embedd_char)

    def reset_parameters(self, embedd_word, embedd_char):
        if embedd_word is None:
            nn.init.uniform_(self.word_embed.weight, -0.1, 0.1)
        if embedd_char is None:
            nn.init.uniform_(self.char_embed.weight, -0.1, 0.1)
        with torch.no_grad():
            self.word_embed.weight[self.word_embed.padding_idx].fill_(0)
            self.char_embed.weight[self.char_embed.padding_idx].fill_(0)

        nn.init.uniform_(self.readout.weight, -0.1, 0.1)
        nn.init.constant_(self.readout.bias, 0.)

    def _get_rnn_output(self, input_word, input_char, mask=None):
        # [batch, length, word_dim]
        word = self.word_embed(input_word)

        # [batch, length, char_length, char_dim]
        char = self.char_cnn(self.char_embed(input_char))

        # apply dropout word on input
        word = self.dropout_in(word)
        char = self.dropout_in(char)

        # concatenate word and char [batch, length, word_dim+char_filter]
        enc = torch.cat([word, char], dim=2)
        # apply dropout rnn input
        enc = self.dropout_rnn_in(enc)

        # output from rnn [batch, length, hidden_size * 2]
        if mask is not None:
            # prepare packed_sequence
            length = mask.sum(dim=1).long()
            packed_enc = pack_padded_sequence(enc, length, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.rnn(packed_enc)
            output, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            output, _ = self.rnn(enc)

        output = self.dropout_out(output)
        # [batch, length, hidden_size]
        output = self.dropout_out(F.elu(self.fc(output)))
        return output

    def forward(self, input_word, input_char, mask=None):
        # output from rnn [batch, length, tag_space]
        output = self._get_rnn_output(input_word, input_char, mask=mask)
        return output

    def loss(self, input_word, input_char, target, mask=None, leading_symbolic=0):
        # [batch, length, hidden_size]
        output = self(input_word, input_char, mask=mask)
        # [batch, length, num_labels] -> [batch, num_labels, length]
        logits = self.readout(output).transpose(1, 2)

        # [batch, length]
        loss = self.criterion(logits, target)
        if mask is not None:
            loss = loss * mask
        # [batch]
        loss = loss.sum(dim=1)
        return loss

    def decode(self, input_word, input_char, mask=None, leading_symbolic=0):
        output = self(input_word, input_char, mask=mask)
        # [batch, length, num_labels] -> [batch, num_labels, length]
        logits = self.readout(output).transpose(1, 2)
        # [batch, length]
        _, preds = torch.max(logits[:, leading_symbolic:], dim=1)
        preds += leading_symbolic
        if mask is not None:
            preds = preds * mask.long()
        return preds


class BiVarRecurrentConv(BiRecurrentConv):
    def __init__(self, word_dim, num_words, char_dim, num_chars, rnn_mode, hidden_size, num_layers, num_labels,
                 embedd_word=None, embedd_char=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33)):
        super(BiVarRecurrentConv, self).__init__(word_dim, num_words, char_dim, num_chars, rnn_mode, hidden_size, num_layers, num_labels,
                                                 embedd_word=embedd_word, embedd_char=embedd_char, p_in=p_in, p_out=p_out, p_rnn=p_rnn)

        self.dropout_rnn_in = None
        self.dropout_out = nn.Dropout2d(p_out)

        if rnn_mode == 'RNN':
            RNN = VarMaskedRNN
        elif rnn_mode == 'LSTM':
            RNN = VarMaskedLSTM
        elif rnn_mode == 'GRU':
            RNN = VarMaskedGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        self.rnn = RNN(word_dim + char_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn, initializer=self.initializer)

    def _get_rnn_output(self, input_word, input_char, mask=None, length=None, hx=None):
        # [batch, length, word_dim]
        word = self.word_embed(input_word)

        # [batch, length, char_length, char_dim]
        char = self.char_cnn(self.char_embed(input_char))

        # apply dropout word on input
        word = self.dropout_in(word)
        char = self.dropout_in(char)

        # concatenate word and char [batch, length, word_dim+char_filter]
        enc = torch.cat([word, char], dim=2)
        # output from rnn [batch, length, 2 * hidden_size]
        output, _ = self.rnn(enc, mask)

        # [batch, length, hidden_size]
        output = F.elu(self.fc(output))
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, tag_space]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)
        return output


class BiRecurrentConvCRF(BiRecurrentConv):
    def __init__(self, word_dim, num_words, char_dim, num_chars, rnn_mode, hidden_size, num_layers, num_labels,
                 embedd_word=None, embedd_char=None, p_in=0.33, p_out=0.5, p_rnn=(0.5, 0.5), bigram=False):
        super(BiRecurrentConvCRF, self).__init__(word_dim, num_words, char_dim, num_chars, rnn_mode, hidden_size, num_layers, num_labels,
                                                 embedd_word=embedd_word, embedd_char=embedd_char, p_in=p_in, p_out=p_out, p_rnn=p_rnn)

        self.crf = ChainCRF(hidden_size, num_labels, bigram=bigram)
        self.readout = None
        self.criterion = None

    def forward(self, input_word, input_char, mask=None):
        # output from rnn [batch, length, hidden_size]
        output = self._get_rnn_output(input_word, input_char, mask=mask)
        # [batch, length, num_label, num_label]
        return self.crf(output, mask=mask)

    def loss(self, input_word, input_char, target, mask=None, leading_symbolic=0):
        # output from rnn [batch, length, hidden_size]
        output = self._get_rnn_output(input_word, input_char, mask=mask)
        # [batch]
        return self.crf.loss(output, target, mask=mask)

    def decode(self, input_word, input_char, mask=None, leading_symbolic=0):
        # output from rnn [batch, length, hidden_size]
        output = self._get_rnn_output(input_word, input_char, mask=mask)
        # [batch, length]
        preds = self.crf.decode(output, mask=mask, leading_symbolic=leading_symbolic)
        if mask is not None:
            preds = preds * mask.long()
        return preds


class BiVarRecurrentConvCRF(BiVarRecurrentConv):
    def __init__(self, word_dim, num_words, char_dim, num_chars, rnn_mode, hidden_size, num_layers, num_labels,
                 embedd_word=None, embedd_char=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33), bigram=False):
        super(BiVarRecurrentConvCRF, self).__init__(word_dim, num_words, char_dim, num_chars, rnn_mode, hidden_size, num_layers, num_labels,
                                                    embedd_word=embedd_word, embedd_char=embedd_char, p_in=p_in, p_out=p_out, p_rnn=p_rnn)

        self.crf = ChainCRF(hidden_size, num_labels, bigram=bigram)
        self.readout = None
        self.criterion = None

    def forward(self, input_word, input_char, mask=None):
        # output from rnn [batch, length, hidden_size]
        output = self._get_rnn_output(input_word, input_char, mask=mask)
        # [batch, length, num_label,  num_label]
        return self.crf(output, mask=mask)

    def loss(self, input_word, input_char, target, mask=None, leading_symbolic=0):
        # output from rnn [batch, length, hidden_size]
        output = self._get_rnn_output(input_word, input_char, mask=mask)
        # [batch]
        return self.crf.loss(output, target, mask=mask)

    def decode(self, input_word, input_char, mask=None, leading_symbolic=0):
        # output from rnn [batch, length, hidden_size]
        output = self._get_rnn_output(input_word, input_char, mask=mask)
        preds = self.crf.decode(output, mask=mask, leading_symbolic=leading_symbolic)
        if mask is not None:
            preds = preds * mask.long()
        return preds
