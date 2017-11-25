__author__ = 'max'

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from ..nn import TreeCRF, VarMaskedGRU, VarMaskedRNN, VarMaskedLSTM
from ..nn import Embedding
from ..nn import utils
from ..nn import Attention, BiLinear
from neuronlp2.tasks import parser


class BiRecurrentConvBiAffine(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, arc_space, type_space,
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
        self.arc_h = nn.Linear(out_dim, arc_space)
        self.arc_c = nn.Linear(out_dim, arc_space)
        self.attention = Attention(arc_space, arc_space, 1, biaffine=biaffine)

        self.type_h = nn.Linear(out_dim, type_space)
        self.type_c = nn.Linear(out_dim, type_space)
        self.bilinear = BiLinear(type_space, type_space, self.num_labels)
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
        char = torch.tanh(char).view(char_size[0], char_size[1], -1)

        # concatenate word and char [batch, length, word_dim+char_filter]
        input = torch.cat([word, char, pos], dim=2)
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

        # output size [batch, length, arc_space]
        arc_h = F.elu(self.arc_h(output))
        arc_c = F.elu(self.arc_c(output))

        # output size [batch, length, type_space]
        type_h = F.elu(self.type_h(output))
        type_c = F.elu(self.type_c(output))

        return (arc_h, arc_c), (type_h, type_c), hn, mask, length

    def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        # output from rnn [batch, length, tag_space]
        arc, type, _, mask, length = self._get_rnn_output(input_word, input_char, input_pos,
                                                          mask=mask, length=length, hx=hx)
        # [batch, length, length]
        out_arc = self.attention(arc[0], arc[1], mask_d=mask, mask_e=mask).squeeze(dim=1)
        return out_arc, type, mask, length

    def loss(self, input_word, input_char, input_pos, heads, types,
             mask=None, length=None, hx=None):
        # out_arc shape [batch, length, length]
        out_arc, out_type, mask, length = self.forward(input_word, input_char, input_pos,
                                                       mask=mask, length=length, hx=hx)
        batch, max_len, _ = out_arc.size()

        if length is not None and heads.size(1) != mask.size(1):
            heads = heads[:, :max_len]
            types = types[:, :max_len]

        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type

        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(out_arc.data).long()
        # get vector for heads [batch, length, type_space],
        type_h = type_h[batch_index, heads.data.t()].transpose(0, 1).contiguous()
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(type_h, type_c)

        # mask invalid position to -inf for log_softmax
        if mask is not None:
            minus_inf = -1e8
            minus_mask = (1 - mask) * minus_inf
            out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # TODO for Pytorch 2.0.4, need to set dim=1 for log_softmax or use softmax then take log
        # first convert out_arc to [max_len, batch, max_len] for log_softmax computation.
        # then convert back to [batch, max_len, max_len]
        loss_arc = self.logsoftmax(out_arc.transpose(0, 1)).transpose(0, 1)
        # convert out_type to [num_labels, length, batch] for log_softmax computation.
        # then convert back to [batch, length, num_labels]
        loss_type = self.logsoftmax(out_type.transpose(0, 2)).transpose(0, 2)

        # mask invalid position to 0 for sum loss
        if mask is not None:
            loss_arc = loss_arc * mask.unsqueeze(2) * mask.unsqueeze(1)
            loss_type = loss_type * mask.unsqueeze(2)
            # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
            num = mask.sum() - batch
        else:
            # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
            num = float(max_len - 1) * batch

        # first create index matrix [length, batch]
        child_index = torch.arange(0, max_len).view(max_len, 1).expand(max_len, batch)
        child_index = child_index.type_as(out_arc.data).long()
        # [length-1, batch]
        loss_arc = loss_arc[batch_index, heads.data.t(), child_index][1:]
        loss_type = loss_type[batch_index, child_index, types.data.t()][1:]

        return -loss_arc.sum() / num, -loss_type.sum() / num

    def _decode_types(self, out_type, heads, leading_symbolic):
        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type
        batch, max_len, _ = type_h.size()
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(type_h.data).long()
        # get vector for heads [batch, length, type_space],
        type_h = type_h[batch_index, heads.t()].transpose(0, 1).contiguous()
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(type_h, type_c)
        # remove the first #leading_symbolic types.
        out_type = out_type[:, :, leading_symbolic:]
        # compute the prediction of types [batch, length]
        _, types = out_type.max(dim=2)
        return types + leading_symbolic

    def decode(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, leading_symbolic=0):
        # out_arc shape [batch, length, length]
        out_arc, out_type, mask, length = self.forward(input_word, input_char, input_pos,
                                                       mask=mask, length=length, hx=hx)
        out_arc = out_arc.data
        batch, max_len, _ = out_arc.size()
        # set diagonal elements to -inf
        out_arc = out_arc + torch.diag(out_arc.new(max_len).fill_(-np.inf))
        # set invalid positions to -inf
        if mask is not None:
            # minus_mask = (1 - mask.data).byte().view(batch, max_len, 1)
            minus_mask = (1 - mask.data).byte().unsqueeze(2)
            out_arc.masked_fill_(minus_mask, -np.inf)

        # compute naive predictions.
        # predition shape = [batch, length]
        _, heads = out_arc.max(dim=1)

        types = self._decode_types(out_type, heads, leading_symbolic)

        return heads.cpu().numpy(), types.data.cpu().numpy()

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
        # out_arc shape [batch, length, length]
        out_arc, out_type, mask, length = self.forward(input_word, input_char, input_pos,
                                                       mask=mask, length=length, hx=hx)

        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type
        batch, max_len, type_space = type_h.size()

        # compute lengths
        if length is None:
            if mask is None:
                length = [max_len for _ in range(batch)]
            else:
                length = mask.data.sum(dim=1).long().cpu().numpy()

        type_h = type_h.unsqueeze(2).expand(batch, max_len, max_len, type_space).contiguous()
        type_c = type_c.unsqueeze(1).expand(batch, max_len, max_len, type_space).contiguous()
        # compute output for type [batch, length, length, num_labels]
        out_type = self.bilinear(type_h, type_c)

        # mask invalid position to -inf for log_softmax
        if mask is not None:
            minus_inf = -1e8
            minus_mask = (1 - mask) * minus_inf
            out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # TODO for Pytorch 2.0.4, need to set dim=1 for log_softmax or use softmax then take log
        # first convert out_arc to [max_len, batch, max_len] for log_softmax computation.
        # then convert back to [batch, max_len, max_len]
        loss_arc = self.logsoftmax(out_arc.transpose(0, 1)).transpose(0, 1)
        # convert out_type to [batch, num_labels, length_c, length_h] for log_softmax computation.
        # then switch (2, 3) to [batch, num_labels, length_h, length_c]
        loss_type = self.logsoftmax(out_type.transpose(1, 3)).transpose(2, 3)
        # [batch, num_labels, length, length]
        energy = loss_arc.unsqueeze(1) + loss_type
        # mask invalid position to 0 for sum loss
        if mask is not None:
            # mask = [batch, 1, length]
            mask = mask.unsqueeze(1)
            energy = loss_arc * mask.unsqueeze(3) * mask.unsqueeze(2)

        return parser.decode_MST(energy.data.cpu().numpy(), length,
                                 leading_symbolic=leading_symbolic, labeled=True)

        # heads_numpy, _ = parser.decode_MST(out_arc.data.cpu().numpy(), length,
        #                                    leading_symbolic=leading_symbolic, labeled=False)
        # heads = torch.from_numpy(heads_numpy).type_as(input_word.data).long()
        #
        # types = self._decode_types(out_type, heads, leading_symbolic)

        # return heads_numpy, types.data.cpu().numpy()


class BiVarRecurrentConvBiAffine(BiRecurrentConvBiAffine):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, arc_space, type_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None,
                 p_in=0.2, p_rnn=0.5, biaffine=False):
        super(BiVarRecurrentConvBiAffine, self).__init__(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                                                         num_filters, kernel_size, rnn_mode, hidden_size, num_layers,
                                                         num_labels, arc_space, type_space,
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

        self.rnn = RNN(word_dim + num_filters + pos_dim, hidden_size, num_layers=num_layers,
                       batch_first=True, bidirectional=True, dropout=p_rnn)

    def _get_rnn_output(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
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
        char = torch.tanh(char).view(char_size[0], char_size[1], -1)

        # concatenate word and char [batch, length, word_dim+char_filter]
        input = torch.cat([word, char, pos], dim=2)
        # apply dropout
        # [batch, length, dim] --> [batch, dim, length] --> [batch, length, dim]
        input = self.dropout_in(input.transpose(1, 2)).transpose(1, 2)
        # output from rnn [batch, length, hidden_size]
        output, hn = self.rnn(input, mask, hx=hx)

        output = self.dropout_rnn(output.transpose(1, 2)).transpose(1, 2)

        # output size [batch, length, arc_space]
        arc_h = F.elu(self.arc_h(output))
        arc_c = F.elu(self.arc_c(output))

        # output size [batch, length, type_space]
        type_h = F.elu(self.type_h(output))
        type_c = F.elu(self.type_c(output))

        return (arc_h, arc_c), (type_h, type_c), hn, mask, length


class StackPtrNet(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, arc_space, type_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None,
                 p_in=0.2, p_rnn=0.5, biaffine=False):

        super(StackPtrNet, self).__init__()
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

        self.encoder = RNN(word_dim + num_filters + pos_dim, hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=True, dropout=p_rnn)

        self.decoder = RNN(word_dim + num_filters + pos_dim, hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=False, dropout=p_rnn)

        self.hx_dense = nn.Linear(2 * hidden_size, hidden_size)
        self.arc_h = nn.Linear(hidden_size, arc_space)  # arc dense for decoder
        self.arc_c = nn.Linear(hidden_size * 2, arc_space)  # arc dense for encoder
        self.attention = Attention(arc_space, arc_space, 1, biaffine=biaffine)

        self.type_h = nn.Linear(hidden_size, type_space)  # type dense for decoder
        self.type_c = nn.Linear(hidden_size * 2, type_space)  # type dense for encoder
        self.bilinear = BiLinear(type_space, type_space, self.num_labels)
        self.logsoftmax = nn.LogSoftmax()

    def _get_encoder_output(self, input_word, input_char, input_pos, mask_e=None, length_e=None, hx=None):
        # hack length from mask
        # we do not hack mask from length for special reasons.
        # Thus, always provide mask if it is necessary.
        if length_e is None and mask_e is not None:
            length_e = mask_e.data.sum(dim=1).long()

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
        char = torch.tanh(char).view(char_size[0], char_size[1], -1)

        # concatenate word and char [batch, length, word_dim+char_filter]
        src_encoding = torch.cat([word, char, pos], dim=2)
        # apply dropout
        input = self.dropout_in(src_encoding)
        # prepare packed_sequence
        if length_e is not None:
            seq_input, hx, rev_order, mask_e = utils.prepare_rnn_seq(input, length_e, hx=hx, masks=mask_e,
                                                                     batch_first=True)
            seq_output, hn = self.encoder(seq_input, hx=hx)
            output, hn = utils.recover_rnn_seq(seq_output, rev_order, hx=hn, batch_first=True)
        else:
            # output from rnn [batch, length, hidden_size]
            output, hn = self.encoder(input, hx=hx)
        output = self.dropout_rnn(output)

        # output size [batch, length, arc_space]
        arc_c = F.elu(self.arc_c(output))

        # output size [batch, length, type_space]
        type_c = F.elu(self.type_c(output))

        return src_encoding, arc_c, type_c, hn, mask_e, length_e

    def _get_decoder_output(self, src_encoding, heads_stack, hx, mask_d=None, length_d=None):
        # hack length from mask
        # we do not hack mask from length for special reasons.
        # Thus, always provide mask if it is necessary.
        if length_d is None and mask_d is not None:
            length_d = mask_d.data.sum(dim=1).long()

        batch, _, _ = src_encoding.size()
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(src_encoding.data).long()
        # get vector for heads [batch, length_decoder, input_dim],
        src_encoding = src_encoding[batch_index, heads_stack.data.t()].transpose(0, 1)
        # apply dropout
        input = self.dropout_in(src_encoding)
        # prepare packed_sequence
        if length_d is not None:
            seq_input, hx, rev_order, mask_d = utils.prepare_rnn_seq(input, length_d, hx=hx, masks=mask_d,
                                                                     batch_first=True)
            seq_output, hn = self.decoder(seq_input, hx=hx)
            # output from rnn [batch, length_decoder, hidden_size]
            output, hn = utils.recover_rnn_seq(seq_output, rev_order, hx=hn, batch_first=True)
        else:
            # output from rnn [batch, length_decoder, hidden_size]
            output, hn = self.decoder(input, hx=hx)
        output = self.dropout_rnn(output)

        # output size [batch, length_decoder, arc_space]
        arc_h = F.elu(self.arc_h(output))

        # output size [batch, length_decoder, type_space]
        type_h = F.elu(self.type_h(output))

        return arc_h, type_h, hn, mask_d, length_d

    def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        raise RuntimeError('Stack Pointer Network does not implement forward')

    def _transform_decoder_init_state(self, hn):
        if isinstance(hn, tuple):
            hn, cn = hn
            # hn [2 * num_layers, batch, hidden_size]
            num_dir, batch, hidden_size = cn.size()
            # first convert cn t0 [batch, 2 * num_layers, hidden_size]
            cn = cn.transpose(0, 1).contiguous()
            # then view to [batch, num_layers, 2 * hidden_size] --> [num_layer, batch, 2 * num_layers]
            cn = cn.view(batch, num_dir / 2, 2 * hidden_size).transpose(0, 1)
            # take hx_dense to [num_layers, batch, hidden_size]
            cn = self.hx_dense(cn)
            # hn is tanh(cn)
            hn = F.tanh(cn)
            hn = (hn, cn)
        else:
            # hn [2 * num_layers, batch, hidden_size]
            num_dir, batch, hidden_size = hn.size()
            # first convert hn t0 [batch, 2 * num_layers, hidden_size]
            hn = hn.transpose(0, 1).contiguous()
            # then view to [batch, num_layers, 2 * hidden_size] --> [num_layer, batch, 2 * num_layers]
            hn = hn.view(batch, num_dir / 2, 2 * hidden_size).transpose(0, 1)
            # take hx_dense to [num_layers, batch, hidden_size]
            hn = F.tanh(self.hx_dense(hn))
        return hn

    def loss(self, input_word, input_char, input_pos, stacked_heads, children, stacked_types,
             mask_e=None, length_e=None, mask_d=None, length_d=None, hx=None):

        # output from encoder [batch, length_encoder, tag_space]
        src_encoding, arc_c, type_c, hn, mask_e, _ = self._get_encoder_output(input_word, input_char, input_pos,
                                                                              mask_e=mask_e, length_e=length_e, hx=hx)

        batch, max_len_e, _ = arc_c.size()
        # transform hn to [num_layers, batch, hidden_size]
        hn = self._transform_decoder_init_state(hn)
        # output from decoder [batch, length_decoder, tag_space]
        arc_h, type_h, _, mask_d, _ = self._get_decoder_output(src_encoding, stacked_heads, hn,
                                                               mask_d=mask_d, length_d=length_d)
        _, max_len_d, _ = arc_h.size()

        if mask_d is not None and children.size(1) != mask_d.size(1):
            stacked_heads = stacked_heads[:, :max_len_d]
            children = children[:, :max_len_d]
            stacked_types = stacked_types[:, :max_len_d]

        # [batch, length_decoder, length_encoder]
        out_arc = self.attention(arc_h, arc_c, mask_d=mask_d, mask_e=mask_e).squeeze(dim=1)

        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(out_arc.data).long()
        # get vector for heads [batch, length_decoder, type_space],
        type_c = type_c[batch_index, children.data.t()].transpose(0, 1).contiguous()
        # compute output for type [batch, length_decoder, num_labels]
        out_type = self.bilinear(type_h, type_c)

        # mask invalid position to -inf for log_softmax
        if mask_e is not None:
            minus_inf = -1e8
            minus_mask_d = (1 - mask_d) * minus_inf
            minus_mask_e = (1 - mask_e) * minus_inf
            out_arc = out_arc + minus_mask_d.unsqueeze(2) + minus_mask_e.unsqueeze(1)

        # TODO for Pytorch 2.0.4, need to set dim=1 for log_softmax or use softmax then take log
        # first convert out_arc to [length_encoder, length_decoder, batch] for log_softmax computation.
        # then convert back to [batch, length_decoder, length_encoder]
        loss_arc = self.logsoftmax(out_arc.transpose(0, 2)).transpose(0, 2)
        # convert out_type to [num_labels, length_decoder, batch] for log_softmax computation.
        # then convert back to [batch, length_decoder, num_labels]
        loss_type = self.logsoftmax(out_type.transpose(0, 2)).transpose(0, 2)

        # get leaf and non-leaf mask
        # shape = [batch, length_decoder]
        mask_leaf = torch.eq(stacked_heads, children).float()
        mask_non_leaf = (1.0 - mask_leaf)

        # mask invalid position to 0 for sum loss
        if mask_e is not None:
            loss_arc = loss_arc * mask_d.unsqueeze(2) * mask_e.unsqueeze(1)
            loss_type = loss_type * mask_d.unsqueeze(2)
            mask_leaf = mask_leaf * mask_d
            mask_non_leaf = mask_non_leaf * mask_d

            # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
            num_leaf = mask_leaf.sum()
            num_non_leaf = mask_non_leaf.sum()
        else:
            # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
            num_leaf = max_len_e
            num_non_leaf = max_len_e - 1

        # first create index matrix [length, batch]
        head_index = torch.arange(0, max_len_d).view(max_len_d, 1).expand(max_len_d, batch)
        head_index = head_index.type_as(out_arc.data).long()
        # [batch, length_decoder]
        loss_arc = loss_arc[batch_index, head_index, children.data.t()].transpose(0, 1)
        loss_arc_leaf = loss_arc * mask_leaf
        loss_arc_non_leaf = loss_arc * mask_non_leaf

        loss_type = loss_type[batch_index, head_index, stacked_types.data.t()].transpose(0, 1)
        loss_type_leaf = loss_type * mask_leaf
        loss_type_non_leaf = loss_type * mask_non_leaf

        return -loss_arc_leaf.sum() / num_leaf, -loss_arc_non_leaf.sum() / num_non_leaf, \
               -loss_type_leaf.sum() / num_leaf, -loss_type_non_leaf.sum() / num_non_leaf, \
               num_leaf, num_non_leaf

    def _decode_per_sentence(self, src_encoding, arc_c, type_c, hx, length, beam, leading_symbolic):
        # src_encoding [length, input_size]
        # arc_c [length, arc_space]
        # type_c [length, type_space]
        # hx [num_direction, hidden_size]
        if length is not None:
            src_encoding = src_encoding[:length]
            arc_c = arc_c[:length]
            type_c = type_c[:length]
        else:
            length = src_encoding.size(0)

        # expand each tensor for beam search
        # [1, length, input_size]
        src_encoding = src_encoding.unsqueeze(0)
        # [1, length, arc_space]
        arc_c = arc_c.unsqueeze(0)
        # [num_direction, 1, hidden_size]
        # hack to handle LSTM
        if isinstance(hx, tuple):
            hx, cx = hx
            hx = hx.unsqueeze(1)
            cx = cx.unsqueeze(1)
            hx = (hx, cx)
        else:
            hx = hx.unsqueeze(1)

        stacked_heads = [[0] for _ in range(beam)]
        children = torch.zeros(beam, 2 * length - 1).type_as(src_encoding.data).long()
        stacked_types = children.new(children.size()).zero_()
        hypothesis_scores = src_encoding.data.new(beam).zero_()
        constraints = np.zeros([beam, length], dtype=np.bool)
        constraints[:, 0] = True

        # temporal tensors for each step.
        new_stacked_heads = [[] for _ in range(beam)]
        new_children = children.new(children.size()).zero_()
        new_stacked_types = stacked_types.new(stacked_types.size()).zero_()
        num_hyp = 1
        num_step = 2 * length - 1
        for t in range(num_step):
            # beam_index = torch.arange(0, num_hyp).type_as(src_encoding.data).long()
            beam_index = src_encoding.data.new(num_hyp).zero_().long()
            # [num_hyp]
            heads = torch.LongTensor([stacked_heads[i][-1] for i in range(num_hyp)]).type_as(children)
            # [num_hyp, 1, input_size]
            input = src_encoding[beam_index, heads].unsqueeze(1)
            # output [num_hyp, 1, hidden_size]
            # hx [num_direction, num_hyp, hidden_size]
            output, hx = self.decoder(input, hx=hx)

            # output size [num_hyp, 1, arc_space]
            arc_h = F.elu(self.arc_h(output))

            # output size [num_hyp, type_space]
            type_h = F.elu(self.type_h(output)).squeeze(dim=1)

            # [num_hyp, length_encoder]
            out_arc = self.attention(arc_h, arc_c[beam_index]).squeeze(dim=1).squeeze(dim=1)
            # [num_hyp, length_encoder]
            hyp_scores = self.logsoftmax(out_arc)
            # [num_hyp, length_encoder]
            new_hypothesis_scores = hypothesis_scores[:num_hyp].unsqueeze(1) + hyp_scores.data
            # [num_hyp * length_encoder]
            new_hypothesis_scores, hyp_index = torch.sort(new_hypothesis_scores.view(-1), dim=0, descending=True)
            base_index = hyp_index / length
            child_index = hyp_index % length

            cc = 0
            ids = []
            new_constraints = np.zeros([beam, length], dtype=np.bool)
            for id in range(num_hyp * length):
                base_id = base_index[id]
                child_id = child_index[id]
                head = heads[base_id]
                new_hyp_score = new_hypothesis_scores[id]
                if head == child_id:
                    assert constraints[base_id, child_id], 'constrains error: %d, %d' % (base_id, child_id)
                    if child_id != 0 or t + 1 == num_step:
                        new_constraints[cc] = constraints[base_id]

                        new_stacked_heads[cc] = [stacked_heads[base_id][i] for i in range(len(stacked_heads[base_id]))]
                        new_stacked_heads[cc].pop()

                        new_children[cc] = children[base_id]
                        new_children[cc, t] = child_id

                        hypothesis_scores[cc] = new_hyp_score
                        ids.append(id)
                        cc += 1
                elif not constraints[base_id, child_id]:
                    new_constraints[cc] = constraints[base_id]
                    new_constraints[cc, child_id] = True

                    new_stacked_heads[cc] = [stacked_heads[base_id][i] for i in range(len(stacked_heads[base_id]))]
                    new_stacked_heads[cc].append(child_id)

                    new_children[cc] = children[base_id]
                    new_children[cc, t] = child_id

                    hypothesis_scores[cc] = new_hyp_score
                    ids.append(id)
                    cc += 1

                if cc == beam:
                    break

            # [num_hyp]
            num_hyp = len(ids)
            if num_hyp == 1:
                index = base_index.new(1).fill_(ids[0])
            else:
                index = torch.from_numpy(np.array(ids)).type_as(base_index)
            base_index = base_index[index]
            child_index = child_index[index]

            # predict types for new hypotheses
            # [num_hyp, type_space]
            hyp_type_c = type_c[child_index]
            hyp_type_h = type_h[base_index]
            # compute output for type [num_hyp, num_labels]
            out_type = self.bilinear(hyp_type_h, hyp_type_c)
            # remove the first #leading_symbolic types.
            out_type = out_type.data[:, leading_symbolic:]
            # compute the prediction of types [num_hyp]
            _, hyp_types = out_type.max(dim=1)
            hyp_types = hyp_types + leading_symbolic
            for i in range(num_hyp):
                base_id = base_index[i]
                new_stacked_types[i] = stacked_types[base_id]
                new_stacked_types[i, t] = hyp_types[i]

            stacked_heads = [[new_stacked_heads[i][j] for j in range(len(new_stacked_heads[i]))] for i in range(num_hyp)]
            constraints = new_constraints
            children.copy_(new_children)
            stacked_types.copy_(new_stacked_types)
            # hx [num_directions, num_hyp, hidden_size]
            # hack to handle LSTM
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx[:, base_index, :]
                cx = cx[:, base_index, :]
                hx = (hx, cx)
            else:
                hx = hx[:, base_index, :]

        children = children.cpu().numpy()[0]
        stacked_types = stacked_types.cpu().numpy()[0]
        heads = np.zeros(length, dtype=np.int32)
        types = np.zeros(length, dtype=np.int32)
        stack = [0]
        for i in range(num_step):
            head = stack[-1]
            child = children[i]
            type = stacked_types[i]
            if head != child:
                heads[child] = head
                types[child] = type
                stack.append(child)
            else:
                stack.pop()

        return heads, types, length

    def decode(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, leading_symbolic=0, beam=1):
        # output from encoder [batch, length_encoder, tag_space]
        # src_encoding [batch, length, input_size]
        # arc_c [batch, length, arc_space]
        # type_c [batch, length, type_space]
        # hn [num_direction, batch, hidden_size]
        src_encoding, arc_c, type_c, hn, mask, length = self._get_encoder_output(input_word, input_char, input_pos,
                                                                                   mask_e=mask, length_e=length, hx=hx)
        hn = self._transform_decoder_init_state(hn)
        batch, max_len_e, _ = src_encoding.size()

        heads = np.zeros([batch, max_len_e], dtype=np.int32)
        types = np.zeros([batch, max_len_e], dtype=np.int32)

        for b in range(batch):
            sent_len = None if length is None else length[b]
            # hack to handle LSTM
            if isinstance(hn, tuple):
                hx, cx = hn
                hx = hx[:, b, :].contiguous()
                cx = cx[:, b, :].contiguous()
                hx = (hx, cx)
            else:
                hx = hn[:, b, :].contiguous()

            hids, tids, sent_len = self._decode_per_sentence(src_encoding[b], arc_c[b], type_c[b], hx, sent_len, beam,
                                                             leading_symbolic)
            heads[b, :sent_len] = hids
            types[b, :sent_len] = tids

        return heads, types


class StackVarPtrNet(StackPtrNet):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, arc_space, type_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None,
                 p_in=0.2, p_rnn=0.5, biaffine=False):

        super(StackVarPtrNet, self).__init__(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                                             num_filters, kernel_size, rnn_mode, hidden_size, num_layers,
                                             num_labels, arc_space, type_space,
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

        self.encoder = RNN(word_dim + num_filters + pos_dim, hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=True, dropout=p_rnn)

        self.decoder = RNN(word_dim + num_filters + pos_dim, hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=False, dropout=p_rnn)

    def _get_encoder_output(self, input_word, input_char, input_pos, mask_e=None, length_e=None, hx=None):
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
        char = torch.tanh(char).view(char_size[0], char_size[1], -1)

        # concatenate word and char [batch, length, word_dim+char_filter]
        src_encoding = torch.cat([word, char, pos], dim=2)
        # apply dropout
        # [batch, length, dim] --> [batch, dim, length] --> [batch, length, dim]
        input = self.dropout_in(src_encoding.transpose(1, 2)).transpose(1, 2)
        # output from rnn [batch, length, hidden_size]
        output, hn = self.encoder(input, mask_e, hx=hx)
        # apply dropout
        output = self.dropout_rnn(output.transpose(1, 2)).transpose(1, 2)

        # output size [batch, length, arc_space]
        arc_c = F.elu(self.arc_c(output))

        # output size [batch, length, type_space]
        type_c = F.elu(self.type_c(output))

        return src_encoding, arc_c, type_c, hn, mask_e, length_e

    def _get_decoder_output(self, src_encoding, heads_stack, hx, mask_d=None, length_d=None):
        batch, _, _ = src_encoding.size()
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(src_encoding.data).long()
        # get vector for heads [batch, length_decoder, input_dim],
        src_encoding = src_encoding[batch_index, heads_stack.data.t()].transpose(0, 1)
        # apply dropout
        # [batch, length_decoder, dim] --> [batch, dim, length_decoder] --> [batch, length_decoder, dim]
        input = self.dropout_in(src_encoding.transpose(1, 2)).transpose(1, 2)
        # output from rnn [batch, length, hidden_size]
        output, hn = self.decoder(input, mask_d, hx=hx)
        # apply dropout
        output = self.dropout_rnn(output.transpose(1, 2)).transpose(1, 2)

        # output size [batch, length_decoder, arc_space]
        arc_h = F.elu(self.arc_h(output))

        # output size [batch, length_decoder, type_space]
        type_h = F.elu(self.type_h(output))

        return arc_h, type_h, hn, mask_d, length_d
