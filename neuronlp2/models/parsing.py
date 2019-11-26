__author__ = 'max'

from overrides import overrides
import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronlp2.nn import TreeCRF, VarGRU, VarRNN, VarLSTM, VarFastLSTM
from neuronlp2.nn import BiAffine, BiLinear, CharCNN
from neuronlp2.tasks import parser


class PriorOrder(Enum):
    DEPTH = 0
    INSIDE_OUT = 1
    LEFT2RIGTH = 2


class DeepBiAffine(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, rnn_mode, hidden_size, num_layers, num_labels, arc_space, type_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33), pos=True, activation='elu'):
        super(DeepBiAffine, self).__init__()

        self.word_embed = nn.Embedding(num_words, word_dim, _weight=embedd_word, padding_idx=1)
        self.pos_embed = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos, padding_idx=1) if pos else None
        self.char_embed = nn.Embedding(num_chars, char_dim, _weight=embedd_char, padding_idx=1)
        self.char_cnn = CharCNN(2, char_dim, char_dim, hidden_channels=char_dim * 4, activation=activation)

        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.num_labels = num_labels

        if rnn_mode == 'RNN':
            RNN = VarRNN
        elif rnn_mode == 'LSTM':
            RNN = VarLSTM
        elif rnn_mode == 'FastLSTM':
            RNN = VarFastLSTM
        elif rnn_mode == 'GRU':
            RNN = VarGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        dim_enc = word_dim + char_dim
        if pos:
            dim_enc += pos_dim

        self.rnn = RNN(dim_enc, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn)

        out_dim = hidden_size * 2
        self.arc_h = nn.Linear(out_dim, arc_space)
        self.arc_c = nn.Linear(out_dim, arc_space)
        self.biaffine = BiAffine(arc_space, arc_space)

        self.type_h = nn.Linear(out_dim, type_space)
        self.type_c = nn.Linear(out_dim, type_space)
        self.bilinear = BiLinear(type_space, type_space, self.num_labels)

        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Tanh()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters(embedd_word, embedd_char, embedd_pos)

    def reset_parameters(self, embedd_word, embedd_char, embedd_pos):
        if embedd_word is None:
            nn.init.uniform_(self.word_embed.weight, -0.1, 0.1)
        if embedd_char is None:
            nn.init.uniform_(self.char_embed.weight, -0.1, 0.1)
        if embedd_pos is None and self.pos_embed is not None:
            nn.init.uniform_(self.pos_embed.weight, -0.1, 0.1)

        with torch.no_grad():
            self.word_embed.weight[self.word_embed.padding_idx].fill_(0)
            self.char_embed.weight[self.char_embed.padding_idx].fill_(0)
            if self.pos_embed is not None:
                self.pos_embed.weight[self.pos_embed.padding_idx].fill_(0)

        nn.init.xavier_uniform_(self.arc_h.weight)
        nn.init.constant_(self.arc_h.bias, 0.)
        nn.init.xavier_uniform_(self.arc_c.weight)
        nn.init.constant_(self.arc_c.bias, 0.)

        nn.init.xavier_uniform_(self.type_h.weight)
        nn.init.constant_(self.type_h.bias, 0.)
        nn.init.xavier_uniform_(self.type_c.weight)
        nn.init.constant_(self.type_c.bias, 0.)

    def _get_rnn_output(self, input_word, input_char, input_pos, mask=None):
        # [batch, length, word_dim]
        word = self.word_embed(input_word)

        # [batch, length, char_length, char_dim]
        char = self.char_cnn(self.char_embed(input_char))

        # apply dropout word on input
        word = self.dropout_in(word)
        char = self.dropout_in(char)

        # concatenate word and char [batch, length, word_dim+char_filter]
        enc = torch.cat([word, char], dim=2)

        if self.pos_embed is not None:
            # [batch, length, pos_dim]
            pos = self.pos_embed(input_pos)
            # apply dropout on input
            pos = self.dropout_in(pos)
            enc = torch.cat([enc, pos], dim=2)

        # output from rnn [batch, length, hidden_size]
        output, _ = self.rnn(enc, mask)

        # apply dropout for output
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        # output size [batch, length, arc_space]
        arc_h = self.activation(self.arc_h(output))
        arc_c = self.activation(self.arc_c(output))

        # output size [batch, length, type_space]
        type_h = self.activation(self.type_h(output))
        type_c = self.activation(self.type_c(output))

        # apply dropout on arc
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=1)
        type = torch.cat([type_h, type_c], dim=1)
        arc = self.dropout_out(arc.transpose(1, 2)).transpose(1, 2)
        arc_h, arc_c = arc.chunk(2, 1)

        # apply dropout on type
        # [batch, length, dim] --> [batch, 2 * length, dim]
        type = self.dropout_out(type.transpose(1, 2)).transpose(1, 2)
        type_h, type_c = type.chunk(2, 1)
        type_h = type_h.contiguous()
        type_c = type_c.contiguous()

        return (arc_h, arc_c), (type_h, type_c)

    def forward(self, input_word, input_char, input_pos, mask=None):
        # output from rnn [batch, length, dim]
        arc, type = self._get_rnn_output(input_word, input_char, input_pos, mask=mask)
        # [batch, length_head, length_child]
        out_arc = self.biaffine(arc[0], arc[1], mask_query=mask, mask_key=mask)
        return out_arc, type

    def loss(self, input_word, input_char, input_pos, heads, types, mask=None):
        # out_arc shape [batch, length_head, length_child]
        out_arc, out_type  = self(input_word, input_char, input_pos, mask=mask)
        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type

        # get vector for heads [batch, length, type_space],
        type_h = type_h.gather(dim=1, index=heads.unsqueeze(2).expand(type_h.size()))
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(type_h, type_c)

        # mask invalid position to -inf for log_softmax
        if mask is not None:
            minus_mask = mask.eq(0).unsqueeze(2)
            out_arc = out_arc.masked_fill(minus_mask, float('-inf'))

        # loss_arc shape [batch, length_c]
        loss_arc = self.criterion(out_arc, heads)
        loss_type = self.criterion(out_type.transpose(1, 2), types)

        # mask invalid position to 0 for sum loss
        if mask is not None:
            loss_arc = loss_arc * mask
            loss_type = loss_type * mask

        # [batch, length - 1] -> [batch] remove the symbolic root.
        return loss_arc[:, 1:].sum(dim=1), loss_type[:, 1:].sum(dim=1)

    def _decode_types(self, out_type, heads, leading_symbolic):
        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type
        # get vector for heads [batch, length, type_space],
        type_h = type_h.gather(dim=1, index=heads.unsqueeze(2).expand(type_h.size()))
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(type_h, type_c)
        # remove the first #leading_symbolic types.
        out_type = out_type[:, :, leading_symbolic:]
        # compute the prediction of types [batch, length]
        _, types = out_type.max(dim=2)
        return types + leading_symbolic

    def decode_local(self, input_word, input_char, input_pos, mask=None, leading_symbolic=0):
        # out_arc shape [batch, length_h, length_c]
        out_arc, out_type = self(input_word, input_char, input_pos, mask=mask)
        batch, max_len, _ = out_arc.size()
        # set diagonal elements to -inf
        diag_mask = torch.eye(max_len, device=out_arc.device, dtype=torch.uint8).unsqueeze(0)
        out_arc.masked_fill_(diag_mask, float('-inf'))
        # set invalid positions to -inf
        if mask is not None:
            minus_mask = mask.eq(0).unsqueeze(2)
            out_arc.masked_fill_(minus_mask, float('-inf'))

        # compute naive predictions.
        # predition shape = [batch, length_c]
        _, heads = out_arc.max(dim=1)

        types = self._decode_types(out_type, heads, leading_symbolic)

        return heads.cpu().numpy(), types.cpu().numpy()

    def decode(self, input_word, input_char, input_pos, mask=None, leading_symbolic=0):
        """
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

        """
        # out_arc shape [batch, length_h, length_c]
        out_arc, out_type = self(input_word, input_char, input_pos, mask=mask)

        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type
        batch, max_len, type_space = type_h.size()

        type_h = type_h.unsqueeze(2).expand(batch, max_len, max_len, type_space).contiguous()
        type_c = type_c.unsqueeze(1).expand(batch, max_len, max_len, type_space).contiguous()
        # compute output for type [batch, length_h, length_c, num_labels]
        out_type = self.bilinear(type_h, type_c)

        if mask is not None:
            minus_mask = mask.eq(0).unsqueeze(2)
            out_arc.masked_fill_(minus_mask, float('-inf'))
        # loss_arc shape [batch, length_h, length_c]
        loss_arc = F.log_softmax(out_arc, dim=1)
        # loss_type shape [batch, length_h, length_c, num_labels]
        loss_type = F.log_softmax(out_type, dim=3).permute(0, 3, 1, 2)
        # [batch, num_labels, length_h, length_c]
        energy = loss_arc.unsqueeze(1) + loss_type

        # compute lengths
        length = mask.sum(dim=1).long().cpu().numpy()
        return parser.decode_MST(energy.cpu().numpy(), length, leading_symbolic=leading_symbolic, labeled=True)


class NeuroMST(DeepBiAffine):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, rnn_mode, hidden_size, num_layers, num_labels, arc_space, type_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33), pos=True, activation='elu'):
        super(NeuroMST, self).__init__(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, rnn_mode, hidden_size, num_layers, num_labels, arc_space, type_space,
                                       embedd_word=embedd_word, embedd_char=embedd_char, embedd_pos=embedd_pos, p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=pos, activation=activation)
        self.biaffine = None
        self.treecrf = TreeCRF(arc_space)

    def forward(self, input_word, input_char, input_pos, mask=None):
        # output from rnn [batch, length, dim]
        arc, type = self._get_rnn_output(input_word, input_char, input_pos, mask=mask)
        # [batch, length_head, length_child]
        out_arc = self.treecrf(arc[0], arc[1], mask=mask)
        return out_arc, type

    @overrides
    def loss(self, input_word, input_char, input_pos, heads, types, mask=None):
        # output from rnn [batch, length, dim]
        arc, out_type = self._get_rnn_output(input_word, input_char, input_pos, mask=mask)
        # [batch]
        loss_arc = self.treecrf.loss(arc[0], arc[1], heads, mask=mask)
        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type

        # get vector for heads [batch, length, type_space],
        type_h = type_h.gather(dim=1, index=heads.unsqueeze(2).expand(type_h.size()))
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(type_h, type_c)
        loss_type = self.criterion(out_type.transpose(1, 2), types)

        # mask invalid position to 0 for sum loss
        if mask is not None:
            loss_type = loss_type * mask

        return loss_arc, loss_type[:, 1:].sum(dim=1)

    @overrides
    def decode(self, input_word, input_char, input_pos, mask=None, leading_symbolic=0):
        """
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

        """
        # out_arc shape [batch, length_h, length_c]
        energy, out_type = self(input_word, input_char, input_pos, mask=mask)
        # compute lengths
        length = mask.sum(dim=1).long()
        heads, _ = parser.decode_MST(energy.cpu().numpy(), length.cpu().numpy(), leading_symbolic=leading_symbolic, labeled=False)
        types = self._decode_types(out_type, torch.from_numpy(heads).type_as(length), leading_symbolic)
        return heads, types.cpu().numpy()


class StackPtrNet(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, rnn_mode, hidden_size,
                 encoder_layers, decoder_layers, num_labels, arc_space, type_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33),
                 pos=True, prior_order='inside_out', grandPar=False, sibling=False, activation='elu'):

        super(StackPtrNet, self).__init__()
        self.word_embed = nn.Embedding(num_words, word_dim, _weight=embedd_word, padding_idx=1)
        self.pos_embed = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos, padding_idx=1) if pos else None
        self.char_embed = nn.Embedding(num_chars, char_dim, _weight=embedd_char, padding_idx=1)
        self.char_cnn = CharCNN(2, char_dim, char_dim, hidden_channels=char_dim * 4, activation=activation)

        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.num_labels = num_labels

        if prior_order in ['deep_first', 'shallow_first']:
            self.prior_order = PriorOrder.DEPTH
        elif prior_order == 'inside_out':
            self.prior_order = PriorOrder.INSIDE_OUT
        elif prior_order == 'left2right':
            self.prior_order = PriorOrder.LEFT2RIGTH
        else:
            raise ValueError('Unknown prior order: %s' % prior_order)

        self.grandPar = grandPar
        self.sibling = sibling

        if rnn_mode == 'RNN':
            RNN_ENCODER = VarRNN
            RNN_DECODER = VarRNN
        elif rnn_mode == 'LSTM':
            RNN_ENCODER = VarLSTM
            RNN_DECODER = VarLSTM
        elif rnn_mode == 'FastLSTM':
            RNN_ENCODER = VarFastLSTM
            RNN_DECODER = VarFastLSTM
        elif rnn_mode == 'GRU':
            RNN_ENCODER = VarGRU
            RNN_DECODER = VarGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        dim_enc = word_dim + char_dim
        if pos:
            dim_enc += pos_dim

        self.encoder_layers = encoder_layers
        self.encoder = RNN_ENCODER(dim_enc, hidden_size, num_layers=encoder_layers, batch_first=True, bidirectional=True, dropout=p_rnn)

        dim_dec = hidden_size // 2
        self.src_dense = nn.Linear(2 * hidden_size, dim_dec)
        self.decoder_layers = decoder_layers
        self.decoder = RNN_DECODER(dim_dec, hidden_size, num_layers=decoder_layers, batch_first=True, bidirectional=False, dropout=p_rnn)

        self.hx_dense = nn.Linear(2 * hidden_size, hidden_size)

        self.arc_h = nn.Linear(hidden_size, arc_space) # arc dense for decoder
        self.arc_c = nn.Linear(hidden_size * 2, arc_space)  # arc dense for encoder
        self.biaffine = BiAffine(arc_space, arc_space)

        self.type_h = nn.Linear(hidden_size, type_space) # type dense for decoder
        self.type_c = nn.Linear(hidden_size * 2, type_space)  # type dense for encoder
        self.bilinear = BiLinear(type_space, type_space, self.num_labels)

        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Tanh()

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters(embedd_word, embedd_char, embedd_pos)

    def reset_parameters(self, embedd_word, embedd_char, embedd_pos):
        if embedd_word is None:
            nn.init.uniform_(self.word_embed.weight, -0.1, 0.1)
        if embedd_char is None:
            nn.init.uniform_(self.char_embed.weight, -0.1, 0.1)
        if embedd_pos is None and self.pos_embed is not None:
            nn.init.uniform_(self.pos_embed.weight, -0.1, 0.1)

        with torch.no_grad():
            self.word_embed.weight[self.word_embed.padding_idx].fill_(0)
            self.char_embed.weight[self.char_embed.padding_idx].fill_(0)
            if self.pos_embed is not None:
                self.pos_embed.weight[self.pos_embed.padding_idx].fill_(0)

        nn.init.xavier_uniform_(self.arc_h.weight)
        nn.init.constant_(self.arc_h.bias, 0.)
        nn.init.xavier_uniform_(self.arc_c.weight)
        nn.init.constant_(self.arc_c.bias, 0.)

        nn.init.xavier_uniform_(self.type_h.weight)
        nn.init.constant_(self.type_h.bias, 0.)
        nn.init.xavier_uniform_(self.type_c.weight)
        nn.init.constant_(self.type_c.bias, 0.)

    def _get_encoder_output(self, input_word, input_char, input_pos, mask=None):
        # [batch, length, word_dim]
        word = self.word_embed(input_word)

        # [batch, length, char_length, char_dim]
        char = self.char_cnn(self.char_embed(input_char))

        # apply dropout word on input
        word = self.dropout_in(word)
        char = self.dropout_in(char)

        # concatenate word and char [batch, length, word_dim+char_filter]
        enc = torch.cat([word, char], dim=2)

        if self.pos_embed is not None:
            # [batch, length, pos_dim]
            pos = self.pos_embed(input_pos)
            # apply dropout on input
            pos = self.dropout_in(pos)
            enc = torch.cat([enc, pos], dim=2)

        # output from rnn [batch, length, hidden_size]
        output, hn = self.encoder(enc, mask)
        # apply dropout
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn

    def _get_decoder_output(self, output_enc, heads, heads_stack, siblings, hx, mask=None):
        batch, _, _ = output_enc.size()
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(output_enc).long()
        # get vector for heads [batch, length_decoder, input_dim],
        src_encoding = output_enc[batch_index, heads_stack.t()].transpose(0, 1)

        if self.sibling:
            # [batch, length_decoder, hidden_size * 2]
            mask_sib = siblings.gt(0).float().unsqueeze(2)
            output_enc_sibling = output_enc[batch_index, siblings.t()].transpose(0, 1) * mask_sib
            src_encoding = src_encoding + output_enc_sibling

        if self.grandPar:
            # [length_decoder, batch]
            gpars = heads[batch_index, heads_stack.t()]
            mask_gpar = gpars.t().ge(0).float().unsqueeze(2)
            # [batch, length_decoder, hidden_size * 2]
            output_enc_gpar = output_enc[batch_index, gpars].transpose(0, 1) * mask_gpar
            src_encoding = src_encoding + output_enc_gpar

        # transform to decoder input
        # [batch, length_decoder, dec_dim]
        src_encoding = self.activation(self.src_dense(src_encoding))
        # output from rnn [batch, length, hidden_size]
        output, hn = self.decoder(src_encoding, mask, hx=hx)
        # apply dropout
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn

    def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        raise RuntimeError('Stack Pointer Network does not implement forward')

    def _transform_decoder_init_state(self, hn):
        if isinstance(hn, tuple):
            hn, cn = hn
            _, batch, hidden_size = cn.size()
            # take the last layers
            # [batch, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            cn = torch.cat([cn[-2], cn[-1]], dim=1).unsqueeze(0)
            # take hx_dense to [1, batch, hidden_size]
            cn = self.hx_dense(cn)
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                cn = torch.cat([cn, cn.new_zeros(self.decoder_layers - 1, batch, hidden_size)], dim=0)
            # hn is tanh(cn)
            hn = torch.tanh(cn)
            hn = (hn, cn)
        else:
            # take the last layers
            # [2, batch, hidden_size]
            hn = hn[-2:]
            # hn [2, batch, hidden_size]
            _, batch, hidden_size = hn.size()
            # first convert hn t0 [batch, 2, hidden_size]
            hn = hn.transpose(0, 1).contiguous()
            # then view to [batch, 1, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            hn = hn.view(batch, 1, 2 * hidden_size).transpose(0, 1)
            # take hx_dense to [1, batch, hidden_size]
            hn = torch.tanh(self.hx_dense(hn))
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                hn = torch.cat([hn, hn.new_zeros(self.decoder_layers - 1, batch, hidden_size)], dim=0)
        return hn

    def loss(self, input_word, input_char, input_pos, heads, stacked_heads, children, siblings, stacked_types, mask_e=None, mask_d=None):
        # output from encoder [batch, length_encoder, hidden_size]
        output_enc, hn = self._get_encoder_output(input_word, input_char, input_pos, mask=mask_e)

        # output size [batch, length_encoder, arc_space]
        arc_c = self.activation(self.arc_c(output_enc))
        # output size [batch, length_encoder, type_space]
        type_c = self.activation(self.type_c(output_enc))

        # transform hn to [decoder_layers, batch, hidden_size]
        hn = self._transform_decoder_init_state(hn)

        # output from decoder [batch, length_decoder, tag_space]
        output_dec, _ = self._get_decoder_output(output_enc, heads, stacked_heads, siblings, hn, mask=mask_d)

        # output size [batch, length_decoder, arc_space]
        arc_h = self.activation(self.arc_h(output_dec))
        type_h = self.activation(self.type_h(output_dec))

        _, max_len_d, _ = arc_h.size()

        # apply dropout
        # [batch, length_decoder, dim] + [batch, length_encoder, dim] --> [batch, length_decoder + length_encoder, dim]
        arc = self.dropout_out(torch.cat([arc_h, arc_c], dim=1).transpose(1, 2)).transpose(1, 2)
        arc_h = arc[:, :max_len_d]
        arc_c = arc[:, max_len_d:]

        type = self.dropout_out(torch.cat([type_h, type_c], dim=1).transpose(1, 2)).transpose(1, 2)
        type_h = type[:, :max_len_d].contiguous()
        type_c = type[:, max_len_d:]

        # [batch, length_decoder, length_encoder]
        out_arc = self.biaffine(arc_h, arc_c, mask_query=mask_d, mask_key=mask_e)

        batch = arc_c.size(0)
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(arc_c).long()
        # get vector for heads [batch, length_decoder, type_space],
        type_c = type_c[batch_index, children.t()].transpose(0, 1).contiguous()
        # compute output for type [batch, length_decoder, num_labels]
        out_type = self.bilinear(type_h, type_c)

        # mask invalid position to -inf for log_softmax
        if mask_e is not None:
            minus_mask_e = mask_e.eq(0).unsqueeze(1)
            minus_mask_d = mask_d.eq(0).unsqueeze(2)
            out_arc = out_arc.masked_fill(minus_mask_d * minus_mask_e, float('-inf'))

        # loss_arc shape [batch, length_decoder]
        loss_arc = self.criterion(out_arc.transpose(1, 2), children)
        loss_type = self.criterion(out_type.transpose(1, 2), stacked_types)

        if mask_d is not None:
            loss_arc = loss_arc * mask_d
            loss_type = loss_type * mask_d

        return loss_arc.sum(dim=1), loss_type.sum(dim=1)

    def _decode_per_sentence(self, output_enc, arc_c, type_c, hx, length, beam, ordered, leading_symbolic):
        def valid_hyp(base_id, child_id, head):
            if constraints[base_id, child_id]:
                return False
            elif not ordered or self.prior_order == PriorOrder.DEPTH or child_orders[base_id, head] == 0:
                return True
            elif self.prior_order == PriorOrder.LEFT2RIGTH:
                return child_id > child_orders[base_id, head]
            else:
                if child_id < head:
                    return child_id < child_orders[base_id, head] < head
                else:
                    return child_id > child_orders[base_id, head]

        # output_enc [length, hidden_size * 2]
        # arc_c [length, arc_space]
        # type_c [length, type_space]
        # hx [decoder_layers, hidden_size]
        output_enc = output_enc[:length]
        arc_c = arc_c[:length]
        type_c = type_c[:length]

        # [decoder_layers, 1, hidden_size]
        # hack to handle LSTM
        if isinstance(hx, tuple):
            hx, cx = hx
            hx = hx.unsqueeze(1)
            cx = cx.unsqueeze(1)
            hx = (hx, cx)
        else:
            hx = hx.unsqueeze(1)

        stacked_heads = [[0] for _ in range(beam)]
        grand_parents = [[0] for _ in range(beam)] if self.grandPar else None
        siblings = [[0] for _ in range(beam)] if self.sibling else None
        children = torch.zeros(beam, 2 * length - 1).type_as(output_enc).long()
        stacked_types = children.new_zeros(children.size())
        hypothesis_scores = output_enc.new_zeros(beam)
        constraints = np.zeros([beam, length], dtype=np.bool)
        constraints[:, 0] = True
        child_orders = np.zeros([beam, length], dtype=np.int64)

        # temporal tensors for each step.
        new_stacked_heads = [[] for _ in range(beam)]
        new_grand_parents = [[] for _ in range(beam)] if self.grandPar else None
        new_siblings = [[] for _ in range(beam)] if self.sibling else None
        new_children = children.new_zeros(children.size())
        new_stacked_types = stacked_types.new_zeros(stacked_types.size())
        num_hyp = 1
        num_step = 2 * length - 1
        for t in range(num_step):
            # [num_hyp]
            heads = torch.LongTensor([stacked_heads[i][-1] for i in range(num_hyp)]).type_as(children)
            gpars = torch.LongTensor([grand_parents[i][-1] for i in range(num_hyp)]).type_as(children) if self.grandPar else None
            sibs = torch.LongTensor([siblings[i].pop() for i in range(num_hyp)]).type_as(children) if self.sibling else None

            # [num_hyp, hidden_size * 2]
            src_encoding = output_enc[heads]

            if self.sibling:
                mask_sibs = sibs.ne(0).float().unsqueeze(1)
                output_enc_sibling = output_enc[sibs] * mask_sibs
                src_encoding = src_encoding + output_enc_sibling

            if self.grandPar:
                output_enc_gpar = output_enc[gpars]
                src_encoding = src_encoding + output_enc_gpar

            # transform to decoder input
            # [num_hyp, dec_dim]
            src_encoding = self.activation(self.src_dense(src_encoding))

            # output [num_hyp, hidden_size]
            # hx [decoder_layer, num_hyp, hidden_size]
            output_dec, hx = self.decoder.step(src_encoding, hx=hx)

            # arc_h size [num_hyp, 1, arc_space]
            arc_h = self.activation(self.arc_h(output_dec.unsqueeze(1)))
            # type_h size [num_hyp, type_space]
            type_h = self.activation(self.type_h(output_dec))

            # [num_hyp, length_encoder]
            out_arc = self.biaffine(arc_h, arc_c.expand(num_hyp, *arc_c.size())).squeeze(dim=1).squeeze(dim=1)

            # [num_hyp, length_encoder]
            hyp_scores = F.log_softmax(out_arc, dim=1)

            new_hypothesis_scores = hypothesis_scores[:num_hyp].unsqueeze(1) + hyp_scores
            # [num_hyp * length_encoder]
            new_hypothesis_scores, hyp_index = torch.sort(new_hypothesis_scores.view(-1), dim=0, descending=True)
            base_index = hyp_index / length
            child_index = hyp_index % length

            cc = 0
            ids = []
            new_constraints = np.zeros([beam, length], dtype=np.bool)
            new_child_orders = np.zeros([beam, length], dtype=np.int64)
            for id in range(num_hyp * length):
                base_id = base_index[id]
                child_id = child_index[id]
                head = heads[base_id]
                new_hyp_score = new_hypothesis_scores[id]
                if child_id == head:
                    assert constraints[base_id, child_id], 'constrains error: %d, %d' % (base_id, child_id)
                    if head != 0 or t + 1 == num_step:
                        new_constraints[cc] = constraints[base_id]
                        new_child_orders[cc] = child_orders[base_id]

                        new_stacked_heads[cc] = [stacked_heads[base_id][i] for i in range(len(stacked_heads[base_id]))]
                        new_stacked_heads[cc].pop()

                        if self.grandPar:
                            new_grand_parents[cc] = [grand_parents[base_id][i] for i in range(len(grand_parents[base_id]))]
                            new_grand_parents[cc].pop()

                        if self.sibling:
                            new_siblings[cc] = [siblings[base_id][i] for i in range(len(siblings[base_id]))]

                        new_children[cc] = children[base_id]
                        new_children[cc, t] = child_id

                        hypothesis_scores[cc] = new_hyp_score
                        ids.append(id)
                        cc += 1
                elif valid_hyp(base_id, child_id, head):
                    new_constraints[cc] = constraints[base_id]
                    new_constraints[cc, child_id] = True

                    new_child_orders[cc] = child_orders[base_id]
                    new_child_orders[cc, head] = child_id

                    new_stacked_heads[cc] = [stacked_heads[base_id][i] for i in range(len(stacked_heads[base_id]))]
                    new_stacked_heads[cc].append(child_id)

                    if self.grandPar:
                        new_grand_parents[cc] = [grand_parents[base_id][i] for i in range(len(grand_parents[base_id]))]
                        new_grand_parents[cc].append(head)

                    if self.sibling:
                        new_siblings[cc] = [siblings[base_id][i] for i in range(len(siblings[base_id]))]
                        new_siblings[cc].append(child_id)
                        new_siblings[cc].append(0)

                    new_children[cc] = children[base_id]
                    new_children[cc, t] = child_id

                    hypothesis_scores[cc] = new_hyp_score
                    ids.append(id)
                    cc += 1

                if cc == beam:
                    break

            # [num_hyp]
            num_hyp = len(ids)
            if num_hyp == 0:
                return None
            else:
                index = torch.from_numpy(np.array(ids)).type_as(base_index)
            base_index = base_index[index]
            child_index = child_index[index]

            # predict types for new hypotheses
            # compute output for type [num_hyp, num_labels]
            out_type = self.bilinear(type_h[base_index], type_c[child_index])
            hyp_type_scores = F.log_softmax(out_type, dim=1)
            # compute the prediction of types [num_hyp]
            hyp_type_scores, hyp_types = hyp_type_scores.max(dim=1)
            hypothesis_scores[:num_hyp] = hypothesis_scores[:num_hyp] + hyp_type_scores

            for i in range(num_hyp):
                base_id = base_index[i]
                new_stacked_types[i] = stacked_types[base_id]
                new_stacked_types[i, t] = hyp_types[i]

            stacked_heads = [[new_stacked_heads[i][j] for j in range(len(new_stacked_heads[i]))] for i in range(num_hyp)]
            if self.grandPar:
                grand_parents = [[new_grand_parents[i][j] for j in range(len(new_grand_parents[i]))] for i in range(num_hyp)]
            if self.sibling:
                siblings = [[new_siblings[i][j] for j in range(len(new_siblings[i]))] for i in range(num_hyp)]
            constraints = new_constraints
            child_orders = new_child_orders
            children.copy_(new_children)
            stacked_types.copy_(new_stacked_types)
            # hx [decoder_layers, num_hyp, hidden_size]
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
            if child != head:
                heads[child] = head
                types[child] = type
                stack.append(child)
            else:
                stacked_types[i] = 0
                stack.pop()

        return heads, types, children, stacked_types

    def decode(self, input_word, input_char, input_pos, mask=None, beam=1, leading_symbolic=0, ordered=True):
        # reset noise for decoder
        self.decoder.reset_noise(0)

        # output from encoder [batch, length_encoder, tag_space]
        # output_enc [batch, length, model_dim]
        # arc_c [batch, length, arc_space]
        # type_c [batch, length, type_space]
        # hn [num_direction, batch, hidden_size]
        output_enc, hn = self._get_encoder_output(input_word, input_char, input_pos, mask=mask)
        # output size [batch, length_encoder, arc_space]
        arc_c = self.activation(self.arc_c(output_enc))
        # output size [batch, length_encoder, type_space]
        type_c = self.activation(self.type_c(output_enc))
        # [decoder_layers, batch, hidden_size
        hn = self._transform_decoder_init_state(hn)
        batch, max_len, _ = output_enc.size()

        heads = np.zeros([batch, max_len], dtype=np.int32)
        types = np.zeros([batch, max_len], dtype=np.int32)

        children = np.zeros([batch, 2 * max_len - 1], dtype=np.int32)
        stack_types = np.zeros([batch, 2 * max_len - 1], dtype=np.int32)

        # compute lengths
        if mask is None:
            length = [max_len] * batch
        else:
            length = mask.sum(dim=1).long()
        for b in range(batch):
            sent_len = length[b]
            # hack to handle LSTM
            if isinstance(hn, tuple):
                hx, cx = hn
                hx = hx[:, b, :].contiguous()
                cx = cx[:, b, :].contiguous()
                hx = (hx, cx)
            else:
                hx = hn[:, b, :].contiguous()

            preds = self._decode_per_sentence(output_enc[b], arc_c[b], type_c[b], hx, sent_len, beam, ordered, leading_symbolic)
            if preds is None:
                preds = self._decode_per_sentence(output_enc[b], arc_c[b], type_c[b], hx, sent_len, beam, False, leading_symbolic)
            hids, tids, chids, stids = preds
            heads[b, :sent_len] = hids
            types[b, :sent_len] = tids

            children[b, :2 * sent_len - 1] = chids
            stack_types[b, :2 * sent_len - 1] = stids

        return heads, types#, children, stack_types

    def decode_batch(self, input_word, input_char, input_pos, mask=None, beam=1, leading_symbolic=0, ordered=True):
        # reset noise for decoder
        self.decoder.reset_noise(0)

        # output from encoder [batch, length_encoder, tag_space]
        # output_enc [batch, length, model_dim]
        # arc_c [batch, length, arc_space]
        # type_c [batch, length, type_space]
        # hn [num_direction, batch, hidden_size]
        output_enc, hn = self._get_encoder_output(input_word, input_char, input_pos, mask=mask)
        device = output_enc.device
        # output size [batch, length_encoder, arc_space]
        arc_c = self.activation(self.arc_c(output_enc))
        # output size [batch, length_encoder, type_space]
        type_c = self.activation(self.type_c(output_enc))
        # [decoder_layers, batch, hidden_size
        hn = self._transform_decoder_init_state(hn)
        batch, max_len, _ = output_enc.size()

        heads = np.zeros([batch, max_len], dtype=np.int32)
        types = np.zeros([batch, max_len], dtype=np.int32)

        stacked_heads = torch.zeros(batch, beam, max_len, device=device, dtype=torch.int64)
        children = torch.zeros(batch, beam, max_len, device=device, dtype=torch.int64)
        stack_types = torch.zeros(batch, beam, max_len, device=device, dtype=torch.int64)
        grand_parents = torch.zeros(batch, beam, max_len, device=device, dtype=torch.int64) if self.grandPar else None
        siblings = torch.zeros(batch, beam, max_len, device=device, dtype=torch.int64) if self.sibling else None

        # compute lengths
        if mask is None:
            length = [max_len] * batch
        else:
            length = mask.sum(dim=1).long()



