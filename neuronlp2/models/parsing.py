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
        # get vector for heads [batch, length_decoder, input_dim],
        enc_dim = output_enc.size(2)
        batch, length_dec = heads_stack.size()
        src_encoding = output_enc.gather(dim=1, index=heads_stack.unsqueeze(2).expand(batch, length_dec, enc_dim))

        if self.sibling:
            # [batch, length_decoder, hidden_size * 2]
            mask_sib = siblings.gt(0).float().unsqueeze(2)
            output_enc_sibling = output_enc.gather(dim=1, index=siblings.unsqueeze(2).expand(batch, length_dec, enc_dim)) * mask_sib
            src_encoding = src_encoding + output_enc_sibling

        if self.grandPar:
            # [batch, length_decoder, 1]
            gpars = heads.gather(dim=1, index=heads_stack).unsqueeze(2)
            # mask_gpar = gpars.ge(0).float()
            # [batch, length_decoder, hidden_size * 2]
            output_enc_gpar = output_enc.gather(dim=1, index=gpars.expand(batch, length_dec, enc_dim)) #* mask_gpar
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

        batch, max_len_d, type_space = type_h.size()
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

        # get vector for heads [batch, length_decoder, type_space],
        type_c = type_c.gather(dim=1, index=children.unsqueeze(2).expand(batch, max_len_d, type_space))
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

    def decode(self, input_word, input_char, input_pos, mask=None, beam=1, leading_symbolic=0):
        # reset noise for decoder
        self.decoder.reset_noise(0)

        # output_enc [batch, length, model_dim]
        # arc_c [batch, length, arc_space]
        # type_c [batch, length, type_space]
        # hn [num_direction, batch, hidden_size]
        output_enc, hn = self._get_encoder_output(input_word, input_char, input_pos, mask=mask)
        enc_dim = output_enc.size(2)
        device = output_enc.device
        # output size [batch, length_encoder, arc_space]
        arc_c = self.activation(self.arc_c(output_enc))
        # output size [batch, length_encoder, type_space]
        type_c = self.activation(self.type_c(output_enc))
        type_space = type_c.size(2)
        # [decoder_layers, batch, hidden_size]
        hn = self._transform_decoder_init_state(hn)
        batch, max_len, _ = output_enc.size()

        heads = torch.zeros(batch, 1, max_len, device=device, dtype=torch.int64)
        types = torch.zeros(batch, 1, max_len, device=device, dtype=torch.int64)

        num_steps = 2 * max_len - 1
        stacked_heads = torch.zeros(batch, 1, num_steps + 1, device=device, dtype=torch.int64)
        siblings = torch.zeros(batch, 1, num_steps + 1, device=device, dtype=torch.int64) if self.sibling else None
        hypothesis_scores = output_enc.new_zeros((batch, 1))

        # [batch, beam, length]
        children = torch.arange(max_len, device=device, dtype=torch.int64).view(1, 1, max_len).expand(batch, beam, max_len)
        constraints = torch.zeros(batch, 1, max_len, device=device, dtype=torch.bool)
        constraints[:, :, 0] = True
        # [batch, 1]
        batch_index = torch.arange(batch, device=device, dtype=torch.int64).view(batch, 1)

        # compute lengths
        if mask is None:
            steps = torch.new_tensor([num_steps] * batch, dtype=torch.int64, device=device)
            mask_sent = torch.ones(batch, 1, max_len, dtype=torch.bool, device=device)
        else:
            steps = (mask.sum(dim=1) * 2 - 1).long()
            mask_sent = mask.unsqueeze(1).bool()

        num_hyp = 1
        mask_hyp = torch.ones(batch, 1, device=device)
        hx = hn
        for t in range(num_steps):
            # [batch, num_hyp]
            curr_heads = stacked_heads[:, :, t]
            curr_gpars = heads.gather(dim=2, index=curr_heads.unsqueeze(2)).squeeze(2)
            curr_sibs = siblings[:, :, t] if self.sibling else None
            # [batch, num_hyp, enc_dim]
            src_encoding = output_enc.gather(dim=1, index=curr_heads.unsqueeze(2).expand(batch, num_hyp, enc_dim))

            if self.sibling:
                mask_sib = curr_sibs.gt(0).float().unsqueeze(2)
                output_enc_sibling = output_enc.gather(dim=1, index=curr_sibs.unsqueeze(2).expand(batch, num_hyp, enc_dim)) * mask_sib
                src_encoding = src_encoding + output_enc_sibling

            if self.grandPar:
                output_enc_gpar = output_enc.gather(dim=1, index=curr_gpars.unsqueeze(2).expand(batch, num_hyp, enc_dim))
                src_encoding = src_encoding + output_enc_gpar

            # transform to decoder input
            # [batch, num_hyp, dec_dim]
            src_encoding = self.activation(self.src_dense(src_encoding))

            # output [batch * num_hyp, dec_dim]
            # hx [decoder_layer, batch * num_hyp, dec_dim]
            output_dec, hx = self.decoder.step(src_encoding.view(batch * num_hyp, -1), hx=hx)
            dec_dim = output_dec.size(1)
            # [batch, num_hyp, dec_dim]
            output_dec = output_dec.view(batch, num_hyp, dec_dim)

            # [batch, num_hyp, arc_space]
            arc_h = self.activation(self.arc_h(output_dec))
            # [batch, num_hyp, type_space]
            type_h = self.activation(self.type_h(output_dec))
            # [batch, num_hyp, length]
            out_arc = self.biaffine(arc_h, arc_c, mask_query=mask_hyp, mask_key=mask)
            # mask invalid position to -inf for log_softmax
            if mask is not None:
                minus_mask_enc = mask.eq(0).unsqueeze(1)
                out_arc.masked_fill_(minus_mask_enc, float('-inf'))

            # [batch]
            mask_last = steps.le(t + 1)
            mask_stop = steps.le(t)
            minus_mask_hyp = mask_hyp.eq(0).unsqueeze(2)
            # [batch, num_hyp, length]
            hyp_scores = F.log_softmax(out_arc, dim=2).masked_fill_(mask_stop.view(batch, 1, 1) + minus_mask_hyp, 0)
            # [batch, num_hyp, length]
            hypothesis_scores = hypothesis_scores.unsqueeze(2) + hyp_scores

            # [batch, num_hyp, length]
            mask_leaf = curr_heads.unsqueeze(2).eq(children[:, :num_hyp]) * mask_sent
            mask_non_leaf = (~mask_leaf) * mask_sent

            # apply constrains to select valid hyps
            # [batch, num_hyp, length]
            mask_leaf = mask_leaf * (mask_last.unsqueeze(1) + curr_heads.ne(0)).unsqueeze(2)
            mask_non_leaf = mask_non_leaf * (~constraints)

            hypothesis_scores.masked_fill_(~(mask_non_leaf + mask_leaf), float('-inf'))
            # [batch, num_hyp * length]
            hypothesis_scores, hyp_index = torch.sort(hypothesis_scores.view(batch, -1), dim=1, descending=True)

            # [batch]
            prev_num_hyp = num_hyp
            num_hyps = (mask_leaf + mask_non_leaf).long().view(batch, -1).sum(dim=1)
            num_hyp = num_hyps.max().clamp(max=beam).item()
            # [batch, hum_hyp]
            hyps = torch.arange(num_hyp, device=device, dtype=torch.int64).view(1, num_hyp)
            mask_hyp = hyps.lt(num_hyps.unsqueeze(1)).float()

            # [batch, num_hyp]
            hypothesis_scores = hypothesis_scores[:, :num_hyp]
            hyp_index = hyp_index[:, :num_hyp]
            base_index = hyp_index / max_len
            child_index = hyp_index % max_len

            # [batch, num_hyp]
            hyp_heads = curr_heads.gather(dim=1, index=base_index)
            hyp_gpars = curr_gpars.gather(dim=1, index=base_index)

            # [batch, num_hyp, length]
            base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, max_len)
            constraints = constraints.gather(dim=1, index=base_index_expand)
            constraints.scatter_(2, child_index.unsqueeze(2), True)

            # [batch, num_hyp]
            mask_leaf = hyp_heads.eq(child_index)
            # [batch, num_hyp, length]
            heads = heads.gather(dim=1, index=base_index_expand)
            heads.scatter_(2, child_index.unsqueeze(2), torch.where(mask_leaf, hyp_gpars, hyp_heads).unsqueeze(2))
            types = types.gather(dim=1, index=base_index_expand)
            # [batch, num_hyp]
            org_types = types.gather(dim=2, index=child_index.unsqueeze(2)).squeeze(2)

            # [batch, num_hyp, num_steps]
            base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, num_steps + 1)
            stacked_heads = stacked_heads.gather(dim=1, index=base_index_expand)
            stacked_heads[:, :, t + 1] = torch.where(mask_leaf, hyp_gpars, child_index)
            if self.sibling:
                siblings = siblings.gather(dim=1, index=base_index_expand)
                siblings[:, :, t + 1] = torch.where(mask_leaf, child_index, torch.zeros_like(child_index))

            # [batch, num_hyp, type_space]
            base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, type_space)
            child_index_expand = child_index.unsqueeze(2).expand(batch, num_hyp, type_space)
            # [batch, num_hyp, num_labels]
            out_type = self.bilinear(type_h.gather(dim=1, index=base_index_expand), type_c.gather(dim=1, index=child_index_expand))
            hyp_type_scores = F.log_softmax(out_type, dim=2)
            # compute the prediction of types [batch, num_hyp]
            hyp_type_scores, hyp_types = hyp_type_scores.max(dim=2)
            hypothesis_scores = hypothesis_scores + hyp_type_scores.masked_fill_(mask_stop.view(batch, 1), 0)
            types.scatter_(2, child_index.unsqueeze(2), torch.where(mask_leaf, org_types, hyp_types).unsqueeze(2))

            # hx [decoder_layer, batch * num_hyp, dec_dim]
            # hack to handle LSTM
            hx_index = (base_index + batch_index * prev_num_hyp).view(batch * num_hyp)
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx[:, hx_index]
                cx = cx[:, hx_index]
                hx = (hx, cx)
            else:
                hx = hx[:, hx_index]

        heads = heads[:, 0].cpu().numpy()
        types = types[:, 0].cpu().numpy()
        return heads, types




