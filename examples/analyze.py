from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-TreeCRF model for Graph-based dependency parsing.
"""

import sys
import os

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm
from torch.optim import Adam, SGD, Adadelta
from neuronlp2.io import get_logger, conllx_stacked_data
from neuronlp2.models import StackPtrNet
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser

uid = uuid.uuid4().get_hex()[:6]

def main():
    args_parser = argparse.ArgumentParser(description='Tuning with stack pointer parser')
    args_parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args_parser.add_argument('--model_name', help='name for saving model file.', required=True)
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--beam', type=int, default=1, help='Beam size for decoding')
    args_parser.add_argument('--gpu', type=bool, default=False, help='Using GPU')
    args_parser.add_argument('--left2right', action='store_true', help='apply left to right prior order.')

    args = args_parser.parse_args()

    logger = get_logger("Analyzer")

    test_path = args.test
    model_path = args.model_path
    model_name = args.model_name

    alphabet_path = os.path.join(model_path, 'alphabets/')
    model_name = os.path.join(model_path, model_name)
    word_alphabet, char_alphabet, pos_alphabet, \
    type_alphabet = conllx_stacked_data.create_alphabets(alphabet_path, None,
                                                         data_paths=[None, None],
                                                         max_vocabulary_size=50000, embedd_dict=None)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    use_gpu = args.gpu
    left2right = args.left2right
    beam = args.beam

    logger.info('use gpu: %s' % use_gpu)

    data_test = conllx_stacked_data.read_stacked_data_to_variable(test_path, word_alphabet, char_alphabet,
                                                                  pos_alphabet, type_alphabet,
                                                                  use_gpu=use_gpu, volatile=True,
                                                                  left2right=left2right)

    punct_set = None
    punctuation = args.punctuation
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)

    pred_writer.start('tmp/analyze_pred')
    gold_writer.start('tmp/analyze_gold')

    network = torch.load(model_name)

    if use_gpu:
        network.cuda()
    else:
        network.cpu()

    test_ucorrect = 0.0
    test_lcorrect = 0.0
    test_ucomlpete_match = 0.0
    test_lcomplete_match = 0.0
    test_total = 0

    test_ucorrect_nopunc = 0.0
    test_lcorrect_nopunc = 0.0
    test_ucomlpete_match_nopunc = 0.0
    test_lcomplete_match_nopunc = 0.0
    test_total_nopunc = 0
    test_total_inst = 0

    test_root_correct = 0.0
    test_total_root = 0

    test_ucorrect_stack_leaf = 0.0
    test_ucorrect_stack_non_leaf = 0.0

    test_lcorrect_stack_leaf = 0.0
    test_lcorrect_stack_non_leaf = 0.0

    test_leaf = 0
    test_non_leaf = 0

    sent = 0
    network.eval()
    for batch in conllx_stacked_data.iterate_batch_stacked_variable(data_test, 1):
        sys.stdout.write('%d, ' % sent)
        sys.stdout.flush()
        sent += 1

        input_encoder, input_decoder = batch
        word, char, pos, heads, types, masks, lengths = input_encoder
        _, children, stacked_types, mask_d, lengths_d = input_decoder
        heads_pred, types_pred, children_pred, stacked_types_pred = network.decode(word, char, pos,
                                                                                   mask=masks, length=lengths, beam=beam)

        children = children.data
        stacked_types = stacked_types.data
        children_pred = torch.from_numpy(children_pred)
        stacked_types_pred = torch.from_numpy(stacked_types_pred)
        mask_d = mask_d.data
        mask_leaf = torch.eq(children, 0).float()
        mask_non_leaf = (1.0 - mask_leaf)
        mask_leaf = mask_leaf * mask_d
        mask_non_leaf = mask_non_leaf * mask_d
        num_leaf = mask_leaf.sum()
        num_non_leaf = mask_non_leaf.sum()

        ucorr_stack = torch.eq(children_pred, children).float()
        lcorr_stack = ucorr_stack * torch.eq(stacked_types_pred, stacked_types).float()
        ucorr_stack_leaf = (ucorr_stack * mask_leaf).sum()
        ucorr_stack_non_leaf = (ucorr_stack * mask_non_leaf).sum()

        lcorr_stack_leaf = (lcorr_stack * mask_leaf).sum()
        lcorr_stack_non_leaf = (lcorr_stack * mask_non_leaf).sum()

        test_ucorrect_stack_leaf += ucorr_stack_leaf
        test_ucorrect_stack_non_leaf += ucorr_stack_non_leaf
        test_lcorrect_stack_leaf += lcorr_stack_leaf
        test_lcorrect_stack_non_leaf += lcorr_stack_non_leaf

        test_leaf += num_leaf
        test_non_leaf += num_non_leaf

        # ------------------------------------------------------------------------------------------------

        word = word.data.cpu().numpy()
        pos = pos.data.cpu().numpy()
        lengths = lengths.cpu().numpy()
        heads = heads.data.cpu().numpy()
        types = types.data.cpu().numpy()

        pred_writer.write(word, pos, heads_pred, types_pred, lengths, symbolic_root=True)
        gold_writer.write(word, pos, heads, types, lengths, symbolic_root=True)

        stats, stats_nopunc, stats_root, num_inst = parser.eval(word, pos, heads_pred, types_pred, heads, types,
                                                                word_alphabet, pos_alphabet, lengths,
                                                                punct_set=punct_set, symbolic_root=True)
        ucorr, lcorr, total, ucm, lcm = stats
        ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
        corr_root, total_root = stats_root

        test_ucorrect += ucorr
        test_lcorrect += lcorr
        test_total += total
        test_ucomlpete_match += ucm
        test_lcomplete_match += lcm

        test_ucorrect_nopunc += ucorr_nopunc
        test_lcorrect_nopunc += lcorr_nopunc
        test_total_nopunc += total_nopunc
        test_ucomlpete_match_nopunc += ucm_nopunc
        test_lcomplete_match_nopunc += lcm_nopunc

        test_root_correct += corr_root
        test_total_root += total_root

        test_total_inst += num_inst

    pred_writer.close()
    gold_writer.close()

    print('test W. Punct:  ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        test_ucorrect, test_lcorrect, test_total, test_ucorrect * 100 / test_total, test_lcorrect * 100 / test_total,
        test_ucomlpete_match * 100 / test_total_inst, test_lcomplete_match * 100 / test_total_inst))
    print('test Wo Punct:  ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        test_ucorrect_nopunc, test_lcorrect_nopunc, test_total_nopunc,
        test_ucorrect_nopunc * 100 / test_total_nopunc, test_lcorrect_nopunc * 100 / test_total_nopunc,
        test_ucomlpete_match_nopunc * 100 / test_total_inst, test_lcomplete_match_nopunc * 100 / test_total_inst))
    print('test Root: corr: %d, total: %d, acc: %.2f%%' % (
        test_root_correct, test_total_root, test_root_correct * 100 / test_total_root))
    print('============================================================================================================================')

    print('Stack leaf:     ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
        test_ucorrect_stack_leaf, test_lcorrect_stack_leaf, test_leaf,
        test_ucorrect_stack_leaf * 100 / test_leaf, test_lcorrect_stack_leaf * 100 / test_leaf))
    print('Stack non_leaf: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
        test_ucorrect_stack_non_leaf, test_lcorrect_stack_non_leaf, test_non_leaf,
        test_ucorrect_stack_non_leaf * 100 / test_non_leaf, test_lcorrect_stack_non_leaf * 100 / test_non_leaf))
    print('============================================================================================================================')


if __name__ == '__main__':
    main()
