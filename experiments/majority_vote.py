import os
import sys
import json

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import argparse
import math
import numpy as np

from neuronlp2.io import get_logger, conllx_data
from neuronlp2.io.common import DIGIT_RE


def find_best_pos(vote):
    bc = 0
    bpos = ''
    for pos, c in vote.items():
        if c > bc:
            bpos = pos
            bc = c
    return bpos


def conll_pos(train_path, dev_path, test_path, word_alphabet, lowercase, fake=False):
    votes = dict()
    num_tokens = 0
    mtf = dict()
    with open(train_path, 'r') as ifile:
        for line in ifile:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split('\t')
            word = DIGIT_RE.sub("0", tokens[1])
            if lowercase:
                word = word.lower()
            pos = tokens[3] if fake else tokens[4]

            if word_alphabet is not None:
                word = word_alphabet.get_index(word)

            if word in votes:
                vote = votes[word]
                if pos in vote:
                    vote[pos] += 1
                else:
                    vote[pos] = 1
            else:
                vote = dict()
                vote[pos] = 1
                votes[word] = vote
            num_tokens += 1
            if pos in mtf:
                mtf[pos] += 1
            else:
                mtf[pos] = 1
    print('number tokens in traning data: %d' % num_tokens)
    print('number distinguished tokens: %d' % len(votes))
    mtf = find_best_pos(mtf)
    votes = {word: find_best_pos(vote) for word, vote in votes.items()}

    corr = 0
    total = 0
    oov = 0
    with open(test_path, 'r') as ifile:
        for line in ifile:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split('\t')
            word = DIGIT_RE.sub("0", tokens[1])
            if lowercase:
                word = word.lower()
            gold = tokens[3] if fake else tokens[4]

            if word_alphabet is not None:
                word = word_alphabet.get_index(word)

            if word in votes:
                pred = votes[word]
            else:
                pred = mtf
                oov += 1
            corr += int(pred == gold)
            total += 1
            # print(word, gold, pred)
    print('total: %d, correct: %d, acc: %.2f, oov: %d' % (total, corr, float(corr) / total * 100, oov))


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description='Majority vote')
    args_parser.add_argument('--train', help='path for training file.')
    args_parser.add_argument('--dev', help='path for dev file.')
    args_parser.add_argument('--test', help='path for test file.', required=True)
    args_parser.add_argument('--model_path', help='path for saving model file.', default=None)
    args_parser.add_argument('--lowercase', action='store_true', help='lowercase all the tokens.')
    args = args_parser.parse_args()

    logger = get_logger('POS')

    model_path = args.model_path
    print(args)

    if model_path is not None:
        logger.info("Loading Alphabets")
        alphabet_path = os.path.join(model_path, 'alphabets')
        word_alphabet, _, _, _ = conllx_data.create_alphabets(alphabet_path, None)
    else:
        word_alphabet = None

    train_path = args.train
    dev_path = args.dev
    test_path = args.test

    # conll_pos(train_path, dev_path, test_path, word_alphabet, lowercase=args.lowercase, fake=False)
    conll_pos(train_path, dev_path, test_path, word_alphabet, lowercase=args.lowercase, fake=True)

