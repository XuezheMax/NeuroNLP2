from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-TreeCRF model for Graph-based dependency parsing.
"""

import sys

sys.path.append(".")
sys.path.append("..")

import time
import argparse

import numpy as np
import torch
from torch.optim import Adam, SGD
from neuronlp2.io import get_logger, conllx_data
from neuronlp2.models import BiRecurrentConvTreeCRF, BiRecurrentConvBiAffine, BiVarRecurrentConvBiAffine
from neuronlp2 import utils


def main():
    parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU'], help='architecture of rnn', required=True)
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of sentences in each batch')
    parser.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in RNN')
    parser.add_argument('--tag_space', type=int, default=128, help='Dimension of tag space')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
    parser.add_argument('--num_filters', type=int, default=50, help='Number of filters in CNN')
    parser.add_argument('--objective', choices=['cross_entropy', 'crf'], default='cross_entropy',
                        help='objective function of training procedure.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.05, help='Decay rate of learning rate')
    parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    parser.add_argument('--dropout', choices=['std', 'variational'], help='type of dropout', required=True)
    parser.add_argument('--p', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--biaffine', action='store_true', help='bi-gram parameter for CRF')
    parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    parser.add_argument('--word_embedding', choices=['glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words',
                        required=True)
    parser.add_argument('--word_path', help='path for word embedding dict')
    parser.add_argument('--char_embedding', choices=['random', 'polyglot'], help='Embedding for characters',
                        required=True)
    parser.add_argument('--char_path', help='path for character embedding dict')
    parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"

    args = parser.parse_args()

    logger = get_logger("GraphParser")

    mode = args.mode
    obj = args.objective
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    tag_space = args.tag_space
    num_layers = args.num_layers
    num_filters = args.num_filters
    learning_rate = args.learning_rate
    momentum = 0.9
    decay_rate = args.decay_rate
    gamma = args.gamma
    schedule = args.schedule
    p = args.p
    biaffine = args.biaffine
    punctuation = args.punctuation

    word_embedding = args.word_embedding
    word_path = args.word_path
    char_embedding = args.char_embedding
    char_path = args.char_path

    pos_dim = 50
    word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path)
    char_dict = None
    char_dim = 50
    if char_embedding != 'random':
        char_dict, char_dim = utils.load_embedding_dict(char_embedding, char_path)
    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, pos_alphabet, \
    type_alphabet = conllx_data.create_alphabets("data/alphabets/mst/", train_path,
                                                 data_paths=[dev_path, test_path],
                                                 max_vocabulary_size=50000, embedd_dict=word_dict)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    logger.info("Reading Data")
    use_gpu = torch.cuda.is_available()

    data_train = conllx_data.read_data_to_variable(train_path, word_alphabet, char_alphabet, pos_alphabet,
                                                   type_alphabet, use_gpu=use_gpu, symbolic_root=True)
    # data_train = conllx_data.read_data(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    # num_data = sum([len(bucket) for bucket in data_train])
    num_data = sum(data_train[1])

    data_dev = conllx_data.read_data_to_variable(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                 use_gpu=use_gpu, volatile=True, symbolic_root=True)
    data_test = conllx_data.read_data_to_variable(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                  use_gpu=use_gpu, volatile=True, symbolic_root=True)

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / word_dim)
        table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in word_dict:
                embedding = word_dict[word]
            elif word.lower() in word_dict:
                embedding = word_dict[word.lower()]
            else:
                embedding = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('word OOV: %d' % oov)
        return torch.from_numpy(table)

    def construct_char_embedding_table():
        if char_dict is None:
            return None

        scale = np.sqrt(3.0 / char_dim)
        table = np.empty([num_chars, char_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
        oov = 0
        for char, index, in char_alphabet.items():
            if char in char_dict:
                embedding = char_dict[char]
            else:
                embedding = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('character OOV: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()
    char_table = construct_char_embedding_table()

    window = 3
    if obj == 'cross_entropy':
        if args.dropout == 'std':
            network = BiRecurrentConvBiAffine(word_dim, num_words,
                                              char_dim, num_chars,
                                              pos_dim, num_pos,
                                              num_filters, window,
                                              mode, hidden_size, num_layers, num_types, tag_space,
                                              embedd_word=word_table, embedd_char=char_table,
                                              p_rnn=p, biaffine=biaffine)
        else:
            network = BiVarRecurrentConvBiAffine(word_dim, num_words,
                                                 char_dim, num_chars,
                                                 pos_dim, num_pos,
                                                 num_filters, window,
                                                 mode, hidden_size, num_layers, num_types, tag_space,
                                                 embedd_word=word_table, embedd_char=char_table,
                                                 p_rnn=p, biaffine=biaffine)
    elif obj == 'crf':
        if args.dropout == 'std':
            raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise RuntimeError('Unknown objective: %s' % obj)

    if use_gpu:
        network.cuda()

    adam_epochs = 10
    adam_rate = 0.001
    if adam_epochs > 0:
        lr = adam_rate
        optim = Adam(network.parameters(), lr=adam_rate, betas=(0.9, 0.9), weight_decay=gamma)
    else:
        lr = learning_rate
        optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
    logger.info("Network: %s, num_layer=%d, hidden=%d, filter=%d, tag_space=%d, crf=%s" % (
        mode, num_layers, hidden_size, num_filters, tag_space, 'biaffine' if biaffine else 'affine'))
    logger.info("training: obj: %s, l2: %f, (#training data: %d, batch: %d, dropout: %.2f)" % (
        obj, gamma, num_data, batch_size, p))

    num_batches = num_data / batch_size + 1
    dev_ucorrect = 0.0
    dev_lcorrect = 0.0
    dev_ucorrect_nopunct = 0.0
    dev_lcorrect_nopunct = 0.0
    best_epoch = 0
    test_ucorrect = 0.0
    test_lcorrect = 0.0
    test_ucorrect_nopunct = 0.0
    test_lcorrect_nopunct = 0.0
    test_total = 0
    test_total_nopunc = 0

    for epoch in range(1, num_epochs + 1):
        print('Epoch %d (%s(%s), learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (
            epoch, mode, args.dropout, lr, decay_rate, schedule))
        train_err = 0.
        train_total = 0.

        start_time = time.time()
        num_back = 0
        network.train()
        for batch in range(1, num_batches + 1):
            word, char, pos, heads, types, masks, lengths = conllx_data.get_batch_variable(data_train, batch_size)

            optim.zero_grad()
            loss = network.loss(word, char, pos, heads, types, mask=masks, length=lengths)
            loss.backward()
            optim.step()

            num_inst = word.size(0) if obj == 'crf' else masks.data.sum() - word.size(0)
            train_err += loss.data[0] * num_inst
            train_total += num_inst

            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            # update log
            if batch % 1 == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                log_info = 'train: %d/%d loss: %.4f, time left (estimated): %.2fs' % (
                    batch, num_batches, train_err / train_total, time_left)
                sys.stdout.write(log_info)
                num_back = len(log_info)

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        print('train: %d loss: %.4f, time: %.2fs' % (num_batches, train_err / train_total, time.time() - start_time))

        if epoch % schedule == 0:
            # lr = lr * decay_rate
            if epoch < adam_epochs:
                lr = adam_rate / (1.0 + epoch * decay_rate)
                optim = Adam(network.parameters(), lr=lr, betas=(0.9, 0.9), weight_decay=gamma)
            else:
                lr = learning_rate / (1.0 + (epoch - adam_epochs) * decay_rate)
                optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)


if __name__ == '__main__':
    main()
