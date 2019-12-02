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
from sklearn.svm import SVC
import torch

from neuronlp2.io import get_logger, conllx_data, iterate_data
from neuronlp2.models import DeepBiAffine, NeuroMST, StackPtrNet


def svm(train_data, train_label, test_data, test_label, kernel='linear'):
    accs = dict()
    for key in train_data:
        x_train = train_data[key]
        y_train = train_label
        x_test = test_data[key]
        y_test = test_label
        clf = SVC(kernel=kernel)
        clf.fit(x_train, y_train)
        acc = clf.score(x_test, y_test)
        print("Accuracy on {} is {}".format(key, acc))
        accs[key] = acc
    return accs


def setup(args):
    logger = get_logger('POS')
    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')

    model_path = args.model_path
    model_name = os.path.join(model_path, 'model.pt')
    print(args)

    logger.info("Loading Alphabets")
    alphabet_path = os.path.join(model_path, 'alphabets')
    assert os.path.exists(alphabet_path)
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_data.create_alphabets(alphabet_path, None)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    logger.info("loading network...")
    hyps = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
    model_type = hyps['model']
    assert model_type in ['DeepBiAffine', 'NeuroMST', 'StackPtr']
    word_dim = hyps['word_dim']
    char_dim = hyps['char_dim']
    use_pos = hyps['pos']
    pos_dim = hyps['pos_dim']
    mode = hyps['rnn_mode']
    hidden_size = hyps['hidden_size']
    arc_space = hyps['arc_space']
    type_space = hyps['type_space']
    p_in = hyps['p_in']
    p_out = hyps['p_out']
    p_rnn = hyps['p_rnn']
    activation = hyps['activation']
    prior_order = None

    if model_type == 'DeepBiAffine':
        num_layers = hyps['num_layers']
        network = DeepBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                               mode, hidden_size, num_layers, num_types, arc_space, type_space,
                               p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=use_pos, activation=activation)
    elif model_type == 'NeuroMST':
        num_layers = hyps['num_layers']
        network = NeuroMST(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                           mode, hidden_size, num_layers, num_types, arc_space, type_space,
                           p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=use_pos, activation=activation)
    elif model_type == 'StackPtr':
        encoder_layers = hyps['encoder_layers']
        decoder_layers = hyps['decoder_layers']
        num_layers = (encoder_layers, decoder_layers)
        prior_order = hyps['prior_order']
        grandPar = hyps['grandPar']
        sibling = hyps['sibling']
        network = StackPtrNet(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                              mode, hidden_size, encoder_layers, decoder_layers, num_types, arc_space, type_space,
                              prior_order=prior_order, activation=activation, p_in=p_in, p_out=p_out, p_rnn=p_rnn,
                              pos=use_pos, grandPar=grandPar, sibling=sibling)
    else:
        raise RuntimeError('Unknown model type: %s' % model_type)

    network = network.to(device)
    network.load_state_dict(torch.load(model_name, map_location=device))
    model = "{}-{}".format(model_type, mode)
    logger.info("Network: %s, num_layer=%s, hidden=%d, act=%s" % (model, num_layers, hidden_size, activation))

    logger.info("Reading Data")
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    data_train = conllx_data.read_bucketed_data(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True)
    data_dev = conllx_data.read_data(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True)
    data_test = conllx_data.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True)

    return network, (data_train, data_dev, data_test), device


def encode(network, data, device, bucketed):
    words = []
    chars = []
    pos = None
    labels = []
    rnn_layers = None
    network.eval()
    for batch in iterate_data(data, batch_size=256, bucketed=bucketed, unk_replace=0., shuffle=False):
        wids = batch['WORD'].to(device)
        chids = batch['CHAR'].to(device)
        postags = batch['POS'].to(device)
        masks = batch['MASK'].to(device)
        nbatch = words.size(0)
        with torch.no_grad():
            out = network.get_layer_outputs(wids, chids, postags, mask=masks)
        word_embed = out['word']
        char_embed = out['char']
        pos_embed = out['pos']
        layer_outs = out['layers']
        num_layers = len(layer_outs)

        if rnn_layers is None:
            rnn_layers = [[] for _ in range(num_layers)]
        else:
            assert len(rnn_layers) == num_layers

        if pos_embed is not None and pos is None:
            pos = []
        lengths = masks.sum(dim=1).long()
        postags = postags - conllx_data.NUM_SYMBOLIC_TAGS
        for b in range(nbatch):
            length = lengths[b]
            words.append(word_embed[b, 1:length])
            chars.append(char_embed[b, 1:length])
            if pos_embed is not None:
                pos.append(pos_embed[b, 1:length])
            labels.append(postags[b, 1:length])
            for rnn_layer, layer_out in zip(rnn_layers, layer_outs):
                rnn_layer.append(layer_out[b, 1:length])

    words = torch.cat(words, dim=0)
    ntokens = words.size(0)

    chars = torch.cat(chars, dim=0)
    assert ntokens == chars.size(0)

    if pos is not None:
        pos = torch.cat(pos, dim=0)
        assert ntokens == pos.size(0)

    labels = torch.cat(labels, dim=0)
    assert ntokens == labels.size(0)

    rnn_layers = [torch.cat(rnn_layer, dim=0) for rnn_layer in rnn_layers]
    for rnn_layer in rnn_layers:
        assert ntokens == rnn_layer.size(0)
    print('number of tokens: %d' % words.size(0))

    features = {"word": words, "char": chars, "pos": pos}
    features.update({'rnn layer{}'.format(i): rnn_layer for i, rnn_layer in enumerate(rnn_layers)})
    return features, labels


def main(args):
    network, (data_train, data_dev, data_test), device = setup(args)

    train_features, train_labels = encode(network, data_train, device, bucketed=True)
    dev_features, dev_labels = encode(network, data_dev, device, bucketed=False)
    test_features, test_labels = encode(network, data_test, device, bucketed=False)

    svm(train_features, train_labels, test_features, test_labels, kernel='linear')


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description='POS tag classification')
    args_parser.add_argument('--probe', choices=['svm',], required=True, help='classifier for probe')
    args_parser.add_argument('--train', help='path for training file.')
    args_parser.add_argument('--dev', help='path for dev file.')
    args_parser.add_argument('--test', help='path for test file.', required=True)
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args = args_parser.parse_args()
    main(args)

