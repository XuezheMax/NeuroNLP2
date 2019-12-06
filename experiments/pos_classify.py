import os
import sys
import json
import gc

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import argparse
import math
import signal
import threading
import multiprocessing
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import torch

from neuronlp2.io import get_logger, conllx_data, iterate_data
from neuronlp2.models import DeepBiAffine, NeuroMST, StackPtrNet
from neuronlp2.models import LinearClassifier, MLPClassifier


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        self.children_pids.append(pid)

    def error_listener(self):
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = "\n\n-- Tracebacks above this line can probably be ignored --\n\n"
        msg += original_trace
        raise Exception(msg)


def run_classifier(probe, key, x_train, y_train, x_test, y_test):
    if probe.startswith('svm'):
        clf = SVC(kernel='linear', max_iter=-1)
        if probe == 'svm2':
            clf = OneVsRestClassifier(clf, n_jobs=8)

    elif probe == 'logistic':
        clf = LogisticRegression(max_iter=100, n_jobs=8)
    else:
        raise ValueError('unknown probe: {}'.format(probe))

    print('training: layer: {}, classifier: {}'.format(key, probe))
    start = time.time()
    clf.fit(x_train.numpy(), y_train.numpy())
    acc = clf.score(x_test.numpy(), y_test.numpy()) * 100
    gc.collect()
    print("Accuracy on {} is {:.2f}, time: {:.2f}s".format(key, acc, time.time() - start))


def classify(probe, train_data, train_label, test_data, test_label):
    mp = multiprocessing.get_context('spawn')
    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)
    processes = []
    for key in train_data:
        x_train = train_data[key]
        y_train = train_label
        x_test = test_data[key]
        y_test = test_label

        process = mp.Process(target=run_classifier,
                             args=(probe, key, x_train, y_train, x_test, y_test, error_queue, ),
                             daemon=False)
        process.start()
        error_handler.add_child(process.pid)
        processes.append(process)

    for process in processes:
        process.join()


# def classify(probe, num_labels, train_data, train_label, test_data, test_label, dev_data, dev_label, device):
#     accuracy = dict()
#     stdv = dict()
#     for key in train_data:
#         x_train = train_data[key]
#         y_train = train_label
#         x_test = test_data[key]
#         y_test = test_label
#         x_dev = dev_data[key]
#         y_dev = dev_label
#         start = time.time()
#         print('training: layer: {}, classifier: {}'.format(key, probe))
#         if probe.startswith('svm'):
#             clf = SVC(kernel='linear')
#             if probe == 'svm2':
#                 clf = OneVsRestClassifier(clf, n_jobs=20)
#             clf.fit(x_train.numpy(), y_train.numpy())
#             acc = clf.score(x_test.numpy(), y_test.numpy()) * 100
#             std = 0.
#         elif probe == 'logistic':
#             clf = LogisticRegression(max_iter=200, n_jobs=20)
#             clf.fit(x_train.numpy(), y_train.numpy())
#             acc = clf.score(x_test.numpy(), y_test.numpy()) * 100
#             std = 0.
#         else:
#             accs = []
#             for run in range(5):
#                 if probe == 'linear':
#                     clf = LinearClassifier(x_train.size(1), num_labels)
#                 elif probe == 'mlp':
#                     clf = MLPClassifier(x_train.size(1), num_labels)
#                 else:
#                     raise ValueError('Unknown Classifier: {}'.format(probe))
#                 clf.fit(x_train, y_train, x_val=x_dev, y_val=y_dev, device=device)
#                 with torch.no_grad():
#                     accs.append(clf.score(x_test, y_test, device=device))
#                 print('{}: {:.2f}'.format(run, accs[run]))
#             accs = np.array(accs)
#             acc = accs.mean()
#             std = accs.std()
#         print("Accuracy on {} is {:.2f} ({:.2f}), time: {:.2f}s".format(key, acc, std, time.time() - start))
#         print('-' * 25)
#         accuracy[key] = acc
#         stdv[key] = std
#         torch.cuda.empty_cache()
#         gc.collect()
#     return accuracy, stdv


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

    return network, (data_train, data_dev, data_test), num_pos - conllx_data.NUM_SYMBOLIC_TAGS, device


def encode(network, data, device, bucketed):
    words = []
    chars = []
    pos = None
    labels = []
    fake_labels = []
    rnn_layers = None
    network.eval()
    for batch in iterate_data(data, batch_size=256, bucketed=bucketed, unk_replace=0., shuffle=False):
        wids = batch['WORD'].to(device)
        chids = batch['CHAR'].to(device)
        postags = batch['POS'].to(device)
        fake_postags = batch['FAKEPOS'].to(device)
        masks = batch['MASK'].to(device)
        nbatch = wids.size(0)
        with torch.no_grad():
            out = network.get_layer_outputs(wids, chids, postags, mask=masks)
        word_embed = out['word'].cpu()
        char_embed = out['char'].cpu()
        pos_embed = out['pos']
        if pos_embed is not None:
            pos_embed = pos_embed.cpu()
        layer_outs = [layer_out.cpu() for layer_out in out['layers']]
        num_layers = len(layer_outs)

        if rnn_layers is None:
            rnn_layers = [[] for _ in range(num_layers)]
        else:
            assert len(rnn_layers) == num_layers

        if pos_embed is not None and pos is None:
            pos = []
        lengths = masks.sum(dim=1).long().cpu()
        postags = (postags - conllx_data.NUM_SYMBOLIC_TAGS).cpu()
        fake_postags = (fake_postags - conllx_data.NUM_SYMBOLIC_TAGS).cpu()
        for b in range(nbatch):
            length = lengths[b]
            words.append(word_embed[b, 1:length])
            chars.append(char_embed[b, 1:length])
            if pos_embed is not None:
                pos.append(pos_embed[b, 1:length])
            labels.append(postags[b, 1:length])
            fake_labels.append(fake_postags[b, 1:length])
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
    fake_labels = torch.cat(fake_labels, dim=0)
    assert ntokens == fake_labels.size(0)

    rnn_layers = [torch.cat(rnn_layer, dim=0) for rnn_layer in rnn_layers]
    for rnn_layer in rnn_layers:
        assert ntokens == rnn_layer.size(0)
    print('number of tokens: %d' % ntokens)

    features = {"word": words, "char": chars, "word+char": torch.cat([words, chars], dim=1)}
    if pos is not None:
        features['pos'] = pos
    features.update({'rnn layer{}'.format(i): rnn_layer for i, rnn_layer in enumerate(rnn_layers)})
    torch.cuda.empty_cache()
    gc.collect()
    return features, labels, fake_labels


def main(args):
    network, (data_train, data_dev, data_test), num_labels, device = setup(args)

    train_features, train_labels, train_fake_labels = encode(network, data_train, device, bucketed=True)
    dev_features, dev_labels, dev_fake_labels = encode(network, data_dev, device, bucketed=False)
    test_features, test_labels, test_fake_labels = encode(network, data_test, device, bucketed=False)

    layer = args.layer
    print("Real POS")
    if layer is None:
        # classify(args.probe, num_labels, train_features, train_labels, test_features, test_labels, dev_features, dev_labels, device)
        classify(args.probe, train_features, train_labels, test_features, test_labels)

        if 'pos' in train_features:
            train_features.pop('pos')
            dev_features.pop('pos')
            test_features.pop('pos')
        print('Fake POS')
        classify(args.probe, train_features, train_fake_labels, test_features, test_fake_labels)
    else:
        x_train = train_features[layer]
        y_train = train_labels
        x_test = test_features[layer]
        y_test = test_labels
        run_classifier(args.probe, layer, x_train, y_train, x_test, y_test)
        if layer != 'pos':
            y_train = train_fake_labels
            y_test = test_fake_labels
            print('Fake POS')
            run_classifier(args.probe, layer, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description='POS tag classification')
    args_parser.add_argument('--probe', choices=['svm1', 'svm2', 'logistic', 'linear', 'mlp'], required=True, help='classifier for probe')
    args_parser.add_argument('--train', help='path for training file.')
    args_parser.add_argument('--dev', help='path for dev file.')
    args_parser.add_argument('--test', help='path for test file.', required=True)
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args_parser.add_argument('--layer', help='layer of model to classify.', default=None)
    args = args_parser.parse_args()
    main(args)

