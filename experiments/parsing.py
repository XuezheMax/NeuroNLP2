"""
Implementation of Graph-based dependency parsing.
"""

import os
import sys
import gc
import json

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import argparse

import numpy as np
import torch
from torch.optim.adamw import AdamW
from torch.optim import SGD
from torch.nn.utils import clip_grad_norm_
from neuronlp2.io import get_logger, conllx_data, iterate_data
from neuronlp2.models import DeepBiAffine, NeuroMST
from neuronlp2.optim import ExponentialScheduler
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser
from neuronlp2.nn.utils import freeze_embedding


def get_optimizer(parameters, optim, learning_rate, lr_decay, betas, eps, amsgrad, weight_decay, warmup_steps):
    if optim == 'sgd':
        optimizer = SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    else:
        optimizer = AdamW(parameters, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad, weight_decay=weight_decay)
    init_lr = 1e-7
    scheduler = ExponentialScheduler(optimizer, lr_decay, warmup_steps, init_lr)
    return optimizer, scheduler


def eval(data, network, pred_writer, gold_writer, punct_set, word_alphabet, pos_alphabet, device):
    network.eval()
    accum_ucorr = 0.0
    accum_lcorr = 0.0
    accum_total = 0
    accum_ucomlpete = 0.0
    accum_lcomplete = 0.0
    accum_ucorr_nopunc = 0.0
    accum_lcorr_nopunc = 0.0
    accum_total_nopunc = 0
    accum_ucomlpete_nopunc = 0.0
    accum_lcomplete_nopunc = 0.0
    accum_root_corr = 0.0
    accum_total_root = 0.0
    accum_total_inst = 0.0
    for data in iterate_data(data, 256):
        words = data['WORD'].to(device)
        chars = data['CHAR'].to(device)
        postags = data['POS'].to(device)
        heads = data['HEAD'].numpy()
        types = data['TYPE'].numpy()
        masks = data['MASK'].to(device)
        lengths = data['LENGTH'].numpy()
        heads_pred, types_pred = network.decode(words, chars, postags, mask=masks, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)

        words = words.cpu().numpy()
        postags = postags.cpu().numpy()
        pred_writer.write(words, postags, heads_pred, types_pred, lengths, symbolic_root=True)
        gold_writer.write(words, postags, heads, types, lengths, symbolic_root=True)

        stats, stats_nopunc, stats_root, num_inst = parser.eval(words, postags, heads_pred, types_pred, heads, types,
                                                                word_alphabet, pos_alphabet, lengths, punct_set=punct_set, symbolic_root=True)
        ucorr, lcorr, total, ucm, lcm = stats
        ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
        corr_root, total_root = stats_root

        accum_ucorr += ucorr
        accum_lcorr += lcorr
        accum_total += total
        accum_ucomlpete += ucm
        accum_lcomplete += lcm

        accum_ucorr_nopunc += ucorr_nopunc
        accum_lcorr_nopunc += lcorr_nopunc
        accum_total_nopunc += total_nopunc
        accum_ucomlpete_nopunc += ucm_nopunc
        accum_lcomplete_nopunc += lcm_nopunc

        accum_root_corr += corr_root
        accum_total_root += total_root

        accum_total_inst += num_inst

    print('W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        accum_ucorr, accum_lcorr, accum_total, accum_ucorr * 100 / accum_total, accum_lcorr * 100 / accum_total,
        accum_ucomlpete * 100 / accum_total_inst, accum_lcomplete * 100 / accum_total_inst))
    print('Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        accum_ucorr_nopunc, accum_lcorr_nopunc, accum_total_nopunc, accum_ucorr_nopunc * 100 / accum_total_nopunc,
        accum_lcorr_nopunc * 100 / accum_total_nopunc,
        accum_ucomlpete_nopunc * 100 / accum_total_inst, accum_lcomplete_nopunc * 100 / accum_total_inst))
    print('Root: corr: %d, total: %d, acc: %.2f%%' %(accum_root_corr, accum_total_root, accum_root_corr * 100 / accum_total_root))
    return (accum_ucorr, accum_lcorr, accum_ucomlpete, accum_lcomplete, accum_total), \
           (accum_ucorr_nopunc, accum_lcorr_nopunc, accum_ucomlpete_nopunc, accum_lcomplete_nopunc, accum_total_nopunc), \
           (accum_root_corr, accum_total_root, accum_total_inst)


def main():
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--config', type=str, help='config file', required=True)
    args_parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    args_parser.add_argument('--loss_type', choices=['sentence', 'token'], default='sentence', help='loss type (default: sentence)')
    args_parser.add_argument('--optim', choices=['sgd', 'adam'], help='type of optimizer', required=True)
    args_parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    args_parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam')
    args_parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam')
    args_parser.add_argument('--eps', type=float, default=1e-8, help='epsilon for adam or adamax')
    args_parser.add_argument('--lr_decay', type=float, default=0.999995, help='Decay rate of learning rate')
    args_parser.add_argument('--amsgrad', action='store_true', help='AMS Grad')
    args_parser.add_argument('--grad_clip', type=float, default=0, help='max norm for gradient clip (default 0: no clip')
    args_parser.add_argument('--warmup_steps', type=int, default=0, metavar='N', help='number of steps to warm up (default: 0)')
    args_parser.add_argument('--reset', type=int, default=10, help='Number of epochs to reset optimizer (default 10)')
    args_parser.add_argument('--weight_decay', type=float, default=0.0, help='weight for l2 norm decay')
    args_parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    args_parser.add_argument('--freeze', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--word_embedding', choices=['glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words', required=True)
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--char_embedding', choices=['random', 'polyglot'], help='Embedding for characters', required=True)
    args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--train', help='path for training file.', required=True)
    args_parser.add_argument('--dev', help='path for dev file.', required=True)
    args_parser.add_argument('--test', help='path for test file.', required=True)
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)

    args = args_parser.parse_args()

    logger = get_logger("Parsing")

    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    train_path = args.train
    dev_path = args.dev
    test_path = args.test

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    optim = args.optim
    learning_rate = args.learning_rate
    lr_decay = args.lr_decay
    amsgrad = args.amsgrad
    eps = args.eps
    betas = (args.beta1, args.beta2)
    warmup_steps = args.warmup_steps
    weight_decay = args.weight_decay
    grad_clip = args.grad_clip

    loss_ty_token = args.loss_type == 'token'
    unk_replace = args.unk_replace
    freeze = args.freeze

    model_path = args.model_path
    model_name = os.path.join(model_path, 'model.pt')
    punctuation = args.punctuation

    word_embedding = args.word_embedding
    word_path = args.word_path
    char_embedding = args.char_embedding
    char_path = args.char_path

    print(args)

    word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path)
    char_dict = None
    if char_embedding != 'random':
        char_dict, char_dim = utils.load_embedding_dict(char_embedding, char_path)
    else:
        char_dict = None
        char_dim = None

    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(model_path, 'alphabets')
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_data.create_alphabets(alphabet_path, train_path,
                                                                                             data_paths=[dev_path, test_path],
                                                                                             embedd_dict=word_dict, max_vocabulary_size=100000)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    logger.info("Reading Data")

    data_train = conllx_data.read_bucketed_data(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True)
    num_data = sum(data_train[1])

    data_dev = conllx_data.read_data(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True)
    data_test = conllx_data.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True)

    result_path = os.path.join(model_path, 'tmp')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / word_dim)
        table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in word_dict:
                embedding = word_dict[word]
            elif word.lower() in word_dict:
                embedding = word_dict[word.lower()]
            else:
                embedding = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
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

    logger.info("constructing network...")

    hyps = json.load(open(args.config, 'r'))
    json.dump(hyps, open(os.path.join(model_path, 'config.json'), 'w'), indent=2)
    model_type = hyps['model']
    assert model_type in ['DeepBiAffine', 'NeuroMST']
    assert word_dim == hyps['word_dim']
    if char_dim is not None:
        assert char_dim == hyps['char_dim']
    else:
        char_dim = hyps['char_dim']
    use_pos = hyps['pos']
    pos_dim = hyps['pos_dim']
    mode = hyps['rnn_mode']
    hidden_size = hyps['hidden_size']
    arc_space = hyps['arc_space']
    type_space = hyps['type_space']
    num_layers = hyps['num_layers']
    p_in = hyps['p_in']
    p_out = hyps['p_out']
    p_rnn = hyps['p_rnn']
    activation = hyps['activation']

    if model_type == 'DeepBiAffine':
        network = DeepBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                               mode, hidden_size, num_layers, num_types, arc_space, type_space,
                               embedd_word=word_table, embedd_char=char_table,
                               p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=use_pos, activation=activation)
    elif model_type == 'NeuroMST':
        network = NeuroMST(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                           mode, hidden_size, num_layers, num_types, arc_space, type_space,
                           embedd_word=word_table, embedd_char=char_table,
                           p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=use_pos, activation=activation)
    else:
        raise RuntimeError('Unknown model type: %s' % model_type)

    if freeze:
        freeze_embedding(network.word_embedd)

    network = network.to(device)

    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)

    optimizer, scheduler = get_optimizer(network.parameters(), optim, learning_rate, lr_decay, betas, eps, amsgrad, weight_decay, warmup_steps)
    model = "{}-{}".format(model_type, mode)
    logger.info("Network: %s, num_layer=%d, hidden=%d, act=%s" % (model, num_layers, hidden_size, activation))
    logger.info("training: l2: %f, (#training data: %d, batch: %d, unk replace: %.2f)" % (weight_decay, num_data, batch_size, unk_replace))
    logger.info("dropout(in, out, rnn): %s(%.2f, %.2f, %s)" % ('variational', p_in, p_out, p_rnn))
    print('# of Parameters: %d' % (sum([param.numel() for param in network.parameters()])))

    best_ucorrect = 0.0
    best_lcorrect = 0.0
    best_ucomlpete = 0.0
    best_lcomplete = 0.0

    best_ucorrect_nopunc = 0.0
    best_lcorrect_nopunc = 0.0
    best_ucomlpete_nopunc = 0.0
    best_lcomplete_nopunc = 0.0
    best_root_correct = 0.0
    best_total = 0
    best_total_nopunc = 0
    best_total_inst = 0
    best_total_root = 0

    best_epoch = 0

    test_ucorrect = 0.0
    test_lcorrect = 0.0
    test_ucomlpete = 0.0
    test_lcomplete = 0.0

    test_ucorrect_nopunc = 0.0
    test_lcorrect_nopunc = 0.0
    test_ucomlpete_nopunc = 0.0
    test_lcomplete_nopunc = 0.0
    test_root_correct = 0.0
    test_total = 0
    test_total_nopunc = 0
    test_total_inst = 0
    test_total_root = 0

    patient = 0
    reset = args.reset
    num_batches = num_data // batch_size + 1
    if optim == 'adam':
        opt_info = 'adam, betas=(%.1f, %.3f), eps=%.1e, amsgrad=%s' % (betas[0], betas[1], eps, amsgrad)
    else:
        opt_info = 'sgd, momentum=0.9, nesterov=True'
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss = 0.
        train_arc_loss = 0.
        train_type_loss = 0.
        num_insts = 0
        num_words = 0
        num_back = 0
        network.train()
        lr = scheduler.get_lr()[0]
        print('Epoch %d (%s, lr=%.6f, lr decay=%.6f, l2=%.1e): ' % (epoch, opt_info, lr, lr_decay, weight_decay))
        if args.cuda:
            torch.cuda.empty_cache()
        gc.collect()
        for step, data in enumerate(iterate_data(data_train, batch_size, bucketed=True, unk_replace=unk_replace, shuffle=True)):
            optimizer.zero_grad()
            words = data['WORD'].to(device)
            chars = data['CHAR'].to(device)
            postags = data['POS'].to(device)
            heads = data['HEAD'].to(device)
            types = data['TYPE'].to(device)
            masks = data['MASK'].to(device)

            nbatch = words.size(0)
            nwords = masks.sum() - nbatch

            loss_arc, loss_type = network.loss(words, chars, postags, heads, types, mask=masks)
            loss_arc = loss_arc.sum()
            loss_type = loss_type.sum()
            loss_total = loss_arc + loss_type
            if loss_ty_token:
                loss = loss_total.div(nwords)
            else:
                loss = loss_total.div(nbatch)
            loss.backward()
            if grad_clip > 0:
                clip_grad_norm_(network.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                num_insts += nbatch
                num_words += nwords
                train_loss += loss_total.item()
                train_arc_loss += loss_arc.item()
                train_type_loss += loss_type.item()

            # update log
            if step % 100 == 0:
                torch.cuda.empty_cache()
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                curr_lr = scheduler.get_lr()[0]
                log_info = '[%d/%d (%.0f%%) lr=%.6f] loss: %.4f (%.4f), arc: %.4f (%.4f), type: %.4f (%.4f)' % (step, num_batches, 100. * step / num_batches, curr_lr,
                                                                                                                train_loss / num_insts, train_loss / num_words,
                                                                                                                train_arc_loss / num_insts, train_arc_loss / num_words,
                                                                                                                train_type_loss / num_insts, train_type_loss / num_words)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        print('total: %d (%d), loss: %.4f (%.4f), arc: %.4f (%.4f), type: %.4f (%.4f), time: %.2fs' % (num_insts, num_words, train_loss / num_insts, train_loss / num_words,
                                                                                                       train_arc_loss / num_insts, train_arc_loss / num_words,
                                                                                                       train_type_loss / num_insts, train_type_loss / num_words,
                                                                                                       time.time() - start_time))
        print('-' * 125)

        # evaluate performance on dev data
        with torch.no_grad():
            pred_filename = os.path.join(result_path, 'pred_dev%d' % epoch)
            pred_writer.start(pred_filename)
            gold_filename = os.path.join(result_path, 'gold_dev%d' % epoch)
            gold_writer.start(gold_filename)

            print('Evaluating dev:')
            dev_stats, dev_stats_nopunct, dev_stats_root = eval(data_dev, network, pred_writer, gold_writer, punct_set, word_alphabet, pos_alphabet, device)

            pred_writer.close()
            gold_writer.close()

            dev_ucorr, dev_lcorr, dev_ucomlpete, dev_lcomplete, dev_total = dev_stats
            dev_ucorr_nopunc, dev_lcorr_nopunc, dev_ucomlpete_nopunc, dev_lcomplete_nopunc, dev_total_nopunc = dev_stats_nopunct
            dev_root_corr, dev_total_root, dev_total_inst = dev_stats_root

            if best_lcorrect_nopunc < dev_lcorr_nopunc or (best_lcorrect_nopunc == dev_lcorr_nopunc and best_ucorrect_nopunc < dev_ucorr_nopunc):
                best_ucorrect_nopunc = dev_ucorr_nopunc
                best_lcorrect_nopunc = dev_lcorr_nopunc
                best_ucomlpete_nopunc = dev_ucomlpete_nopunc
                best_lcomplete_nopunc = dev_lcomplete_nopunc

                best_ucorrect = dev_ucorr
                best_lcorrect = dev_lcorr
                best_ucomlpete = dev_ucomlpete
                best_lcomplete = dev_lcomplete

                best_root_correct = dev_root_corr
                best_total = dev_total
                best_total_nopunc = dev_total_nopunc
                best_total_root = dev_total_root
                best_total_inst = dev_total_inst

                best_epoch = epoch
                patient = 0
                torch.save(network.state_dict(), model_name)

                pred_filename = os.path.join(result_path, 'pred_test%d' % epoch)
                pred_writer.start(pred_filename)
                gold_filename = os.path.join(result_path, 'gold_test%d' % epoch)
                gold_writer.start(gold_filename)

                print('Evaluating test:')
                test_stats, test_stats_nopunct, test_stats_root = eval(data_test, network, pred_writer, gold_writer, punct_set, word_alphabet, pos_alphabet, device)

                test_ucorrect, test_lcorrect, test_ucomlpete, test_lcomplete, test_total = test_stats
                test_ucorrect_nopunc, test_lcorrect_nopunc, test_ucomlpete_nopunc, test_lcomplete_nopunc, test_total_nopunc = test_stats_nopunct
                test_root_correct, test_total_root, test_total_inst = test_stats_root

                pred_writer.close()
                gold_writer.close()
            else:
                patient += 1

            print('-' * 125)
            print('best dev  W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                best_ucorrect, best_lcorrect, best_total, best_ucorrect * 100 / best_total, best_lcorrect * 100 / best_total,
                best_ucomlpete * 100 / dev_total_inst, best_lcomplete * 100 / dev_total_inst,
                best_epoch))
            print('best dev  Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                best_ucorrect_nopunc, best_lcorrect_nopunc, best_total_nopunc,
                best_ucorrect_nopunc * 100 / best_total_nopunc, best_lcorrect_nopunc * 100 / best_total_nopunc,
                best_ucomlpete_nopunc * 100 / best_total_inst, best_lcomplete_nopunc * 100 / best_total_inst,
                best_epoch))
            print('best dev  Root: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
                best_root_correct, best_total_root, best_root_correct * 100 / best_total_root, best_epoch))
            print('-' * 125)
            print('best test W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                test_ucorrect, test_lcorrect, test_total, test_ucorrect * 100 / test_total, test_lcorrect * 100 / test_total,
                test_ucomlpete * 100 / test_total_inst, test_lcomplete * 100 / test_total_inst,
                best_epoch))
            print('best test Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                test_ucorrect_nopunc, test_lcorrect_nopunc, test_total_nopunc,
                test_ucorrect_nopunc * 100 / test_total_nopunc, test_lcorrect_nopunc * 100 / test_total_nopunc,
                test_ucomlpete_nopunc * 100 / test_total_inst, test_lcomplete_nopunc * 100 / test_total_inst,
                best_epoch))
            print('best test Root: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
                test_root_correct, test_total_root, test_root_correct * 100 / test_total_root, best_epoch))
            print('=' * 125)

            if patient >= reset:
                logger.info('reset optimizer momentums')
                scheduler.reset_state()
                patient = 0


if __name__ == '__main__':
    main()
