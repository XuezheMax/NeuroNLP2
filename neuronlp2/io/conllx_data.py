__author__ = 'max'

import os.path
import random
import numpy as np
from .reader import CoNLLXReader
from .alphabet import Alphabet
from .logger import get_logger
import utils

# Special vocabulary symbols - we always put them at the start.
PAD = b"_PAD"
PAD_POS = b"_PAD_POS"
PAD_TYPE = b"_<PAD>"
PAD_CHAR = b"_PAD_CHAR"
ROOT = b"_ROOT"
ROOT_POS = b"_ROOT_POS"
ROOT_TYPE = b"_<ROOT>"
ROOT_CHAR = b"_ROOT_CHAR"
END = b"_END"
END_POS = b"_END_POS"
END_TYPE = b"_<END>"
END_CHAR = b"_END_CHAR"
_START_VOCAB = [PAD, ROOT, END]

UNK_ID = 0
PAD_ID_WORD = 1
PAD_ID_CHAR = 1
PAD_ID_TAG = 0

NUM_SYMBOLIC_TAGS = 3



_buckets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 140]


def create_alphabets(alphabet_directory, data_paths, max_vocabulary_size, min_occurence=1, normalize_digits=True):
    logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet('word', defualt_value=True)
    char_alphabet = Alphabet('character', defualt_value=True)
    pos_alphabet = Alphabet('pos')
    type_alphabet = Alphabet('type')
    if not os.path.isdir(alphabet_directory):
        logger.info("Creating Alphabets: %s" % alphabet_directory)

        char_alphabet.add(PAD_CHAR)
        pos_alphabet.add(PAD_POS)
        type_alphabet.add(PAD_TYPE)

        char_alphabet.add(ROOT_CHAR)
        pos_alphabet.add(ROOT_POS)
        type_alphabet.add(ROOT_TYPE)

        char_alphabet.add(END_CHAR)
        pos_alphabet.add(END_POS)
        type_alphabet.add(END_TYPE)

        vocab = dict()
        for data_path in data_paths:
            logger.info("Processing data: %s" % data_path)
            with open(data_path, 'r') as file:
                for line in file:
                    line = line.decode('utf-8')
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    tokens = line.split('\t')
                    for char in tokens[1]:
                        char_alphabet.add(char)

                    word = utils.DIGIT_RE.sub(b"0", tokens[1]) if normalize_digits else tokens[1]
                    pos = tokens[4]
                    type = tokens[7]

                    pos_alphabet.add(pos)
                    type_alphabet.add(type)

                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurence]
        logger.info("Total Vocabulary Size: %d" % len(vocab_list))

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        for word in vocab_list:
            word_alphabet.add(word)

        logger.info("Word Alphabet Size: %d" % word_alphabet.size())
        logger.info("Character Alphabet Size: %d" % char_alphabet.size())
        logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
        logger.info("Type Alphabet Size: %d" % type_alphabet.size())

        word_alphabet.save(alphabet_directory)
        char_alphabet.save(alphabet_directory)
        pos_alphabet.save(alphabet_directory)
        type_alphabet.save(alphabet_directory)

    else:
        word_alphabet.load(alphabet_directory)
        char_alphabet.load(alphabet_directory)
        pos_alphabet.load(alphabet_directory)
        type_alphabet.load(alphabet_directory)

    word_alphabet.close()
    char_alphabet.close()
    pos_alphabet.close()
    type_alphabet.close()
    return word_alphabet, char_alphabet, pos_alphabet, type_alphabet


def read_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_size=None,
              normalize_digits=True):
    data = [[] for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    inst = reader.getNext(normalize_digits)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids])
                break

        inst = reader.getNext(normalize_digits)
    reader.close()
    print("Total number of data: %d" % counter)
    return data


def get_batch(data, batch_size):
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])

    bucket_length = _buckets[bucket_id]

    wid_inputs = np.empty([batch_size, bucket_length], dtype=np.int32)
    cid_inputs = np.empty([batch_size, bucket_length, utils.MAX_CHAR_LENGTH], dtype=np.int32)
    pid_inputs = np.empty([batch_size, bucket_length], dtype=np.int32)
    hid_inputs = np.empty([batch_size, bucket_length], dtype=np.int32)
    tid_inputs = np.empty([batch_size, bucket_length], dtype=np.int32)

    masks = np.zeros([batch_size, bucket_length], dtype=np.float32)

    for b in range(batch_size):
        wids, cid_seqs, pids, hids, tids = random.choice(data[bucket_id])

        inst_size = len(wids)
        # word ids
        wid_inputs[b, :inst_size] = wids
        wid_inputs[b, inst_size:] = PAD_ID_WORD
        for c, cids in enumerate(cid_seqs):
            cid_inputs[b, c, :len(cids)] = cids
            cid_inputs[b, c, len(cids):] = PAD_ID_CHAR
        cid_inputs[b, inst_size:, :] = PAD_ID_CHAR
        # pos ids
        pid_inputs[b, :inst_size] = pids
        pid_inputs[b, inst_size:] = PAD_ID_TAG
        # type ids
        tid_inputs[b, :inst_size] = tids
        tid_inputs[b, inst_size:] = PAD_ID_TAG
        # heads
        hid_inputs[b, :inst_size] = hids
        hid_inputs[b, inst_size:] = -1
        # masks
        masks[b, :inst_size] = 1.0

    return wid_inputs, cid_inputs, pid_inputs, hid_inputs, tid_inputs, masks


def iterate_batch(data, batch_size, shuffle=False):
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        bucket_length = _buckets[bucket_id]
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int32)
        cid_inputs = np.empty([bucket_size, bucket_length, utils.MAX_CHAR_LENGTH], dtype=np.int32)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int32)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int32)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int32)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids = inst
            inst_size = len(wids)
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = -1
            # masks
            masks[i, :inst_size] = 1.0

        indices = None
        if shuffle:
            indices = np.arange(bucket_size)
            np.random.shuffle(indices)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield wid_inputs[excerpt], cid_inputs[excerpt], pid_inputs[excerpt], hid_inputs[excerpt], \
                  tid_inputs[excerpt], masks[excerpt]
