__author__ = 'max'

import random
import copy
import numpy as np
from .reader import CoNLLXReader
import utils
import torch
from torch.autograd import Variable
from .conllx_data import _buckets, PAD_ID_WORD, PAD_ID_CHAR, PAD_ID_TAG, UNK_ID
from .conllx_data import NUM_SYMBOLIC_TAGS
from .conllx_data import create_alphabets


def _get_children(heads):
    child_ids = [[] for _ in range(len(heads))]
    # skip the symbolic root.
    for child in range(1, len(heads)):
        head = heads[child]
        child_ids[head].append(child)

    return child_ids


def _generate_stack_inputs(types, child_ids):
    new_child_ids = copy.deepcopy(child_ids)
    for childs in new_child_ids:
        random.shuffle(childs)
    stacked_heads = []
    children = []
    stacked_types = []
    stack = [0]
    while len(stack) > 0:
        head = stack[-1]
        stacked_heads.append(head)
        child_id = new_child_ids[head]
        if len(child_id) == 0:
            children.append(0)
            stacked_types.append(PAD_ID_TAG)
            stack.pop()
        else:
            child = child_id.pop(0)
            children.append(child)
            stack.append(child)
            stacked_types.append(types[child])

    return stacked_heads, children, stacked_types


def read_stacked_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_size=None, normalize_digits=True):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                children = _get_children(inst.heads)
                data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, children])
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    reader.close()
    print("Total number of data: %d" % counter)
    return data, max_char_length


def get_batch(data, batch_size, word_alphabet=None, unk_replace=0., use_gpu=False, volatile=False):
    data, max_char_length = data
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
    char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)

    wid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    cid_inputs = np.empty([batch_size, bucket_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    hid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    tid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)

    masks_e = np.zeros([batch_size, bucket_length], dtype=np.float32)
    single = np.zeros([batch_size, bucket_length], dtype=np.int64)
    lengths_e = np.empty(batch_size, dtype=np.int64)

    stack_hid_inputs = np.empty([batch_size, 2 * bucket_length - 1], dtype=np.int64)
    chid_inputs = np.empty([batch_size, 2 * bucket_length - 1], dtype=np.int64)
    stack_tid_inputs = np.empty([batch_size, 2 * bucket_length - 1], dtype=np.int64)

    masks_d = np.zeros([batch_size, 2 * bucket_length - 1], dtype=np.float32)
    lengths_d = np.empty(batch_size, dtype=np.int64)

    for b in range(batch_size):
        wids, cid_seqs, pids, hids, tids, child_ids = random.choice(data[bucket_id])
        inst_size = len(wids)
        lengths_e[b] = inst_size
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
        hid_inputs[b, inst_size:] = PAD_ID_TAG
        # masks
        masks_e[b, :inst_size] = 1.0

        for j, wid in enumerate(wids):
            if word_alphabet.is_singleton(wid):
                single[b, j] = 1

        stack_hids, chids, stack_tids = _generate_stack_inputs(tids, child_ids)
        inst_size_decoder = 2 * inst_size - 1
        lengths_d[b] = inst_size_decoder

        # stacked heads
        stack_hid_inputs[b, :inst_size_decoder] = stack_hids
        stack_hid_inputs[b, inst_size_decoder:] = PAD_ID_TAG
        # children
        chid_inputs[b, :inst_size_decoder] = chids
        chid_inputs[b, inst_size_decoder:] = PAD_ID_TAG
        # stacked types
        stack_tid_inputs[b, :inst_size_decoder] = stack_tids
        stack_tid_inputs[b, inst_size_decoder:] = PAD_ID_TAG
        # masks_d
        masks_d[b, :inst_size_decoder] = 1.0

    if unk_replace:
        noise = np.random.binomial(1, unk_replace, size=[batch_size, bucket_length])
        wid_inputs = wid_inputs * (1 - noise * single)

    words = Variable(torch.from_numpy(wid_inputs), volatile=volatile)
    chars = Variable(torch.from_numpy(cid_inputs), volatile=volatile)
    pos = Variable(torch.from_numpy(pid_inputs), volatile=volatile)
    heads = Variable(torch.from_numpy(hid_inputs), volatile=volatile)
    types = Variable(torch.from_numpy(tid_inputs), volatile=volatile)
    masks_e = Variable(torch.from_numpy(masks_e), volatile=volatile)
    lengths_e = torch.from_numpy(lengths_e)

    stacked_heads = Variable(torch.from_numpy(stack_hid_inputs), volatile=volatile)
    children = Variable(torch.from_numpy(chid_inputs), volatile=volatile)
    stacked_types = Variable(torch.from_numpy(stack_tid_inputs), volatile=volatile)
    masks_d = Variable(torch.from_numpy(masks_d), volatile=volatile)
    lengths_d = torch.from_numpy(lengths_d)

    if use_gpu:
        words = words.cuda()
        chars = chars.cuda()
        pos = pos.cuda()
        heads = heads.cuda()
        types = types.cuda()
        masks_e = masks_e.cuda()
        lengths_e = lengths_e.cuda()
        stacked_heads = stacked_heads.cuda()
        children = children.cuda()
        stacked_types = stacked_types.cuda()
        masks_d = masks_d.cuda()
        lengths_d = lengths_d.cuda()

    return (words, chars, pos, heads, types, masks_e, lengths_e), \
           (stacked_heads, children, stacked_types, masks_d, lengths_d)


def iterate_batch(data, batch_size, word_alphabet=None, unk_replace=0., shuffle=False, use_gpu=False, volatile=False):
    data, max_char_length = data
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
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks_e = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)
        lengths_e = np.empty(bucket_size, dtype=np.int64)

        stack_hid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        chid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        stack_tid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)

        masks_d = np.zeros([bucket_size, 2 * bucket_length - 1], dtype=np.float32)
        lengths_d = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids, child_ids = inst
            inst_size = len(wids)
            lengths_e[i] = inst_size
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
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks_e
            masks_e[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

            stack_hids, chids, stack_tids = _generate_stack_inputs(tids, child_ids)
            inst_size_decoder = 2 * inst_size - 1
            lengths_d[i] = inst_size_decoder

            # stacked heads
            stack_hid_inputs[i, :inst_size_decoder] = stack_hids
            stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # children
            chid_inputs[i, :inst_size_decoder] = chids
            chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # stacked types
            stack_tid_inputs[i, :inst_size_decoder] = stack_tids
            stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # masks_d
            masks_d[i, :inst_size_decoder] = 1.0


        if unk_replace:
            noise = np.random.binomial(1, unk_replace, size=[bucket_size, bucket_length])
            wid_inputs = wid_inputs * (1 - noise * single)

        words = Variable(torch.from_numpy(wid_inputs), volatile=volatile)
        chars = Variable(torch.from_numpy(cid_inputs), volatile=volatile)
        pos = Variable(torch.from_numpy(pid_inputs), volatile=volatile)
        heads = Variable(torch.from_numpy(hid_inputs), volatile=volatile)
        types = Variable(torch.from_numpy(tid_inputs), volatile=volatile)
        masks_e = Variable(torch.from_numpy(masks_e), volatile=volatile)
        lengths_e = torch.from_numpy(lengths_e)

        stacked_heads = Variable(torch.from_numpy(stack_hid_inputs), volatile=volatile)
        children = Variable(torch.from_numpy(chid_inputs), volatile=volatile)
        stacked_types = Variable(torch.from_numpy(stack_tid_inputs), volatile=volatile)
        masks_d = Variable(torch.from_numpy(masks_d), volatile=volatile)
        lengths_d = torch.from_numpy(lengths_d)

        if use_gpu:
            words = words.cuda()
            chars = chars.cuda()
            pos = pos.cuda()
            heads = heads.cuda()
            types = types.cuda()
            masks_e = masks_e.cuda()
            lengths_e = lengths_e.cuda()
            stacked_heads = stacked_heads.cuda()
            children = children.cuda()
            stacked_types = stacked_types.cuda()
            masks_d = masks_d.cuda()
            lengths_d = lengths_d.cuda()

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            if use_gpu:
                indices = indices.cuda()

        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield (words[excerpt], chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt],
                   masks_e[excerpt], lengths_e[excerpt]), \
                  (stacked_heads[excerpt], children[excerpt], stacked_types[excerpt],
                   masks_d[excerpt], lengths_d[excerpt])

