__author__ = 'max'

import re
import numpy as np
import torch
MAX_CHAR_LENGTH = 45

# Regular expressions used to normalize digits.
DIGIT_RE = re.compile(r"\d")


def get_batch(data, batch_size, word_id, single_id, unk_replace=0.):
    data, data_size = data
    batch_size = min(data_size, batch_size)
    index = torch.randperm(data_size).long()[:batch_size]

    words = data[word_id]
    single = data[single_id]
    index = index.to(words.device)
    max_length = words.size(1)
    words = words[index]
    if unk_replace:
        ones = single.new_ones(batch_size, max_length)
        noise = single.new_empty(batch_size, max_length).bernoulli_(unk_replace).long()
        words = words * (ones - single[index] * noise)

    batch = [words,] + [field[index] for i, field in enumerate(data) if i != word_id and i != single_id]
    return batch


def get_bucketed_batch(data, batch_size, word_id, single_id, unk_replace=0.):
    data_tensor, bucket_sizes = data
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])

    data = data_tensor[bucket_id]
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    index = torch.randperm(bucket_size).long()[:batch_size]

    words = data[word_id]
    single = data[single_id]
    index = index.to(words.device)
    bucket_length = words.size(1)
    words = words[index]
    if unk_replace:
        ones = single.new_ones(batch_size, bucket_length)
        noise = single.new_empty(batch_size, bucket_length).bernoulli_(unk_replace).long()
        words = words * (ones - single[index] * noise)

    batch = [words,] + [field[index] for i, field in enumerate(data) if i != word_id and i != single_id]
    return batch


def iterate_batch(data, batch_size, word_id, single_id, unk_replace=0., shuffle=False):
    data, data_size = data

    words = data[word_id]
    single = data[single_id]
    max_length = words.size(1)
    if unk_replace:
        ones = single.new_ones(data_size, max_length)
        noise = single.new_empty(data_size, max_length).bernoulli_(unk_replace).long()
        words = words * (ones - single * noise)

    indices = None
    if shuffle:
        indices = torch.randperm(data_size).long()
        indices = indices.to(words.device)
    for start_idx in range(0, data_size, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        batch = [field[excerpt] for i, field in enumerate(data) if i != single_id]
        yield batch


def iterate_bucketed_batch(data, batch_size, word_id, single_id, unk_replace=0., shuffle=False):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(bucket_sizes))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        data = data_tensor[bucket_id]
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        words = data[word_id]
        single = data[single_id]
        bucket_length = words.size(1)
        if unk_replace:
            ones = single.new_ones(bucket_size, bucket_length)
            noise = single.new_empty(bucket_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            indices = indices.to(words.device)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            batch = [field[excerpt] for i, field in enumerate(data) if i != single_id]
            yield batch
