__author__ = 'max'

from overrides import overrides
import numpy as np
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW

from neuronlp2.optim import ExponentialScheduler


def iterate_batch(data, labels, batch_size, shuffle=False):
    data_size = data.size(0)
    indices = None
    if shuffle:
        indices = torch.randperm(data_size).long()
        indices = indices.to(data.device)
    for start_idx in range(0, data_size, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield data[excerpt], labels[excerpt]


class Classifier:
    def __init__(self):
        self.core = nn.Module()
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def loss(self, x, y):
        out = self.core(x)
        loss = self.criterion(out, y)
        return loss

    def score(self, x, y, device):
        self.core = self.core.to(device)
        self.core.eval()
        corr = 0.
        total = x.size(0)
        for bx, by in iterate_batch(x, y, 1024, shuffle=False):
            bx = bx.to(device)
            by = by.to(device)
            out = self.core(bx)
            pred = torch.argmax(out, dim=1)
            corr += pred.eq(by).float().sum().item()
        return corr / total * 100

    def fit(self, x_train, y_train, x_val=None, y_val=None, device=torch.device('cpu')):
        optimizer = AdamW(self.core.parameters(), lr=1e-3, weight_decay=5e-4)
        init_lr = 1e-7
        scheduler = ExponentialScheduler(optimizer, 0.999995, 30, init_lr)
        self.core = self.core.to(device)

        for epoch in range(100):
            self.core.train()
            for x, y in iterate_batch(x_train, y_train, batch_size=1024, shuffle=True):
                x = x.to(device)
                y = y.to(device)
                loss = self.loss(x, y)
                loss.backward()
                optimizer.step()
                scheduler.step()

            if x_val is not None:
                with torch.no_grad():
                    acc = self.score(x_val, y_val, device)
                    print('epoch: {}, acc: {.2f}'.format(epoch, acc))


class LinearClassifier(Classifier):
    def __init__(self, num_features, num_labels):
        super(LinearClassifier, self).__init__()
        self.core = nn.Sequential(
            nn.Dropout(p=0.33),
            nn.Linear(num_features, num_labels)
        )
