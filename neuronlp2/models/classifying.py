__author__ = 'max'

from overrides import overrides
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
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def clone_core(self):
        return nn.Module()

    def loss(self, x, y):
        out = self.core(x)
        loss = self.criterion(out, y).mean()
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
        if x_val is not None:
            best_core = self.clone_core()
        else:
            best_core = None
        best_acc = 0.
        patient = 0
        for epoch in range(500):
            self.core.train()
            for x, y in iterate_batch(x_train, y_train, batch_size=4096, shuffle=True):
                x = x.to(device)
                y = y.to(device)
                loss = self.loss(x, y)
                loss.backward()
                optimizer.step()
                scheduler.step()

            if x_val is not None:
                with torch.no_grad():
                    acc = self.score(x_val, y_val, device)
                    # print('epoch: {}, acc: {:.2f}'.format(epoch, acc))
                    if best_acc < acc:
                        best_core.load_state_dict(self.core.state_dict())
                        best_acc = acc
                        patient = 0
                    else:
                        patient += 1
            else:
                best_acc = acc

            if patient > 4:
                break
        if best_core is not None:
            self.core.load_state_dict(best_core.state_dict())


class LinearClassifier(Classifier):
    def __init__(self, num_features, num_labels):
        super(LinearClassifier, self).__init__()
        self.dropout = 0.4
        self.core = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(num_features, num_labels)
        )
        self.num_features = num_features
        self.num_labels = num_labels

    @overrides
    def clone_core(self):
        core = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.num_features, self.num_labels)
        )
        return core


class MLPClassifier(Classifier):
    def __init__(self, num_features, num_labels, hidden_features=None):
        super(MLPClassifier, self).__init__()
        self.dropout = 0.4
        if hidden_features is None:
            hidden_features = 4 * num_features
        self.core = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(num_features, hidden_features),
            nn.ELU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_features, num_labels)
        )
        self.hidden_features = hidden_features
        self.num_features = num_features
        self.num_labels = num_labels

    @overrides
    def clone_core(self):
        core = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.num_features, self.hidden_features),
            nn.ELU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_features, self.num_labels)
        )
        return core
