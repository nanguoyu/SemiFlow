"""
@File : data_utils.py
@Author: Dong Wang
@Date : 2020/5/2
"""
from ..engine import backend
import math
import warnings


def DataShuffle(x, y, seed: int = None):
    """Shuffle data"""
    assert len(x) == len(y), 'X does not match Y'
    if not seed:
        seed = 1
    backend.random.seed(seed)
    I = backend.random.permutation(len(x))
    return x[I], y[I]


class BatchSpliter(object):
    def __init__(self, x, y, batch_size, shuffle=True):
        if shuffle:
            self.x, self.y = DataShuffle(x, y)
        else:
            self.x, self.y = x, y
        self.num = len(x)
        self.batch_size = batch_size
        self.num_batch = int(self.num / self.batch_size)

        self._split()

    def shuffle(self):
        self.x, self.y = DataShuffle(self.x, self.y)

    def get_batch(self):
        n = 0
        while n < self.num_batch:
            yield self.x[self.index[n][0]:self.index[n][1]], self.y[self.index[n][0]:self.index[n][1]]
            n = n + 1

    def _split(self):
        self.num_batch = int(self.num / self.batch_size)
        if self.num_batch <= 0:
            warnings.warn('batch_size should be <= the num of data', UserWarning)
            self.batch_size = 1
            num_batch = self.num

        index = []
        last = 0
        for i in range(self.num_batch):
            index.append([last, last + self.batch_size])
            last = last + self.batch_size
        if index[-1][-1] <= self.num - 1:
            index.append([index[-1][-1], self.num])
            self.num_batch += 1
        self.index = index


def split_train_val(x, y, validation_split):
    assert len(x) == len(y)
    assert 0 < validation_split < 1
    num = len(x)
    num_train = int(num * (1 - validation_split))
    x_train = x[0:num_train]
    y_train = y[0:num_train]

    x_val = x[num_train:-1]
    y_val = y[num_train:-1]
    return x_train, y_train, x_val, y_val
