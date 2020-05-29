"""
@File : optimizer.py
@Author: Dong Wang
@Date : 2020/5/1
"""
from .engine.core import backend
from .layer.core import Layer
from .utils import BatchSpliter
import six


class Optimizer(object):

    def __init__(self, learning_rate, loss, batch_size, epochs, **kwargs):
        self.learning_rate = learning_rate
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        super(Optimizer, self).__init__(**kwargs)

    def _updateParameters(self):
        raise NotImplementedError


class GradientDescentOptimizer(Optimizer):

    def __init__(self, learning_rate, loss, **kwargs):
        self.learning_rate = learning_rate
        self.loss = loss
        self.spliter = None
        super(GradientDescentOptimizer, self).__init__(**kwargs)

    def build(self, x, y, epochs, batch_size):
        self.spliter = BatchSpliter(x, y, batch_size=batch_size)
        self.epochs = epochs
        self.batch_size = batch_size

    def _updateParameters(self):
        pass

    def _ForwardPropagation(self, data, params, grads, batch, **kwargs):
        # TODO optimizer.GradientDescentOptimizer.ForwardPropagation
        for epoch in range(self.epochs):
            for xbatch, ybatch in self.spliter.get_batch():
                # xbatch, ybatch
                pass

    def _BackwardPropagation(self):
        for epoch in range(self.epochs):
            for xbatch, ybatch in self.spliter.get_batch():
                # xbatch, ybatch
                pass


def getOptimizer(opt, loss, learning_rate=0.0005):
    if isinstance(opt, six.string_types):
        # TODO Return initializers
        if opt == 'GD':
            return GradientDescentOptimizer(learning_rate, loss, learning_rate=0.0005)
    else:
        ValueError('Could not interpret '
                   'initializer:', opt)
