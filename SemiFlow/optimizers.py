"""
@File : optimizers.py
@Author: Dong Wang
@Date : 2020/5/1
"""
from .engine.core import backend
from . import losses
from .layer.core import Layer
from .utils import BatchSpliter
import six


class Optimizer(object):

    def __init__(self, loss, learning_rate, **kwargs):
        self.learning_rate = learning_rate
        self.loss = losses.get(loss)
        super(Optimizer, self).__init__(**kwargs)

    def _updateParameters(self):
        raise NotImplementedError


class GradientDescentOptimizer(Optimizer):

    def __init__(self, loss, learning_rate, **kwargs):
        self.spliter = None
        self.epochs = None
        self.batch_size = None
        self.last_layer = None
        self.first_layer = None
        super(GradientDescentOptimizer, self).__init__(loss, learning_rate, **kwargs)

    def build(self, x, y, epochs, batch_size, first_layer, last_layer):
        # Called at the beginning of training
        self.spliter = BatchSpliter(x, y, batch_size=batch_size)
        self.epochs = epochs
        self.batch_size = batch_size
        self.first_layer = first_layer
        self.last_layer = last_layer
        # Bind networks and loss
        self.loss.inbound = last_layer
        last_layer.outbound = self.loss

    def _updateParameters(self):
        pass

    def _ForwardPropagation(self):
        # TODO optimizer.GradientDescentOptimizer.ForwardPropagation
        for xbatch, ybatch in self.spliter.get_batch():
            # xbatch, ybatch
            # postorder_nodes = self._get_prerequisite(operation)
            #
            # for node in postorder_nodes:
            #     if isinstance(node, Placeholder):
            #         node.output_value = feed_dict[node]
            #     else:
            #         node.compute_output()
            # return operation.output_value
            pass

    def _BackwardPropagation(self):
        for epoch in range(self.epochs):
            for xbatch, ybatch in self.spliter.get_batch():
                # xbatch, ybatch
                pass


def get(opt, loss, learning_rate=0.0005):
    if isinstance(opt, six.string_types):
        opt = opt.lower()
        if opt == 'sgd':
            return GradientDescentOptimizer(loss=loss, learning_rate=learning_rate)
        elif opt == 'rmsprop':
            # TODO Implement RMSprop
            return GradientDescentOptimizer(loss=loss, learning_rate=learning_rate)
        else:
            # TODO other Optimizer
            return GradientDescentOptimizer(loss=loss, learning_rate=learning_rate)
    else:
        ValueError('Could not interpret '
                   'initializer:', opt)
