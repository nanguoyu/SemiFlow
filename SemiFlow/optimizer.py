"""
@File : optimizer.py
@Author: Dong Wang
@Date : 2020/5/1
"""
from .engine.core import backend
from .layer.core import Layer


class Optimizer(Layer):

    def __init__(self, learning_rate, loss, **kwargs):
        self.learning_rate = learning_rate
        self.loss = loss
        super(Optimizer, self).__init__(**kwargs)

    def ForwardPropagation(self, **kwargs):
        raise NotImplementedError

    def BackwardPropagation(self, **kwargs):
        pass


class GradientDescentOptimizer(Optimizer):

    def __init__(self, learning_rate, loss, **kwargs):
        self.learning_rate = learning_rate
        self.loss = loss
        super(GradientDescentOptimizer, self).__init__(**kwargs)

    def ForwardPropagation(self, data, params, grads, batch, **kwargs):
        # TODO optimizer.GradientDescentOptimizer.ForwardPropagation

        # for batch in
        pass
