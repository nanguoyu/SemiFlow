"""
@File : reshape.py
@Author: Dong Wang
@Date : 2020/6/25
"""
from ..engine.core import backend
from .core import Layer
from . import InputLayer


class Flatten(Layer):

    def __init__(self, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.isInitialized = False

    def BackwardPropagation(self, grad=None):
        if grad is None:
            grad = backend.ones_like(self.output_value)
        x = self.inbound[0]
        input_shape = x.output_value.shape
        return grad.reshape(input_shape)

    def ForwardPropagation(self):
        assert self.isInitialized, "you should init_para"
        x = self.inbound[0]
        inputs = x.output_value
        self.output_value = inputs.reshape(inputs.shape[0], -1)
        return self.output_value

    def InitParams(self):
        x = self.inbound[0]
        if hasattr(x, 'output_shape'):
            in_h, in_w, input_channel = x.output_shape
            self.shape = [in_h * in_w * input_channel]
        self.isInitialized = True
