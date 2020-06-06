"""
@File : dense.py
@Author: Dong Wang
@Date : 2020/4/30
"""

from ..engine.core import backend
from .core import Layer
from .. import activations
from .. import initializers


class Dense(Layer):
    def __init__(self, units,
                 activation=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 **kwargs):
        """
        Args:
            units: number of neural, dimensionality of output
            activation: activation function
                If you don't specify anything linear activation is applied
            kernel_initializer: Initializer for the `kernel` weights matrix
            bias_initializer: Initializer for the bias vector
        """
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.getInitializer(kernel_initializer)
        self.bias_initializer = initializers.getInitializer(bias_initializer)

    def BackwardPropagation(self):
        # For example : self.output_value = 0
        pass

    def ForwardPropagation(self):
        pass

    def _init_params(self):
        # TODO init params by self.kernel_initializer and self.bias_initializer
        units = self.units
        if hasattr(self, 'input_shape'):
            input_shape = self.input_shape
        self.params = {'kernel': None, 'bias': None}
