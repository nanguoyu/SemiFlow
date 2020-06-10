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
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.isInitialized = False

    def BackwardPropagation(self, grad=None):
        """
        dw = grad*x
        db = grad
        Args:
            grad: gradients from output layer

        Returns: gradients of this layer

        """

        x, = [layer.output_value for layer in self.inbound]  # TODO support multi-inputs
        w = self.params['kernel']
        b = self.params['bias']

        if grad is None:
            grad = backend.ones_like(self.output_value)

        grad_wrt_w = grad * x
        while backend.ndim(grad_wrt_w) > len(backend.shape(w)):
            grad_wrt_w = backend.sum(grad_wrt_w, axis=0)
        for axis, size in enumerate(backend.shape(w)):
            if size == 1:
                grad_wrt_w = backend.sum(grad_wrt_w, axis=axis, keepdims=True)

        grad_wrt_b = grad
        while backend.ndim(grad_wrt_b) > len(backend.shape(b)):
            grad_wrt_b = backend.sum(grad_wrt_b, axis=0)
        for axis, size in enumerate(backend.shape(b)):
            if size == 1:
                grad_wrt_b = backend.sum(grad_wrt_b, axis=axis, keepdims=True)

        return [grad_wrt_w, grad_wrt_b]

    def ForwardPropagation(self):
        """
        logits = x * w+b
        output = activation(logits)
        Returns:output

        """
        x = self.inbound[0]
        logits = backend.matmul(x.output_value, self.params['kernel']) + self.params['bias']
        # ToDo Implement logits
        self.output_value = self.activation.ForwardPropagation(logits)
        return self.output_value

    def InitParams(self):
        self.isInitialized = True
        output_shape = self.units
        if hasattr(self, 'input_shape'):
            input_shape = self.input_shape
            self.shape = (input_shape, output_shape)
        else:
            input_shape = self.inbound[0].shape[-1]
            self.shape = (input_shape, output_shape)
        # print(self.name+'.InitParams', self.shape)
        kernel = self.kernel_initializer(shape=[input_shape, output_shape])
        bias = self.kernel_initializer(shape=[output_shape])
        self.params = {
            'kernel': kernel,
            'bias': bias}

    def UpdateParams(self):
        pass
