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
        self.original_activation_name = self.activation.name  # Sometime, the optimizer may optimize the activation
        # function
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
        if grad is None:
            grad = backend.ones_like(self.output_value)
        # print("BP:", self.name+"."+self.activation.name)
        # Activation
        grad = self.activation.BackwardPropagation(grads=grad)
        # print("BP:", self.name)
        # Layer
        x, = [layer.output_value for layer in self.inbound]  # TODO support multi-inputs
        w = self.params['kernel']
        b = self.params['bias']
        # print("x.T.shape", x.T.shape, "grad.shape", grad.shape)
        # print("w.shape", w.shape, "b.shape", b.shape)
        grad_wrt_w = backend.matmul(x.T, grad)
        # while backend.ndim(grad_wrt_w) > len(backend.shape(w)):
        #     grad_wrt_w = backend.sum(grad_wrt_w, axis=0)
        # for axis, size in enumerate(backend.shape(w)):
        #     if size == 1:
        #         grad_wrt_w = backend.sum(grad_wrt_w, axis=axis, keepdims=True)

        grad_wrt_b = backend.sum(grad, axis=0)
        # while backend.ndim(grad_wrt_b) > len(backend.shape(b)):
        #     grad_wrt_b = backend.sum(grad_wrt_b, axis=0)
        # for axis, size in enumerate(backend.shape(b)):
        #     if size == 1:
        #         grad_wrt_b = backend.sum(grad_wrt_b, axis=axis, keepdims=True)

        # print("grad_wrt_w.shape", grad_wrt_w.shape,"grad_wrt_b.shape", grad_wrt_b.shape, "grad_wrt_x.shape",
        # grad_wrt_x.shape)

        self.grads['kernel'] = grad_wrt_w
        self.grads['bias'] = grad_wrt_b
        grad_wrt_x = backend.matmul(grad, w.T)
        return grad_wrt_x

    def ForwardPropagation(self):
        assert self.isInitialized, "you should init_para"
        """
        logits = x * w+b
        output = activation(logits)
        Returns:output

        """
        # print("FP:", self.name)
        x = self.inbound[0]
        w = self.params['kernel']
        b = self.params['bias']
        logits = backend.matmul(x.output_value, w) + b
        # print("FP:", self.name+"."+self.activation.name)
        self.output_value = self.activation.ForwardPropagation(logits)
        if hasattr(self, 'dtype'):
            self.output_value = self.output_value.astype(self.dtype)
        return self.output_value

    def InitParams(self):
        # print("Init ", self.name)
        output_shape = self.units
        if hasattr(self, 'input_shape'):
            input_shape = self.input_shape[-1]
            self.shape = (input_shape, output_shape)
        else:
            input_shape = self.inbound[0].shape[-1]
            self.shape = (input_shape, output_shape)
        # print(self.name+'.InitParams', self.shape)
        kernel = self.kernel_initializer(shape=[input_shape, output_shape])
        # bias = self.kernel_initializer(shape=[output_shape])
        bias = self.bias_initializer(shape=[1]) * backend.ones([output_shape]).T
        self.params = {
            'kernel': kernel,
            'bias': bias}
        self.grads = {
            'kernel': backend.array([]),
            'bias': backend.array([])
        }
        self.isInitialized = True
