"""
@File : rnn.py
@Author: Dong Wang
@Date : 2020/10/27
"""
from ..engine.core import backend
from .core import Layer
from .. import activations
from .. import initializers


class RNN(Layer):
    def __init__(self, units,
                 activation='tanh',
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
        super(RNN, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.isInitialized = False

    def ForwardPropagation(self):
        assert self.isInitialized, "you should init_para"
        """
        a[t] = b + W*h[t-1] + U*x[t]
        h[t] = tanh(a[t])
        o[t] = c + V*h[t]
        y[t] = softmax(o[t])
        Returns:y[t]

        """

        # print("FP:", self.name)
        x = self.inbound[0].output_value
        batch_size, time_step, input_shape = x.shape
        a = backend.empty(shape=[batch_size, time_step, self.units])
        h = backend.empty(shape=[batch_size, time_step + 1, self.units])
        # o = backend.empty(shape=[batch_size, time_step, input_shape])
        o = backend.empty(shape=[batch_size, time_step, self.units])
        h[:, -1] = backend.zeros(shape=[batch_size, self.units])

        for t in range(time_step):
            a[:, t] = self.params['b'] + backend.matmul(x[:, t], self.params['U'].T) + backend.matmul(
                h[:, t - 1], self.params['W'].T)
            h[:, t] = self.activation.ForwardPropagation(a[:, t])
            o[:, t] = self.params['c'] + backend.matmul(h[:, t], self.params['V'].T)

        # Return the last output
        self.output_value = o[:, -1]
        if hasattr(self, 'dtype'):
            self.output_value = self.output_value.astype(self.dtype)
        self.h, self.a, self.X = h, a, x
        return self.output_value

    def BackwardPropagation(self, grad=None):
        """
        BPTT
        Args:
            grad: gradients from output layer

        Returns: gradients of this layer

        This function is forked from Tinynn
        """
        assert self.isInitialized, "you should init_para"
        if grad is None:
            grad = backend.ones_like(self.output_value)

        time_step = self.X.shape[1]
        for param in self.params.keys():
            self.grads[param] = backend.zeros_like(self.params[param])

        grad_wrt_x = backend.empty_like(self.X)
        for t in reversed(range(time_step)):
            # grads w.r.t param V and c
            self.grads["c"] += grad.sum(axis=0)
            self.grads["V"] += backend.matmul(grad.T, self.h[:, t])
            # grads w.r.t h
            d_h = backend.matmul(grad, self.params["V"])
            d_a = d_h * self.activation.BackwardPropagation(self.a[:, t])
            # grads w.r.t input X
            grad_wrt_x[:, t] = backend.matmul(d_a, self.params["U"])
            # grads w.r.t params U, W and b
            for i in range(min(time_step, t + 1)):
                self.grads["U"] += backend.matmul(d_a.T, self.X[:, t - i])
                self.grads["W"] += backend.matmul(d_a.T, self.h[:, t - i - 1])
                self.grads["b"] += d_a.sum(axis=0)
                d_h = backend.matmul(d_a, self.params["W"])
                d_a = d_h * self.activation.BackwardPropagation(self.a[:, t - i - 1])
        return grad_wrt_x

    def InitParams(self):
        # print("Init ", self.name)
        if hasattr(self, 'input_shape'):
            time_step, input_shape = self.input_shape
        else:
            # ToDo: support RNN in non-first layer
            time_step, input_shape = self.inbound[0].shape
        # print(self.name+'.InitParams', self.shape)
        W = self.kernel_initializer(shape=[self.units, self.units])
        b = self.bias_initializer(shape=[1]) * backend.ones([self.units]).T
        U = self.kernel_initializer(shape=[self.units, input_shape])
        # c = self.bias_initializer(shape=[1]) * backend.ones([input_shape]).T
        c = self.bias_initializer(shape=[1]) * backend.ones([self.units]).T
        # V = self.kernel_initializer(shape=[input_shape, self.units])
        V = self.kernel_initializer(shape=[self.units, self.units])
        self.params = {
            'W': W,
            'b': b,
            'U': U,
            'c': c,
            'V': V
        }
        self.grads = {
            'W': backend.array([]),
            'b': backend.array([]),
            'U': backend.array([]),
            'c': backend.array([]),
            'V': backend.array([])
        }
        self.isInitialized = True
