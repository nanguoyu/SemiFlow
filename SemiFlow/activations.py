"""
@File : activations.py
@Author: Dong Wang
@Date : 2020/4/30
"""
from .engine.core import backend
from .layer.core import Layer
import six


def get(act):
    if not act:
        return Linear()
    elif isinstance(act, six.string_types):
        return searchActivation(act)
    elif callable(act):
        return act
    else:
        ValueError('Could not interpret '
                   'activation function:', act)


def searchActivation(activation_str: str):
    activation_str = activation_str.lower()

    if activation_str == 'sigmoid':
        return Sigmoid()
    elif activation_str == 'relu':
        return Relu()
    elif activation_str == 'tanh':
        return Tanh()
    elif activation_str == 'softmax':
        return Softmax()
    elif activation_str == 'linear':
        return Linear()
    elif activation_str == 'softplus':
        return Softplus()
    else:
        raise ValueError('Could not find such activation ',
                         activation_str)


# Todo All activation.BackwardPropagation should be reviewed
class Activation(Layer):
    def __init__(self, **kwargs):
        """Activation abstract class Constructor"""
        super(Activation, self).__init__(**kwargs)
        self.input_value = None

    def ForwardPropagation(self, **kwargs):
        raise NotImplementedError

    def BackwardPropagation(self, **kwargs):
        raise NotImplementedError


# Activation classes
# Sigmoid Relu tanh softplus gelu

class Linear(Activation):
    def __init__(self, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.name = 'linear'

    def ForwardPropagation(self, inputs):
        self.output_value = inputs
        self.input_value = inputs
        return self.output_value

    def BackwardPropagation(self, grads=None):
        # TODO review the derivative of Linear activation function
        if grads is None:
            grads = backend.ones_like(self.output_value)
        return grads


class Sigmoid(Activation):
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)
        self.name = 'sigmoid'

    def ForwardPropagation(self, inputs):
        self.output_value = sigmoid(inputs)
        self.input_value = inputs
        return self.output_value

    def BackwardPropagation(self, grads=None):
        if grads is None:
            grads = backend.ones_like(self.output_value)
        return grads * self.input_value * (1 - self.input_value)


class Relu(Activation):
    def __init__(self, **kwargs):
        super(Relu, self).__init__(**kwargs)
        self.name = 'relu'

    def ForwardPropagation(self, inputs):
        self.output_value = relu(inputs)
        self.input_value = inputs
        return self.output_value

    def BackwardPropagation(self, grads=None):
        if grads is None:
            grads = backend.ones_like(self.output_value)
        grad_wrt_x = (self.input_value > 0.0) * grads
        return grad_wrt_x


class Tanh(Activation):
    def __init__(self, **kwargs):
        super(Tanh, self).__init__(**kwargs)
        self.name = 'tanh'

    def ForwardPropagation(self, inputs):
        self.output_value = tanh(inputs)
        self.input_value = inputs
        return self.output_value

    def BackwardPropagation(self, grads=None):
        if grads is None:
            grads = backend.ones_like(self.output_value)
        return grads * (1 - self.input_value ** 2)


class Softplus(Activation):
    def __init__(self, **kwargs):
        super(Softplus, self).__init__(**kwargs)
        self.name = 'softplus'

    def ForwardPropagation(self, inputs):
        self.output_value = softplus(inputs)
        self.input_value = inputs
        return self.output_value

    def BackwardPropagation(self, grads=None):
        if grads is None:
            grads = backend.ones_like(self.output_value)
        return grads * 1 / (1 + backend.exp(self.input_value))


class Softmax(Activation):
    def __init__(self, **kwargs):
        super(Activation, self).__init__(**kwargs)
        self.name = 'softmax'

    def ForwardPropagation(self, inputs):
        self.input_value = inputs
        self.output_value = softmax(inputs)
        return self.output_value

    def BackwardPropagation(self, grads=None):
        """
        z = Logits = self.input_value
        f = softmax(Logits) = self.output_value
        df_dz
        Args:
            grads:

        Returns:

        """
        if grads is None:
            grads = backend.ones_like(self.output_value)
        df_dz = backend.zeros([len(self.input_value), len(self.output_value)])
        for j in range(len(self.input_value)):
            for i in range(len(self.output_value)):
                if i == j:
                    df_dz[j, i] = self.output_value[j, i] * (1 - self.output_value[j, i])
                else:
                    df_dz[j, i] = -self.output_value[j, i] * self.output_value[j, j]

        # Todo: check df_dz
        return grads * df_dz


# TODO implement Gelu activation class

# activation function

def sigmoid(x):
    """Sigmoid activation function

    # Arguments
        x:Input tensor

    # Returns
        The sigmoid activation: 1/(1+exp(-x))
    """
    return 1. / (1. + backend.exp(-x))


def relu(x):
    """reLu activation function

    # Arguments
        x:Input tensor

    # Returns
        The ReLu activation: max(x,0)
    """

    return backend.maximum(x, 0.0)


def tanh(x):
    """tanh activation function

    # Arguments
        x:Input tensor

    # Returns
        The tanh activation: (exp(x)-exp(-x))/(exp(x)+exp(-x))
    """
    return (backend.exp(x) - backend.exp(-x)) / (backend.exp(x) + backend.exp(-x))


def softplus(x):
    """soft activation function

    # Arguments
        x:Input tensor

    # Returns
        The softplus activation: ln(1+exp(x))
    """
    return backend.log(1 + backend.exp(x))


def gelu(x):
    """GELU activation function
    The original definition of GELU is x*cdf(x). https://arxiv.org/abs/1606.08415
    # TODO consider  an approximation from BERT : x * 0.5 * (1.0 + erf( x / sqrt(x) ) )

    # Arguments
        x:Input tensor

    # Returns
        An approximation of GELU activation: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x+ 0.044715*x^3)  ))
    """
    return 0.5 * x * (1 + tanh(backend.sqrt(2 / backend.pi) * (x + 0.044715 * x ^ 3)))


def softmax(x, axis=-1):
    """softmax activation function
    Args:
        x:logits
        axis:
    softmax = exp(logits) / sum(exp(logits), axis)
    Returns:

    """
    ndim = backend.ndim(x)
    if ndim == 2:
        return backend.exp(x) / backend.sum(backend.exp(x), axis=-1, keepdims=True)
    elif ndim > 2:
        e = backend.exp(x - backend.max(x, axis=axis, keepdims=True))
        s = backend.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D. '
                         'Received input: %s' % x)
