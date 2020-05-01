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
        # TODO Return Linear
        pass
    elif isinstance(act, six.string_types):
        # TODO Return activation function
        pass
    elif callable(act):
        return act
    else:
        ValueError('Could not interpret '
                   'activation function:', act)


def searchActivation(activation_str: str):
    activation_str = activation_str.lower()

    if activation_str == 'sigmoid':
        pass
    elif activation_str == 'relu':
        pass
    elif activation_str == 'elu':
        pass
    elif activation_str == 'selu':
        pass
    elif activation_str == 'tanh':
        pass
    elif activation_str == 'softmax':
        pass
    elif activation_str == 'linear':
        pass
    elif activation_str == 'softplus':
        pass
    elif activation_str == 'softsign':
        pass
    else:
        raise ValueError('Could not find such activation ',
                         activation_str)


class Activation(Layer):
    def __init__(self, **kwargs):
        """Activation abstract class Constructor"""
        super(Activation, self).__init__(**kwargs)

    def ForwardPropagation(self, **kwargs):
        raise NotImplementedError

    def BackwardPropagation(self, **kwargs):
        raise NotImplementedError


# Activation classes
# Sigmoid Relu tanh softplus gelu

class Sigmoid(Activation):
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)

    def ForwardPropagation(self, inputs):
        return sigmoid(inputs)

    def BackwardPropagation(self, grads):
        return sigmoid(grads) * (1.0 - sigmoid(grads))


class Relu(Activation):
    def __init__(self, **kwargs):
        super(Relu, self).__init__(**kwargs)

    def ForwardPropagation(self, inputs):
        return relu(inputs)

    def BackwardPropagation(self, grads):
        return grads > 0


class Tanh(Activation):
    def __init__(self, **kwargs):
        super(Tanh, self).__init__(**kwargs)

    def ForwardPropagation(self, inputs):
        return tanh(inputs)

    def BackwardPropagation(self, grads):
        return 1 - tanh(grads) ** 2


class Softplus(Activation):
    def __init__(self, **kwargs):
        super(Softplus, self).__init__(**kwargs)

    def ForwardPropagation(self, inputs):
        return softplus(inputs)

    def BackwardPropagation(self, grads):
        return 1 / (1 + backend.exp(grads))


# TODO implement Gelu activation class

# activation function

def sigmoid(x):
    """Sigmoid activation function

    # Arguments
        x:Input tensor

    # Returns
        The sigmoid activation: 1/(1+exp(-x))
    """
    return 1 / (1 + backend.exp(-x))


def relu(x):
    """ReLu activation function

    # Arguments
        x:Input tensor

    # Returns
        The ReLu activation: max(x,0)
    """

    return backend.max(x, 0)


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
