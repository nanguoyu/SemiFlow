"""
@File : activations.py
@Author: Dong Wang
@Date : 2020/3/31
@Description :Basic activation functions
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .engine import backend


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
