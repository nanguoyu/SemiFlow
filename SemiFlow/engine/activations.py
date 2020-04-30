"""
@File : activations.py
@Author: Dong Wang
@Date : 2020/3/31
@Description :Basic activation functions
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from SemiFlow.engine import backend, DEFAULT_GRAPH, Node, Add, MatMul, Multiply, Square, Log, Negative, Operation, \
    Variable, \
    Placeholder
from SemiFlow.engine.utils import deprecated


# TODO modify activation methods to class


class Sigmoid(Operation):
    """ An sigmoid operation.
    """

    def __init__(self, x, name=None):
        """Sigmoid constructor.
        """
        super(Sigmoid, self).__init__(x, name=None)

    def compute_output(self):
        """ Compute and return the value of Sigmoid operation.
        """
        x, = self.input_nodes
        self.output_value = 1 / (1 + backend.exp(-x.output_value))
        return self.output_value

    def compute_gradient(self, grad=None):
        """Compute and return the value of Sigmoid operation
        """
        x = self.input_nodes[0].output_value
        if grad is None:
            grad = backend.ones_like(self.output_value)
        return grad * self.output_value * (1 - self.output_value)

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

    # def __matmul__(self, other):
    #     return MatMul(self, other)
    def dot(self, other):
        return MatMul(self, other)


class ReLU(Operation):
    """An ReLu Operation
    """

    def __init__(self, x, name=None):
        """Sigmoid constructor.
        """
        super(ReLU, self).__init__(x, name=None)

    def compute_output(self):
        """ Compute and return the value of ReLu operation.
        """
        x, = self.input_nodes
        self.output_value = backend.max([x.output_value, 0])
        return self.output_value

    def compute_gradient(self, grad=None):
        """Compute and return the value of Sigmoid operation
              """
        x = self.input_nodes[0].output_value
        if grad is None:
            grad = backend.ones_like(self.output_value)
        if self.output_value <= 0:
            """ Let d_ReLu(0) = 0
            """
            return grad * 0
        else:
            return grad * 1

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

    # def __matmul__(self, other):
    #     return MatMul(self, other)
    def dot(self, other):
        return MatMul(self, other)


@deprecated
def sigmoid(x):
    """Sigmoid activation function

    # Arguments
        x:Input tensor

    # Returns
        The sigmoid activation: 1/(1+exp(-x))
    """
    return 1 / (1 + backend.exp(-x))


@deprecated
def relu(x):
    """ReLu activation function

    # Arguments
        x:Input tensor

    # Returns
        The ReLu activation: max(x,0)
    """

    return backend.max(x, 0)


@deprecated
def tanh(x):
    """tanh activation function

    # Arguments
        x:Input tensor

    # Returns
        The tanh activation: (exp(x)-exp(-x))/(exp(x)+exp(-x))
    """
    return (backend.exp(x) - backend.exp(-x)) / (backend.exp(x) + backend.exp(-x))


@deprecated
def softplus(x):
    """soft activation function

    # Arguments
        x:Input tensor

    # Returns
        The softplus activation: ln(1+exp(x))
    """
    return backend.log(1 + backend.exp(x))


@deprecated
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
