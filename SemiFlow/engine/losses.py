"""
@File : losses.py
@Author: Dong Wang
@Date : 2020/3/31
@Description :Basic loss function
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from SemiFlow.engine import backend
from SemiFlow.engine import Operation


def binary_cross_entropy(y, p):
    """
    binary cross entropy loss function
    :param y: Y_true
    :param p: Y_predict
    :return: loss value
    """
    assert y.shape[0] == p.shape[0], "wrong shape"
    loss = 0
    for i in range(y.shape[0]):
        loss += - (y[i] * backend.log(p[i]) + (1 - y[i]) * backend.log(1 - p[i]))
    return loss / y.shape[0]


class ReduceSum(Operation):
    """ Reduce sum operation.
    """

    def __init__(self, x, axis=None, name=None):
        """ Operation constructor.
        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.
        :param axis: The dimensions to reduce. If `None`, reduces all dimensions.
        :type axis: int.
        """
        super(ReduceSum, self).__init__(x, name=name)
        self.axis = axis

    def compute_output(self):
        """ Compute and return the value of sigmoid function.
        """
        x, = self.input_nodes
        self.output_value = backend.sum(x.output_value, self.axis)
        return self.output_value

    def compute_gradient(self, grad=None):
        """ Compute the gradient for negative operation wrt input value.
        :param grad: The gradient of other operation wrt the negative output.
        :type grad: ndarray.
        """
        input_value = self.input_nodes[0].output_value

        if grad is None:
            grad = backend.ones_like(self.output_value)
        output_shape = backend.array(backend.shape(input_value))
        output_shape[self.axis] = 1.0
        tile_scaling = backend.shape(input_value) // output_shape
        grad = backend.reshape(grad, output_shape)
        # print(backend.shape(input_value), output_shape)
        return backend.tile(grad, tile_scaling)
