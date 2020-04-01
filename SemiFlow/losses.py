"""
@File : losses.py
@Author: Dong Wang
@Date : 2020/3/31
@Description :Basic loss function
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .engine import backend


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
