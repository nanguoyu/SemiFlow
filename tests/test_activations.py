"""
@File : test_activations.py
@Author: Dong Wang
@Date : 2020/5/1
"""
import pytest
import numpy as np
from SemiFlow.activations import Sigmoid, Relu, Tanh, Softplus, Softmax


def test_sigmoid():
    SIGMOID = Sigmoid()
    M = SIGMOID.ForwardPropagation(np.array([[3, -1], [0, 2]]))
    N = SIGMOID.BackwardPropagation(np.array([[1., -2.], [-7, 0.2]]))
    assert round(M[0][0],
                 3) == 0.953 and round(M[0][1],
                                       3) == 0.269 and round(M[1][0],
                                                             3) == 0.5 and round(M[1][1],
                                                                                 3) == 0.881
    assert round(N[0][0],
                 3) == 0.197 and round(N[0][1],
                                       3) == 0.105 and round(N[1][0],
                                                             3) == 0.001 and round(N[1][1],
                                                                                   3) == 0.248


def test_Relu():
    RELU = Relu()
    M = RELU.ForwardPropagation(np.array([3., -1.]))
    N = RELU.BackwardPropagation(np.array([1., -2.]))
    assert M[0] == 3 and M[1] == 0
    assert N[0] == 1 and N[1] == 0


def test_Tanh():
    TANH = Tanh()
    M = TANH.ForwardPropagation(np.array([[3, -1.], [0, 2.]]))
    N = TANH.BackwardPropagation(np.array([[1., -2.], [-7., 0.2]]))
    # print('\n', M)
    # print('\n', N)
    assert round(M[0][0],
                 3) == 0.995 and round(M[0][1],
                                       3) == -0.762 and round(M[1][0],
                                                              3) == 0 and round(M[1][1],
                                                                                3) == 0.964
    assert round(N[0][0],
                 3) == round(4.19974342e-01, 3) and round(N[0][1],
                                                          3) == round(7.06508249e-02, 3) and round(N[1][0],
                                                                                                   3) == round(
        3.32610934e-06, 3) and round(N[1][1],
                                     3) == round(9.61042983e-01, 3)


def test_Softplus():
    SOFTPLUS = Softplus()
    M = SOFTPLUS.ForwardPropagation(np.array([[3, -1.], [0, 2.]]))
    N = SOFTPLUS.BackwardPropagation(np.array([[1., -2.], [-7., 0.2]]))
    # print('\n', M)
    # print('\n', N)
    assert round(M[0][0],
                 3) == 3.049 and round(M[0][1],
                                       3) == 0.313 and round(M[1][0],
                                                             3) == 0.693 and round(M[1][1],
                                                                                   3) == 2.127
    assert round(N[0][0],
                 3) == 0.269 and round(N[0][1],
                                       3) == 0.881 and round(N[1][0],
                                                             3) == 0.999 and round(N[1][1],
                                                                                   3) == 0.450


def test_Softmax():
    x = np.array([[1., 2., 3.],
                  [1., 4., 9.]])
    SOFTMAX = Softmax()
    M = SOFTMAX.ForwardPropagation(x)
    N = SOFTMAX.BackwardPropagation(np.array([[1., -2.], [-7., 0.2]]))
    print('\n', M)
    print('\n', N)
