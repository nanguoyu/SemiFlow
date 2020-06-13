"""
@File : test_activations.py
@Author: Dong Wang
@Date : 2020/5/1
"""
import pytest
import numpy as np
from SemiFlow import activations


def test_None():
    NONE = activations.get(None)
    M = NONE.ForwardPropagation(np.array([[3, -1], [0, 2]]))
    N = NONE.BackwardPropagation(np.array([[1., -2.], [-7, 0.2]]))
    print('\n', M)
    print('\n', N)


def test_linear():
    LINEAR = activations.get('linear')
    M = LINEAR.ForwardPropagation(np.array([[3, -1], [0, 2]]))
    N = LINEAR.BackwardPropagation(np.array([[1., -2.], [-7, 0.2]]))
    print('\n', M)
    print('\n', N)


def test_sigmoid():
    SIGMOID = activations.get('Sigmoid')
    M = SIGMOID.ForwardPropagation(np.array([[3, -1], [0, 2]]))
    N = SIGMOID.BackwardPropagation(np.array([[1., -2.], [-7, 0.2]]))
    # print('\n', M)
    # print('\n', N)
    assert round(M[0][0],
                 3) == 0.953 and round(M[0][1],
                                       3) == 0.269 and round(M[1][0],
                                                             3) == 0.5 and round(M[1][1],
                                                                                 3) == 0.881
    assert round(N[0][0],
                 3) == round(-6, 3) and round(N[0][1],
                                              3) == round(4, 3) and round(N[1][0],
                                                                          3) == round(0, 3) and round(N[1][1],
                                                                                                      3) == round(-0.4,
                                                                                                                  3)


def test_Relu():
    RELU = activations.get('Relu')
    M = RELU.ForwardPropagation(np.array([3., -1.]))
    N = RELU.BackwardPropagation(np.array([1., -2.]))
    assert M[0] == 3 and M[1] == 0
    assert N[0] == 1 and N[1] == 0


def test_Tanh():
    TANH = activations.get('Tanh')
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
                 3) == round(-8, 3) and round(N[0][1],
                                              3) == round(0, 3) and round(N[1][0],
                                                                          3) == round(
        -7, 3) and round(N[1][1],
                         3) == round(-0.6, 3)


def test_Softplus():
    SOFTPLUS = activations.get('Softplus')
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
                 3) == round(0.04742587, 3) and round(N[0][1],
                                                      3) == round(-1.46211716, 3) and round(N[1][0],
                                                                                            3) == round(-3.5,
                                                                                                        3) and round(
        N[1][1],
        3) == round(0.02384058, 3)


def test_Softmax():
    x = np.array([[1., 2., 3.],
                  [1., 4., 9.]])
    SOFTMAX = activations.get('Softmax')
    M = SOFTMAX.ForwardPropagation(x)
    N = SOFTMAX.BackwardPropagation(grads=np.array([[1., 1.]]))
    # print('\n', M)
    # print('\n', N)
    assert round(M[0][0], 3) == round(9.00305732e-02, 3)
    assert round(M[0][1], 3) == round(2.44728471e-01, 3)
    assert round(M[0][2], 3) == round(6.65240956e-01, 3)
    assert round(M[1][0], 3) == round(3.33106430e-04, 3)
    assert round(M[1][1], 3) == round(6.69062149e-03, 3)
    assert round(M[1][2], 3) == round(9.92976272e-01, 3)
