"""
@File : test_activations.py
@Author: Dong Wang
@Date : 2020/5/1
"""
import pytest

from SemiFlow.activations import Sigmoid, Relu, Tanh, Softplus


def test_sigmoid():
    SIGMOID = Sigmoid()
    M = SIGMOID.ForwardPropagation(1)
    N = SIGMOID.BackwardPropagation(1)


def test_Relu():
    RELU = Relu()
    M = RELU.ForwardPropagation(1)
    N = RELU.BackwardPropagation(1)


def test_Tanh():
    TANH = Tanh()
    M = TANH.ForwardPropagation(1)
    N = TANH.BackwardPropagation(1)


def test_Softplus():
    SOFTPLUS = Softplus()
    M = SOFTPLUS.ForwardPropagation(1)
    N = SOFTPLUS.BackwardPropagation(1)
