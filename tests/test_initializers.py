"""
@File : test_initializers.py
@Author: Dong Wang
@Date : 2020/6/7
"""
import numpy as np
from SemiFlow import initializers


def test_none():
    none = initializers.get(None)


def test_zeros():
    zeros = initializers.get('zeros')
    parameters = zeros((2, 3))
    assert (parameters.shape == (2, 3) and parameters.sum() == 0)


def test_ones():
    ones = initializers.get('ones')
    parameters = ones((2, 3))
    assert (parameters.shape == (2, 3) and parameters.sum() == 2 * 3)


def test_random_normal():
    mean = 0.
    stddev = 0.05
    random_normal = initializers.get('random_normal', mean=mean, stddev=stddev, seed=12)
    parameters = random_normal(shape=(2, 3))
    assert (parameters.shape == (2, 3)
            and np.abs(parameters.mean() - mean) <= 3 * stddev
            and np.abs(parameters.std() - stddev) <= 3 * stddev)


def test_random_uniform():
    minval = -0.05
    maxval = 0.05
    random_uniform = initializers.get('random_uniform', minval=minval, maxval=maxval, seed=12)
    parameters = random_uniform(shape=(2, 3))
    assert (parameters.shape == (2, 3)
            and np.abs(parameters.min() - minval) <= 0.1 * (maxval - minval)
            and np.abs(parameters.max() - maxval) <= 0.1 * (maxval - minval))
