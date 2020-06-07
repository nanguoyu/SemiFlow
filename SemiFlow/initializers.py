"""
@File : initializers.py
@Author: Dong Wang
@Date : 2020/4/30
"""

from .engine.core import backend
import six


def searchInit(init_str: str, **kwargs):
    init_str = init_str.lower()
    if init_str == 'zeros':
        return Zeros()
    elif init_str == 'ones':
        return Ones()
    elif init_str == 'glorot_uniform':
        raise ValueError("glorot_uniform will be supported in the future")
        # return GlorotUniform()
    elif init_str == 'random_normal':
        return RandomNormal(**kwargs)
    elif init_str == 'random_uniform':
        return RandomUniform(**kwargs)
    else:
        raise ValueError('Could not support ', init_str)


def get(init, **kwargs):
    if not init:
        init = 'random_normal'
    if isinstance(init, six.string_types):
        return searchInit(init_str=init, **kwargs)
    elif callable(init):
        return init
    else:
        raise ValueError('Could not interpret '
                         'init:')


class Initializer(object):
    """Initializer base class: all initializers inherit from this class.
    """

    def __call__(self, shape, dtype=None):
        raise NotImplementedError


class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.
    """

    def __call__(self, shape, dtype=None):
        return backend.zeros(shape)


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1.
    """

    def __call__(self, shape, dtype=None):
        return backend.ones(shape)


class RandomNormal(Initializer):
    """Initializer that generates tensors with a normal distribution.
    """

    def __init__(self, mean=0., stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape, dtype=None):
        backend.random.seed(self.seed)
        x = backend.random.normal(self.mean, self.stddev, shape)
        if self.seed is not None:
            self.seed += 1
        return x


class RandomUniform(Initializer):
    """Initializer that generates tensors with a uniform distribution.

    # Arguments
        minval: A python scalar or a scalar tensor. Lower bound of the range
          of random values to generate.
        maxval: A python scalar or a scalar tensor. Upper bound of the range
          of random values to generate.  Defaults to 1 for float types.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape, dtype=None):
        backend.random.seed(self.seed)
        x = backend.random.uniform(self.minval, self.maxval, shape)
        if self.seed is not None:
            self.seed += 1
        return x


# TODO initializers.glorot_uniform
class GlorotUniform(Initializer):
    def __call__(self, shape, dtype=None):
        raise NotImplementedError
