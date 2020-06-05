"""
@File : initializers.py
@Author: Dong Wang
@Date : 2020/4/30
"""

from .engine.core import backend
import six


def getInitializer(init):
    if isinstance(init, six.string_types):
        # TODO Return initializers
        pass
    else:
        ValueError('Could not interpret '
                   'initializer:', init)


class Initializer(object):
    """Initializer base class: all initializers inherit from this class.
    """

    def __call__(self, shape, dtype=None):
        raise NotImplementedError


class zeros(Initializer):
    """Initializer that generates tensors initialized to 0.
    """

    def __call__(self, shape, dtype=None):
        # TODO initializers.zeros
        pass


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1.
    """

    def __call__(self, shape, dtype=None):
        # TODO initializers.ones
        pass

# TODO initializers.glorot_uniform
