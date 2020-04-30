"""
@File : initializers.py
@Author: Dong Wang
@Date : 2020/4/30
"""

from .engine.core import backend
import six


def get(init):
    if isinstance(init, six.string_types):
        # TODO Return initializers
        pass
    else:
        ValueError('Could not interpret '
                   'initializer:', init)
