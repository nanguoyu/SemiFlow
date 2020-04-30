"""
@File : activations.py
@Author: Dong Wang
@Date : 2020/4/30
"""
from .engine.core import backend
import six


def get(act):
    if not act:
        # TODO Return Linear
        pass
    elif isinstance(act, six.string_types):
        # TODO Return activation function
        pass
    else:
        ValueError('Could not interpret '
                   'activation function:', act)
