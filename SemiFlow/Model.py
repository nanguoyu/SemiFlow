"""
@File : Model.py
@Author: Dong Wang
@Date : 2020/3/31
"""


class Model(object):
    def __init__(self):
        """Model constructor
        """
        raise NotImplementedError

    def _ForwardPropagation(self):
        """Forward propagation of the Computational Graph
        """
        raise NotImplementedError

    def _BackPropagation(self):
        """Back propagation of the Computational Graph
        """
        raise NotImplementedError

    def train(self):
        """train the model by epoch learning rule
        """
        raise NotImplementedError

    def fit(self):
        """test a data point and return the result
        """
        raise NotImplementedError

    def evaluate(self):
        """evaluate test data and print metrics
        """
        raise NotImplementedError
