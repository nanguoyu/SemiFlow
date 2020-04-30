"""
@File : Model.py
@Author: Dong Wang
@Date : 2020/3/31
"""
from .engine.core import backend


class Model(object):
    def __init__(self):
        """Model constructor
        """
        pass

    def fit(self, **kwargs):
        """train the model by epoch learning rule
        """
        raise NotImplementedError

    def predict(self, **kwargs):
        """test a data point and return the result
        """
        raise NotImplementedError

    def evaluate(self, **kwargs):
        """evaluate test data and print metrics
        """
        raise NotImplementedError

    def compile(self, **kwargs):
        """build model
        """
        raise NotImplementedError


class Sequential(Model):
    def __init__(self):
        self.layers = []
        super(Sequential, self).__init__()

    def evaluate(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 verbose=1,  # TODO Model.Sequential.evaluate.verbose
                 **kwargs):
        # TODO Model.Sequential.evaluate
        pass

    def predict(self, **kwargs):
        # TODO Model.Sequential.predict
        pass

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,  # TODO Model.Sequential.fit.verbose
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            steps_per_epoch=None,
            **kwargs):
        # TODO Model.Sequential.train
        pass

    def compile(self,
                loss=None,
                optimizer=None,
                **kwargs):
        """
        Args:
            loss: loss function
            optimizer: learning method
        """
        # TODO Model.Sequential.compile
        pass

    def add(self, layer):
        """add a layer to the model

        Args:
            layer:

        Returns:

        """
        # TODO Model.Sequential.add
        pass

    def summary(self):
        # TODO Model.Sequential.summary
        pass
