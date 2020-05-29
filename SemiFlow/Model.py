"""
@File : Model.py
@Author: Dong Wang
@Date : 2020/3/31
"""
from .engine.core import backend
from .layer.core import Layer
from .utils import DataShuffle, BatchSpliter, split_train_val
from .optimizer import getOptimizer


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
        super(Sequential, self).__init__()
        self.layers = []
        self.optimizer = None
        self.isComplied = False

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,  # TODO Model.Sequential.fit.verbose
            callbacks=None,
            validation_split=0.,
            validation_data=None,  # TODO support other validation data.
            shuffle=True,
            **kwargs):
        assert self.isComplied, "Model should be compiled before fit"

        x_raw = x
        y_raw = y

        if shuffle:
            x, y = DataShuffle(x, y)

        if validation_split != 0:
            x_train, y_train, x_val, y_val = split_train_val(x, y, validation_split)
        else:
            x_train, y_train = x, y

        self._train(x_train, y_train, epochs, batch_size)

    def _train(self, x, y, epochs, batch_size):
        # self.loss, self.optimizer, layers, layers.layer.param
        # Note self.optimizer manages the training process
        self.optimizer.build(x, y, epochs, batch_size)

    def evaluate(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 verbose=1,
                 **kwargs):
        # TODO Model.Sequential.evaluate
        pass

    def predict(self, **kwargs):
        # TODO Model.Sequential.predict
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
        # TODO Model.Sequential.compile.
        if not optimizer:
            # TODO implement the default optimizer
            optimizer = 'Default'
        if not loss:
            raise ValueError("loss is needed")

        # Optimizer
        self.optimizer = getOptimizer(optimizer, loss=loss)

        # Something about optimizer and loss
        # Init params

        # Add loss as the last layer. But it is not a layer.

        # _init_params
        # from loss-layer apply _get_prerequisite postorder_traverse

        self.isComplied = True

    def add(self, layer):
        """add a layer to the model

        Args:
            layer: a instance of layer

        """
        if not isinstance(layer, Layer):
            raise TypeError('Wrong layer type')
        layer.name = layer.__class__.__name__ + str(len(self.layers))
        if len(self.layers):
            L = self.layers[-1]
            L.outbound.append(layer)
            layer.inbound.append(L)
        # For sequential model, new layer will be added to the array.
        self.layers.append(layer)

    def summary(self):
        print("\n" + 20 * "=")
        for layer in self.layers:
            print(layer.name)
            print(20 * "-")
        print(20 * "=")
