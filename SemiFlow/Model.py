"""
@File : Model.py
@Author: Dong Wang
@Date : 2020/3/31
"""
from .engine.core import backend
from .layer.core import Layer
from .layer.input import InputLayer
from .utils import DataShuffle, split_train_val
from . import optimizers


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
        self.input_layer = None
        self.first_layer = None  # To infer dtype and inputs
        self.last_layer = None  # outputLayer is used for tracking loss
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
        assert x is not None and y is not None, "X_train and Y_train are needed"

        if shuffle:
            x, y = DataShuffle(x, y)

        if validation_data is not None:
            x_train, y_train = x, y
            x_val, y_val = validation_data[0], validation_data[1]
        else:
            if validation_split != 0:
                x_train, y_train, x_val, y_val = split_train_val(x, y, validation_split)
            else:
                x_train, y_train = x, y

        self._train(x_train, y_train, epochs, batch_size)

    def _train(self, x, y, epochs, batch_size):
        # Note self.optimizer manages the training process
        self.optimizer.build(x, y, epochs, batch_size, self.first_layer, self.last_layer)
        self.optimizer.ForwardPropagation()

    def compile(self,
                loss=None,
                optimizer=None,
                **kwargs):
        """

        Args:
            loss: loss function
            optimizer: learning method

        """
        if not optimizer:
            optimizer = 'sgd'
        if not loss:
            raise ValueError("loss is needed")

        # Optimizer
        self.optimizer = optimizers.get(optimizer, loss=loss)
        # add Input_layer
        if hasattr(self.first_layer, "input_shape"):
            shape = self.first_layer.input_shape
        else:
            raise ValueError("You should specify the input shape")
        self.input_layer = InputLayer(dtype=None, shape=shape)
        self.input_layer.name = self.input_layer.__class__.__name__
        self.first_layer.inbound.append(self.input_layer)
        self.input_layer.outbound.append(self.first_layer)
        # Init_parameters
        layer = self.first_layer
        while layer:
            layer.InitParams()
            if not layer.outbound:
                break
            layer = layer.outbound[0]
        self.isComplied = True

    def add(self, layer):
        """add a layer to the model

        Args:
            layer: a instance of layer

        """
        if not isinstance(layer, Layer):
            raise TypeError('Wrong layer type')
        layer.name = layer.__class__.__name__ + str(len(self.layers))
        if not self.first_layer:
            self.first_layer = layer
            self.last_layer = layer
        else:
            self.last_layer.outbound.append(layer)
            layer.inbound.append(self.last_layer)
            self.last_layer = layer
        # For sequential model, new layer will be added to the array.
        self.layers.append(layer)

    def summary(self):
        print("\n" + 20 * "=")
        layer = self.first_layer
        while layer:
            print(layer.name)
            print(20 * "-")
            if not layer.outbound:
                break
            layer = layer.outbound[0]
        print(20 * "=")

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
