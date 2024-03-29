"""
@File : Model.py
@Author: Dong Wang
@Date : 2020/3/31
"""
from .engine.core import backend
from .layer.core import Layer
from .layer.input import InputLayer
from .activations import Activation
from .losses import Loss
from .utils import DataShuffle, split_train_val
from . import optimizers
import collections
import copy


class Model(object):
    def __init__(self):
        """Abstract model constructor
        """
        self.first_layer = None  # To infer dtype and inputs
        self.last_layer = None  # outputLayer is used for tracking loss
        self.isComplied = False

    def fit(self, **kwargs):
        """Train the model by epoch learning rule
        """
        raise NotImplementedError

    def predict(self, **kwargs):
        """Predict the result of data points and return the results
        """
        raise NotImplementedError

    def evaluate(self, **kwargs):
        """Evaluate test data and print metrics
        """
        raise NotImplementedError

    def compile(self, **kwargs):
        """Build model
        """
        raise NotImplementedError

    def save(self, **kwargs):
        """Save model weights
        """

        raise NotImplementedError

    def load(self, **kwargs):
        """Load model weights
        """

        raise NotImplementedError

    @property
    def parameters(self):
        """obtain the current parameters in numpy.ndarray"""
        raise NotImplementedError

    @parameters.setter
    def parameters(self, **kwargs):
        raise NotImplementedError

    @property
    def state_dict(self):
        """Obtain the current parameters in Python Dict"""
        raise NotImplementedError


class Sequential(Model):
    """Sequential model is a kind model that each layer is followed by one layer.
     # Example

    ```python
    # Optionally, the first layer can receive an `input_shape` argument:
    model = Sequential()
    model.add(Dense(32, input_shape=(500,)))

    # Afterwards, we do automatic shape inference:
    model.add(Dense(32))

    # This builds the model for the first time:
    model.fit(x, y, batch_size=32, epochs=10)

    ```
    """

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
            batch_size=16,
            epochs=1,
            verbose=1,  # TODO Model.Sequential.fit.verbose
            callbacks=None,
            validation_split=0.,
            validation_data=None,  # TODO support other validation data.
            shuffle=True,
            **kwargs):
        """Train the model for a fixed number of epochs (iterations on a dataset).

        Note: model.compile() should be called before model.fit().

        Args:
            x: Input data. It should be a Numpy array.
            y: Target data. It should be a Numpy array.
            batch_size: Integer. The number of a group samples per gradient update. The default value is 16.
            epochs: Integer. Number of epochs to train the model. After an epoch, x and y would be visited.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of callback instances to apply during training and validation.
            validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data.
            validation_data: Validation data to evaluate model metrics at the end of each epoch.
            shuffle: Boolean. Whether to shuffle the training data.
            **kwargs: More parameters will be supported.

        Returns:
            A history object which is a record of training loss and other metrics values.

        """
        assert self.isComplied, "Model should be compiled before fit"
        assert x is not None and y is not None, "X_train and Y_train are needed"

        if shuffle:
            x, y = DataShuffle(x, y)

        if validation_data is not None and validation_data[0] is not None and validation_data[1] is not None:
            x_train, y_train = x, y
            x_val, y_val = validation_data[0], validation_data[1]
        else:
            if validation_split != 0:
                x_train, y_train, x_val, y_val = split_train_val(x, y, validation_split)
            else:
                x_train, y_train = x, y
                x_val, y_val = None, None

        history = self._train(x_train, y_train, x_val, y_val, epochs, batch_size)
        return history

    def _train(self, x_train, y_train, x_val, y_val, epochs, batch_size):
        """Protected training function
        Args:
            x_train: Input data in training.
            y_train: Target data in training.
            x_val: Input data in validation.
            y_val: Target data in validation.
            epochs: Number of epochs to train the model.
            batch_size: The number of a group samples per gradient update.

        """
        self.optimizer.ForwardPropagation(x_train, y_train, x_val, y_val, epochs, batch_size)
        return self.optimizer.GetHistory()

    def compile(self,
                loss=None,
                optimizer=None,
                learning_rate=None,
                metrics=None,  # Todo: implement metrics
                **kwargs):
        """Compile function used before fit
        Args:
            loss: loss function to measure the distance between prediction of model and ground truth.
            optimizer: learning rule deciding how to update gradients. The default one is S
            learning_rate: Float. The default learning rate is 0.005.
            metrics: metrics recorded during training and validation

        """
        if not optimizer:
            optimizer = 'sgd'
        if not loss:
            raise ValueError("loss is needed")

        # Optimizer
        self.optimizer = optimizers.get(optimizer, loss=loss, learning_rate=learning_rate, metrics=metrics)
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
        self.optimizer.build(self.first_layer, self.last_layer)
        self.isComplied = True

    def add(self, layer):
        """Add a layer to the model

        Args:
            layer: an instance of layer

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
        """Print out model structure"""
        print("\n" + 20 * "=")
        layer = self.first_layer
        while layer and not isinstance(layer, InputLayer) and not isinstance(layer, Activation):
            print(layer.name)
            print(20 * "-")
            if not layer.outbound:
                break
            layer = layer.outbound[0]
        print(20 * "=")

    @property
    def parameters(self):
        weights = collections.OrderedDict()
        layer = self.first_layer
        while layer and not isinstance(layer, InputLayer) and not isinstance(layer, Activation) \
                and not isinstance(layer, Loss):
            name = layer.name
            parameters = layer.params
            weights[name] = parameters
            if not layer.outbound:
                break
            layer = layer.outbound[0]

        return weights

    @parameters.setter
    def parameters(self, weights):
        layer = self.first_layer
        while layer and not isinstance(layer, InputLayer) and not isinstance(layer, Activation) \
                and not isinstance(layer, Loss):
            name = layer.name
            loaded_layer = weights[name]
            keys = loaded_layer.keys()
            for key in keys:
                if isinstance(loaded_layer[key], list):
                    loaded_layer[key] = backend.array(loaded_layer[key])
                assert isinstance(loaded_layer[key], backend.ndarray)
                assert loaded_layer[key].shape == layer.params[key].shape
                layer.params[key] = loaded_layer[key]
            if not layer.outbound:
                break
            layer = layer.outbound[0]

    @property
    def state_dict(self):
        weights = collections.OrderedDict()
        layer = self.first_layer
        while layer and not isinstance(layer, InputLayer) and not isinstance(layer, Activation) \
                and not isinstance(layer, Loss):
            name = layer.name
            parameters = copy.deepcopy(layer.params)
            weights[name] = parameters
            for key in parameters.keys():
                weights[name][key] = parameters[key].tolist()
            if not layer.outbound:
                break
            layer = layer.outbound[0]

        return weights

    def save(self,
             path,
             **kwargs):
        """Save model weights

        Args:
            path: The path to save weight file
            **kwargs:
        """
        parameters = self.parameters
        for layer_name, parameters_dict in parameters.items():
            keys = parameters_dict.keys()
            for parameter_name in keys:
                parameters[layer_name][parameter_name] = backend.array(parameters[layer_name][parameter_name])
        weights = backend.array([parameters])

        backend.save(path, weights, allow_pickle=True)

    def load(self,
             path,
             **kwargs):
        """Load model weights

        Args:
            path: The path to load weight file
            **kwargs:
        """

        weights = backend.load(path, allow_pickle=True)[0]

        if not isinstance(weights, collections.OrderedDict):
            raise ValueError(f'{path} is not a weight file for SemiFlow')

        layer = self.first_layer
        while layer and not isinstance(layer, InputLayer) and not isinstance(layer, Activation) \
                and not isinstance(layer, Loss):
            name = layer.name
            loaded_layer = weights[name]
            keys = loaded_layer.keys()
            for key in keys:
                if isinstance(loaded_layer[key], list):
                    loaded_layer[key] = backend.array(loaded_layer[key])
                assert isinstance(loaded_layer[key], backend.ndarray)
                assert loaded_layer[key].shape == layer.params[key].shape
                layer.params[key] = loaded_layer[key]
            if not layer.outbound:
                break
            layer = layer.outbound[0]

    def evaluate(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 verbose=1,
                 **kwargs):
        """Returns the loss value & metrics values for the model in test mode.

        Computation is done in batches.

        # TODO Model.Sequential.evaluate
        """
        pass

    def predict(self, **kwargs):
        """Generates output predictions for the input samples.

        Computation is done in batches.

        # TODO Model.Sequential.predict
        """

        pass
