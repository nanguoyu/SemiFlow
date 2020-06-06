"""
@File : losses.py
@Author: Dong Wang
@Date : 2020/6/5
"""
import six

from .engine.core import backend
from .Model import Layer


class Loss(Layer):

    def __init__(self, fn, name=None, **kwargs):
        super(Loss, self).__init__()
        self.fn = fn
        self.name = name
        self._fn_kwargs = kwargs

    def ForwardPropagation(self, y_true):
        x, = self.inbound
        self.output_value = self.fn(y_true, x.output_value, self._fn_kwargs)
        return self.output_value

    def BackwardPropagation(self, **kwargs):
        raise NotImplementedError


# Loss class

class MeanSquaredError(Loss):
    def __init__(self, name='mean_squared_error'):
        super(MeanSquaredError, self).__init__(
            fn=mean_squared_error,
            name=name)

    def BackwardPropagation(self, **kwargs):
        pass


class MeanAbsoluteError(Loss):
    def __init__(self, name='mean_absolute_error'):
        super(MeanAbsoluteError, self).__init__(
            fn=mean_absolute_error,
            name=name)

    def BackwardPropagation(self, **kwargs):
        pass


class BinaryCrossentropy(Loss):
    def __init__(self,
                 from_logits=False,
                 label_smoothing=0,
                 name='binary_crossentropy'):
        super(BinaryCrossentropy, self).__init__(
            fn=binary_crossentropy,
            name=name,
            from_logits=from_logits,
            label_smoothing=label_smoothing)

    def BackwardPropagation(self, **kwargs):
        pass


class CategoricalCrossentropy(Loss):

    def __init__(self,
                 label_smoothing=0,
                 name='categorical_crossentropy'):
        super(CategoricalCrossentropy, self).__init__(
            fn=categorical_crossentropy,
            name=name,
            label_smoothing=label_smoothing)

    def BackwardPropagation(self, **kwargs):
        pass


# Loss function

def mean_squared_error(y_true, y_pred):
    assert y_true.shape[0] == y_pred.shape[0], "wrong shape"
    return backend.mean(backend.square(y_pred - y_true), axis=-1)


def mean_absolute_error(y_true, y_pred):
    assert y_true.shape[0] == y_pred.shape[0], "wrong shape"
    return backend.mean(backend.abs(y_pred - y_true), axis=-1)


def binary_crossentropy(y_true, y_pred):
    assert y_true.shape[0] == y_pred.shape[0], "wrong shape"
    # TODO correct binary_crossentropy
    loss = 0
    for i in range(y_true.shape[0]):
        loss += - (y_true[i] * backend.log(y_pred[i]) + (1 - y_true[i]) * backend.log(1 - y_pred[i]))
    return loss / y_true.shape[0]


def categorical_crossentropy(y_true, y_pred, label_smoothing=0):
    """
    Computes the crossentropy loss between the labels and predictions.
    Args:
        y_true: labels
        y_pred: prediction value after activation function
        label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
                        meaning the confidence on label values are relaxed

    Returns:
    """
    assert y_true.shape[0] == y_pred.shape[0], "wrong shape"

    if label_smoothing != 0:
        # TODO smooth label
        # https://towardsdatascience.com/label-smoothing-making-model-robust-to-incorrect-labels-2fae037ffbd0
        # y_true = y_true*(1-label_smoothing) + label_smoothing/num_classes
        pass
    # Resize y_true to (max min), avoiding zero
    epsilon = backend.finfo(backend.float32).eps
    y_pred[y_pred > 1 - epsilon] = 1 - epsilon
    y_pred[y_pred < epsilon] = epsilon
    return -backend.sum(y_true * backend.log(y_pred), axis=-1)


def getLoss(loss):
    if isinstance(loss, six.string_types):
        if loss.lower() == 'mse':
            return MeanSquaredError()
        elif loss.lower() == 'mae':
            return MeanAbsoluteError()
        elif loss.lower() == 'binary_crossentropy':
            return BinaryCrossentropy()
        elif loss.lower() == 'categorical_crossentropy':
            return CategoricalCrossentropy()
        else:
            raise ValueError('Could not support ', loss, ' now')
    else:
        raise ValueError('Could not interpret '
                         'loss:', loss)
