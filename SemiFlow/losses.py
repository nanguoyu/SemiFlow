"""
@File : losses.py
@Author: Dong Wang
@Date : 2020/6/5
"""
import six

from .engine.core import backend
from .Model import Layer


class Loss(Layer):

    def __init__(self, fn, name='Loss', **kwargs):
        super(Loss, self).__init__()
        self.fn = fn
        self.name = name
        self._fn_kwargs = kwargs
        self.input_value = None
        self.y_true = None
        self.isLoss = True  # To distinguish layer and loss. It maybe redundant.

    def ForwardPropagation(self, y_true):
        x = self.inbound[0]
        self.input_value = x.output_value
        self.output_value = self.fn(y_true, self.input_value, **self._fn_kwargs)
        self.y_true = y_true
        self.output_value /= y_true.shape[0]
        return self.output_value

    def BackwardPropagation(self, **kwargs):
        raise NotImplementedError


# Loss class

class MeanSquaredError(Loss):
    def __init__(self, name='mean_squared_error'):
        super(MeanSquaredError, self).__init__(
            fn=mean_squared_error,
            name=name)

    def BackwardPropagation(self, grads=None, **kwargs):
        # Todo Check MAE.BP
        if not grads:
            grads = backend.ones_like(self.output_value)
        return grads * (self.input_value - self.y_true) / self.y_true.shape[0]


class MeanAbsoluteError(Loss):
    def __init__(self, name='mean_absolute_error'):
        super(MeanAbsoluteError, self).__init__(
            fn=mean_absolute_error,
            name=name)

    def BackwardPropagation(self, grads=None, **kwargs):
        if not grads:
            grads = backend.ones_like(self.output_value)
        return grads * backend.sign(self.input_value - self.y_true) / self.y_true.shape[0]


class BinaryCrossentropy(Loss):
    def __init__(self,
                 label_smoothing=0,
                 name='binary_crossentropy'):
        super(BinaryCrossentropy, self).__init__(
            fn=binary_crossentropy,
            name=name,
            label_smoothing=label_smoothing)

    def BackwardPropagation(self, grads=None, **kwargs):
        pass


class CategoricalCrossentropy(Loss):

    def __init__(self,
                 label_smoothing=0,
                 name='categorical_crossentropy'):
        super(CategoricalCrossentropy, self).__init__(
            fn=categorical_crossentropy,
            name=name,
            label_smoothing=label_smoothing)

    def BackwardPropagation(self, grads=None, **kwargs):
        print("This is BP of CE ", self.input_value.shape)
        if not grads:
            grads = backend.ones_like(self.output_value)
        # Todo CategoricalCrossentropy.BackwardPropagation
        return grads * backend.multiply(self.y_true, 1 / self.input_value) / self.y_true.shape[0]


class SoftmaxCategoricalCrossentropy(Loss):
    """
    If the activation function of layer is softmax and loss function is CategoricalCrossentropy,
    Linear() will replace it. SoftmaxCategoricalCrossentropy will become loss function.

    """

    def __init__(self, name="softmax_categorical_crossentropy"):
        super(SoftmaxCategoricalCrossentropy, self).__init__(
            fn=softmax_categorical_crossentropy,
            name=name)

    def ForwardPropagation(self, y_true):
        # print("FP:", self.name)
        x = self.inbound[0]
        self.input_value = x.output_value
        self.output_value = self.fn(y_true, self.input_value, **self._fn_kwargs)
        self.y_true = y_true
        self.output_value /= y_true.shape[0]
        return self.output_value

    def BackwardPropagation(self, grads=None, **kwargs):
        """
        Returns: shape: the number of data * the number of parameters
        """
        # print("BP:", self.name)
        if not grads:
            grads = backend.ones_like(self.output_value)
        # return grads * (self.y_pred - self.y_true) / self.y_true.shape[0]
        return grads * (softmax2(self.input_value, t=1) - self.y_true) / self.y_true.shape[0]


# Loss function

def mean_squared_error(y_true, y_pred):
    assert y_true.shape[0] == y_pred.shape[0], "wrong shape"
    return backend.mean(backend.square(y_pred - y_true), axis=0)


def mean_absolute_error(y_true, y_pred):
    assert y_true.shape[0] == y_pred.shape[0], "wrong shape"
    return backend.mean(backend.abs(y_pred - y_true), axis=0)


def binary_crossentropy(y_true, y_pred, label_smoothing=0):
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
    return backend.sum(
        -backend.sum(y_true * backend.log(y_pred) + (1 - y_true) * backend.log(1 - y_pred), axis=-1),
        axis=0)


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
    return backend.sum(-backend.sum(y_true * backend.log(y_pred), axis=-1), axis=0)


def softmax_categorical_crossentropy(y_true, logits, axis=-1):
    """softmax categorical crossentropy function

    Args:
        y_true:
        logits:
        axis:

    Returns:

    """
    x = logits
    x_max = backend.max(x, axis=axis, keepdims=True)
    exps = backend.exp(x - x_max)
    y_pred = exps / backend.sum(exps, axis=axis, keepdims=True)
    epsilon = backend.finfo(backend.float32).eps
    y_pred[y_pred > 1 - epsilon] = 1 - epsilon
    y_pred[y_pred < epsilon] = epsilon
    return backend.sum(-backend.sum(y_true * backend.log(y_pred), axis=-1), axis=0) / y_pred.shape[0]
    # nll = -(log_softmax(logits, t=1, axis=1) * y_true).sum(axis=-1)
    # return backend.sum(nll, axis=0) / y_true.shape[0]


def get(loss):
    if not loss:
        raise ValueError('You should specify a loss function')
    if isinstance(loss, six.string_types):
        return searchLoss(loss_str=loss)
    elif callable(loss):
        return loss
    else:
        raise ValueError('Could not interpret '
                         'loss:', loss)


def searchLoss(loss_str: str):
    loss_str = loss_str.lower()
    if loss_str == 'mse':
        return MeanSquaredError()
    elif loss_str == 'mae':
        return MeanAbsoluteError()
    elif loss_str == 'binary_crossentropy':
        return BinaryCrossentropy()
    elif loss_str == 'categorical_crossentropy':
        return CategoricalCrossentropy()
    elif loss_str == 'softmax_categorical_crossentropy':
        return SoftmaxCategoricalCrossentropy()
    else:
        raise ValueError('Could not support ', loss_str)


def log_softmax(x, t=1.0, axis=-1):
    x_ = x / t
    x_max = backend.max(x_, axis=axis, keepdims=True)
    exps = backend.exp(x_ - x_max)
    exp_sum = backend.sum(exps, axis=axis, keepdims=True)
    return x_ - x_max - backend.log(exp_sum)


def softmax2(x, t=1.0, axis=-1):
    x_ = x / t
    x_max = backend.max(x_, axis=axis, keepdims=True)
    exps = backend.exp(x_ - x_max)
    return exps / backend.sum(exps, axis=axis, keepdims=True)
