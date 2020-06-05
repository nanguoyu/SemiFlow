"""
@File : losses.py
@Author: Dong Wang
@Date : 2020/6/5
"""
from .engine.core import backend


def mean_squared_error(y_true, y_pred):
    if not backend.is_tensor(y_pred):
        y_pred = backend.constant(y_pred)
    y_true = backend.cast(y_true, y_pred.dtype)
    return backend.mean(backend.square(y_pred - y_true), axis=-1)


def mean_absolute_error(y_true, y_pred):
    if not backend.is_tensor(y_pred):
        y_pred = backend.constant(y_pred)
    y_true = backend.cast(y_true, y_pred.dtype)
    return backend.mean(backend.abs(y_pred - y_true), axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    if not backend.is_tensor(y_pred):
        y_pred = backend.constant(y_pred)
    y_true = backend.cast(y_true, y_pred.dtype)
    first_log = backend.log(backend.clip(y_pred, backend.epsilon(), None) + 1.)
    second_log = backend.log(backend.clip(y_true, backend.epsilon(), None) + 1.)
    return backend.mean(backend.square(first_log - second_log), axis=-1)


def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = backend.constant(y_pred) if not backend.is_tensor(y_pred) else y_pred
    y_true = backend.cast(y_true, y_pred.dtype)
    if label_smoothing is not 0:
        smoothing = backend.cast_to_floatx(label_smoothing)
        y_true = backend.switch(backend.greater(smoothing, 0),
                                lambda: y_true * (1.0 - smoothing) + 0.5 * smoothing,
                                lambda: y_true)
    return backend.mean(
        backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1)
