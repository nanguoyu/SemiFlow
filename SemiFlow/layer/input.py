"""
@File : input.py
@Author: Dong Wang
@Date : 2020/6/7
"""
from ..engine.core import backend
from .core import Layer


class InputLayer(Layer):
    def __init__(self,
                 dtype=None,
                 shape=None,
                 **kwargs):
        """
        Args:
            units: number of neural, dimensionality of output
            activation: activation function
                If you don't specify anything linear activation is applied
            kernel_initializer: Initializer for the `kernel` weights matrix
            bias_initializer: Initializer for the bias vector
        """
        super(InputLayer, self).__init__(**kwargs)
        self.dtype = dtype
        self.shape = shape

    def BackwardPropagation(self, grad=None):
        """
        dc = grad*0
        Args:
            grad: gradients from output layer

        Returns: gradients of this layer

        """

        return 0

    def ForwardPropagation(self, feed=None):
        """
        feed: input data
        Returns:output_value

        """
        # TODO check feed.shape
        if len(list(feed.shape)) == 1 and len(self.shape) == 1:
            if list(feed.shape) == self.shape:
                self.output_value = feed
            else:
                raise ValueError(
                    " expect shape: ", self.shape,
                    " but get ", list(feed.shape[0]))
        else:
            if list(feed.shape[1:]) == self.shape:
                self.output_value = feed
            else:
                raise ValueError(
                    " expect shape: ", self.shape,
                    " but get ", list(feed.shape[1:]))
        if hasattr(self, 'dtype'):
            self.output_value = self.output_value.astype(self.dtype)
        return self.output_value
