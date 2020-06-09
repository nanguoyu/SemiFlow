"""
@File : core.py
@Author: Dong Wang
@Date : 2020/4/30
"""
from ..engine import backend


class Layer(object):
    def __init__(self, **kwargs):
        """Layer constructor
        """
        self.name = None
        # These properties should be set by the user via keyword arguments.
        # note that 'dtype', 'input_shape' and 'batch_input_shape'
        # are only applicable to input layers: do not pass these keywords
        # to non-input layers.
        allowed_kwargs = {'input_shape',
                          'batch_input_shape',  # TODO Layer.init.batch_input_shape
                          'batch_size',
                          'dtype',  # TODO Layer.init.dtype
                          'name',
                          'trainable',  # TODO Layer.init.trainable
                          }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        if 'input_shape' in kwargs:
            if isinstance(kwargs['input_shape'], tuple):
                if len(kwargs['input_shape']) == 1:
                    self.input_shape = kwargs['input_shape'][0]
                else:
                    self.input_shape = tuple(kwargs['input_shape'])
            else:
                self.input_shape = kwargs['input_shape']
        if 'name' in kwargs:
            self.name = kwargs.get('name')

        # The input layer of this layer
        self.inbound = []
        # The output layer of this layer
        self.outbound = []
        # The shape of this layer
        self.shape = None
        # The params
        self.params = None
        # The output_value
        self.output_value = None

    def ForwardPropagation(self, **kwargs):
        """Forward propagation
        """
        raise NotImplementedError

    def BackwardPropagation(self, **kwargs):
        """Backward propagation
        """
        raise NotImplementedError

    def InitParams(self):
        """Initialize parameters"""
        raise NotImplementedError
