"""
@File : pooling.py
@Author: Dong Wang
@Date : 2020/6/25
"""
from ..engine.core import backend
from .core import Layer


class MaxPooling2D(Layer):

    def __init__(self,
                 pooling_size=None,
                 strides=(1, 1),
                 padding='SAME',
                 **kwargs):
        """
        Args:
            pooling_size: An integer or tuple/list of 2 integers, specifying the
                height and width of the 2D pooling window.
                Can be a single integer to specify the same value for all spatial dimensions.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution
                along the height and width.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.
            padding: one of `"valid"` or `"same"` (case-insensitive).
        """
        super(MaxPooling2D, self).__init__(**kwargs)
        # pooling_size
        if isinstance(pooling_size, tuple):
            self.pooling_size = pooling_size
        elif isinstance(pooling_size, int):
            self.pooling_size = (pooling_size, pooling_size)
        # strides
        if isinstance(strides, tuple):
            self.strides = strides
        elif isinstance(strides, int):
            self.strides = (strides, strides)
        # padding
        if isinstance(padding, str):
            self.padding = padding.lower()
        else:
            raise TypeError("padding should be str")
        self.padding_width = None
        self.output_shape = None
        self.isInitialized = True

    def ForwardPropagation(self):
        assert self.isInitialized, "you should init_para"
        """ convolution operation
        A method to accelerate convolution from,
        [1] Kumar Chellapilla, Sidd Puri, Patrice Simard. High Performance Convolutional Neural Networks
        for Document Processing. Tenth International Workshop on Frontiers in Handwriting Recognition,
        Université de Rennes 1, Oct 2006, La Baule (France). ffinria-00112631f

        Returns:output

        """
        x = self.inbound[0]
        inputs = x.output_value
        inputs_padded = backend.pad(inputs, pad_width=self.padding_width, mode="constant")

        batch_size, input_h, input_w, input_channel = inputs.shape
        out_h, out_w, out_c = self.output_shape
        output_value = backend.zeros(shape=(batch_size, out_h, out_w, out_c))
        argmax = backend.empty(shape=(batch_size, out_h, out_w, out_c), dtype=int)
        for r in range(out_h):
            r_start = r * self.strides[0]
            for c in range(out_w):
                c_start = c * self.strides[1]
                pool = inputs_padded[:, r_start:r_start + self.pooling_size[0], c_start:c_start + self.pooling_size[1],
                       :]
                # Todo I dont understand why compute argmax. Is it about back-propagation?
                pool = pool.reshape((batch_size, -1, input_channel))
                _argmax = backend.argmax(pool, axis=1)[:, backend.newaxis, :]
                # Todo It may fail when input_channel = 1
                argmax[:, r, c, :] = _argmax.squeeze()

                _max_pool = backend.take_along_axis(pool, _argmax, axis=1).squeeze()
                output_value[:, r, c, :] = _max_pool

        self.inputs_padded_shape = inputs_padded.shape
        self.output_value = output_value
        self.argmax = argmax
        return self.output_value

    def BackwardPropagation(self, grad=None):
        """
        This function is modified from https://github.com/borgwang/tinynn/
        blob/58c1e76e90a4ff1ee671707595df9fbc1f84f963/tinynn/core/layer.py#L250

        Args:
            grad: gradients from output layer

        Returns: gradients of this layer

        """
        k_h, k_w = self.pooling_size
        s_h, s_w = self.strides
        batch_sz, in_h, in_w, in_c = self.inputs_padded_shape
        output_h, output_w = self.output_value.shape[1:3]
        pad_h, pad_w = self.padding_width[1:3]
        if grad is None:
            grad = backend.ones_like(self.output_value)

        grad_wrt_x = backend.zeros(self.inputs_padded_shape)
        for r in range(output_h):
            r_start = r * s_h
            for c in range(output_w):
                c_start = c * s_w
                _argmax = self.argmax[:, r, c, :]
                mask = backend.eye(k_h * k_w)[_argmax].transpose((0, 2, 1))
                _grad = grad[:, r, c, :][:, backend.newaxis, :]
                patch = backend.repeat(_grad, k_h * k_w, axis=1) * mask
                patch = patch.reshape((batch_sz, k_h, k_w, in_c))
                grad_wrt_x[:, r_start:r_start + k_h, c_start:c_start + k_w, :] += patch

        grad_wrt_x = grad_wrt_x[:, pad_h[0]:in_h - pad_h[1], pad_w[0]:in_w - pad_w[1], :]
        return grad_wrt_x

    def InitParams(self):
        x = self.inbound[0]
        self.padding_width = padding(kernel_shape=self.pooling_size,
                                     padding_mode=self.padding)
        h_pad, w_pad = self.padding_width[1:3]
        if hasattr(self, 'input_shape'):
            input_channel = self.input_shape[-1]
            in_h, in_w = self.input_shape[0:2]
        else:
            # when x in {InputLayer, MaxPooling, Conv2D}
            in_h, in_w, input_channel = x.output_shape
        output_channel = input_channel
        shape = list(self.pooling_size)
        shape.append(input_channel)
        shape.append(output_channel)
        self.shape = shape

        in_h = in_h + h_pad[0] + h_pad[1]
        in_w = in_w + w_pad[0] + w_pad[1]
        out_h = (in_h - self.shape[0]) // self.strides[0] + 1
        out_w = (in_w - self.shape[1]) // self.strides[1] + 1

        self.output_shape = (out_h, out_w, output_channel)
        self.isInitialized = True


def padding(kernel_shape, padding_mode='same'):
    # Calculating how many pixels should we add to the inputs
    def get_padding_1d(k, mode):
        if mode == 'same':
            # The center of kernel locates the corner
            pads = k - 1
            half = pads // 2
            return (half, half) if pads % 2 == 0 else (half, half + 1)
        else:
            # Otherwise
            return 0, 0

    h_pad = get_padding_1d(kernel_shape[0], mode=padding_mode)
    w_pad = get_padding_1d(kernel_shape[1], mode=padding_mode)

    padding_width = (0, 0), h_pad, w_pad, (0, 0)
    return padding_width
