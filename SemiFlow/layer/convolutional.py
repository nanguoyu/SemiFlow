"""
@File : convolutional.py
@Author: Dong Wang
@Date : 2020/6/21
"""
from ..engine.core import backend
from .core import Layer
from .. import activations
from .. import initializers


class Conv2D(Layer):
    def __init__(self, filters,
                 kernel_size=None,
                 strides=(1, 1),
                 padding='SAME',
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=None,
                 **kwargs):
        """
        Args:
            filters:Integer, the dimensionality of the output space
            kernel_size: An integer or tuple/list of 2 integers, specifying the
                height and width of the 2D convolution window.
                Can be a single integer to specify the same value for all spatial dimensions.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution
                along the height and width.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.
            padding: one of `"valid"` or `"same"` (case-insensitive).
            activation: activation function
                If you don't specify anything linear activation is applied
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix
            bias_initializer: Initializer for the bias vector
        """
        super(Conv2D, self).__init__(**kwargs)
        self.filters = filters
        # kernel_size
        if isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        elif isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
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
        self.activation = activations.get(activation)
        self.original_activation_name = self.activation.name  # Sometime, the optimizer may optimize the activation
        # function
        self.kernel_initializer = initializers.get(kernel_initializer)
        if use_bias:
            self.bias_initializer = initializers.get(bias_initializer)
        self.padding_width = None
        self.output_shape = None
        self.isInitialized = False

    def BackwardPropagation(self, grad=None):
        """

        Args:
            grad: gradients from output layer

        Returns: gradients of this layer

        """
        k_h, k_w, in_c, out_c = self.shape
        s_h, s_w = self.strides
        batch_sz, in_h, in_w, in_c = self.inputs_padded_shape

        if grad is None:
            grad = backend.ones_like(self.output_value)

        grad = self.activation.BackwardPropagation(grads=grad)
        flat_grad = grad.reshape((-1, out_c))

        pad_h, pad_w = self.padding_width[1:3]

        x_padded = self.col
        grad_wrt_w = backend.matmul(x_padded.T, flat_grad)
        self.grads["kernel"] = grad_wrt_w.reshape(self.shape)
        self.grads["bias"] = backend.sum(flat_grad, axis=0)

        W = self.W
        grad_wrt_x_padded = backend.matmul(grad, W.T)
        grad_wrt_x = backend.zeros(self.inputs_padded_shape, dtype=self.dtype)
        for i, r in enumerate(range(0, in_h - k_h + 1, s_h)):
            for j, c in enumerate(range(0, in_w - k_w + 1, s_w)):
                patch = grad_wrt_x_padded[:, i, j, :]
                patch = patch.reshape((batch_sz, k_h, k_w, in_c))
                grad_wrt_x[:, r:r + k_h, c:c + k_w, :] += patch

        # cut off gradients of padding
        grad_wrt_x = grad_wrt_x[:, pad_h[0]:in_h - pad_h[1], pad_w[0]:in_w - pad_w[1], :]
        return grad_wrt_x

    def ForwardPropagation(self):
        assert self.isInitialized, "you should init_para"
        """ convolution operation
        A method to accelerate convolution from
        [1] Kumar Chellapilla, Sidd Puri, Patrice Simard. High Performance Convolutional Neural Networks
        for Document Processing. Tenth International Workshop on Frontiers in Handwriting Recognition,
        Université de Rennes 1, Oct 2006, La Baule (France). ffinria-00112631f
        
        Returns:output

        """
        x = self.inbound[0]
        inputs = x.output_value
        kernel = self.params['kernel']
        inputs_padded = backend.pad(inputs, pad_width=self.padding_width, mode="constant")
        batch_sz, in_h, in_w, _ = inputs_padded.shape
        out_h, out_w, out_c = self.output_shape

        # convolution => im2col
        # implementation：https://zhuanlan.zhihu.com/p/63974249
        col = im2col(inputs_padded, self.shape[0], self.shape[1], self.strides[0], self.strides[1], dtype=self.dtype)
        w = kernel.reshape(-1, kernel.shape[-1])
        z = backend.matmul(col, w)

        # reshape output
        z = z.reshape(batch_sz, z.shape[0] // batch_sz, self.shape[-1])
        z = z.reshape(batch_sz, out_h, out_w, self.shape[-1])
        if hasattr(self, 'bias_initializer'):
            b = self.params['bias']
            z += b
        self.output_value = self.activation.ForwardPropagation(z)
        self.inputs_padded_shape, self.col, self.W = inputs_padded.shape, col, w
        if hasattr(self, 'dtype'):
            self.output_value = self.output_value.astype(self.dtype)
        return self.output_value

    def InitParams(self):
        # print("Init ", self.name)
        # self.shape = (filter , filter , input_channel , output_channel)
        x = self.inbound[0]
        # assert isinstance(x, InputLayer) or isinstance(x, MaxPooling2D) or isinstance(x, Conv2D), x.name + "should
        # not " \ "be followed" \ " by " + self.name

        self.padding_width = padding(kernel_shape=self.kernel_size,
                                     padding_mode=self.padding)
        h_pad, w_pad = self.padding_width[1:3]
        output_channel = self.filters
        if hasattr(self, 'input_shape'):
            # When a Conv2D layer is the first layer.
            input_channel = self.input_shape[-1]
            in_h, in_w = self.input_shape[0:2]
        else:
            # when x in {MaxPooling, Conv2D}
            in_h, in_w, input_channel = x.output_shape
        # kernel shape
        shape = list(self.kernel_size)
        shape.append(input_channel)
        shape.append(output_channel)
        self.shape = shape
        # output_shape
        in_h = in_h + h_pad[0] + h_pad[1]
        in_w = in_w + w_pad[0] + w_pad[1]
        out_h = (in_h - self.shape[0]) // self.strides[0] + 1
        out_w = (in_w - self.shape[1]) // self.strides[1] + 1
        self.output_shape = (out_h, out_w, output_channel)
        # print(self.name+'.InitParams', self.shape)

        kernel = self.kernel_initializer(shape=self.shape)
        self.params = {
            'kernel': kernel}
        self.grads = {
            'kernel': backend.array([]),
        }
        if hasattr(self, 'bias_initializer'):
            bias = self.kernel_initializer(shape=[1]) * backend.ones([output_channel]).T
            self.params['bias'] = bias
            self.grads['bias'] = backend.array([])
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


def im2col(img, k_h, k_w, s_h, s_w, dtype='float32'):
    """Transform padded image into column matrix.
    This function is forked from https://github.com/borgwang/tinynn/blob/58c1e76e90a4ff1ee671707595df9fbc1f84f963/
    tinynn/core/layer.py#L628

    Args:
        img: padded inputs of shape (B, in_h, in_w, in_c)
        k_h: kernel height
        k_w: kernel width
        s_h: stride height
        s_w: stride width
        dtype: data type

    Returns: column matrix of shape (B*out_h*out_w, k_h*k_h*inc)

    """
    batch_sz, h, w, in_c = img.shape
    # calculate result feature map size
    out_h = (h - k_h) // s_h + 1
    out_w = (w - k_w) // s_w + 1
    # allocate space for column matrix
    col = backend.empty((batch_sz * out_h * out_w, k_h * k_w * in_c), dtype=dtype)
    # fill in the column matrix
    batch_span = out_w * out_h
    for r in range(out_h):
        r_start = r * s_h
        matrix_r = r * out_w
        for c in range(out_w):
            c_start = c * s_w
            patch = img[:, r_start: r_start + k_h, c_start: c_start + k_w, :]
            patch = patch.reshape(batch_sz, -1)
            col[matrix_r + c::batch_span, :] = patch
    return col
