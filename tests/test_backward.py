"""
@File : test_backward.py
@Author: Dong Wang
@Date : 2020/6/13
"""
import numpy as np
from SemiFlow.layer import Dense, InputLayer, Conv2D, MaxPooling2D, Flatten
from SemiFlow.losses import SoftmaxCategoricalCrossentropy, softmax_categorical_crossentropy


def test_mlp_backward():
    print("\n")
    print("\n")
    x = np.array([[1, 0], [1, 1], [0, 1]])
    y = np.array([[0., 1.], [1., 0.], [0, 1]])
    print("x shape:", x.shape, "y shape:", y.shape)
    print("\n")
    print("x", x)
    print("y", y)
    print("\n")
    # Build network
    input0 = InputLayer(shape=[2])
    dense1 = Dense(units=2, activation='linear', input_shape=(2,))
    loss2 = SoftmaxCategoricalCrossentropy()

    input0.outbound.append(dense1)
    dense1.inbound.append(input0)
    dense1.outbound.append(loss2)
    loss2.inbound.append(dense1)

    # Init parameters of dense1
    dense1.name = 'dense1'
    kernel = np.array([[1., 2.], [2., 1.]])
    bias = np.array([1., 1.])
    dense1.params = {
        'kernel': kernel,
        'bias': bias}
    dense1.grads = {
        'kernel': np.array([]),
        'bias': np.array([])
    }
    dense1.isInitialized = True
    print("kernel", kernel)
    print("bias", bias)
    # ForwardPropagation
    inputs = input0.ForwardPropagation(feed=x)
    print("inputs", inputs)
    f1 = dense1.ForwardPropagation()
    print("f1", f1)
    # BackwardPropagation
    logits = np.array([[-10., 10.], [10., -10.], [-10., 10.]])
    loss2.y_true = np.array([[0., 1.], [1., 0.], [0., 1.]])
    loss2.input_value = logits
    loss2.output_value = softmax_categorical_crossentropy(y_true=loss2.y_true,
                                                          logits=logits)
    print("loss2.output_value", loss2.output_value)
    grad_loss = loss2.BackwardPropagation(grads=None)
    print("grad_loss", grad_loss)
    grad_dense = dense1.BackwardPropagation(grad=grad_loss)
    print("grad_dense", grad_dense)


def test_conv2d_backward():
    conv1 = Conv2D(32, kernel_size=(3, 3),
                   activation='linear',
                   input_shape=(5, 5, 1),
                   use_bias=False, name='conv1',
                   dtype='float64', )
    conv1.InitParams()
    assert list(conv1.shape) == [3, 3, 1, 32]
    input0 = InputLayer(shape=[5, 5, 1], name='input0', dtype='float64')
    input0.outbound.append(conv1)
    conv1.inbound.append(input0)
    x = np.ones([2, 5, 5, 1])
    inputs = input0.ForwardPropagation(feed=x)
    print("\n")
    assert list(inputs.shape) == [2, 5, 5, 1]
    c1 = conv1.ForwardPropagation()
    assert list(c1.shape) == [2, 5, 5, 32]
    grad_wrt_x = conv1.BackwardPropagation()
    # Todo check the value
    assert list(grad_wrt_x.shape) == [2, 5, 5, 1]


def test_maxpooling_2d_backward():
    Pooling1 = MaxPooling2D(pooling_size=(3, 3),
                            padding='SAME',
                            input_shape=(5, 5, 2),
                            name='conv1',
                            dtype='float32')

    input0 = InputLayer(shape=[5, 5, 2], name='input0', dtype='float32')
    input0.outbound.append(Pooling1)
    Pooling1.inbound.append(input0)
    x = np.ones([2, 5, 5, 2])
    x[0, :, :, 0][0][0] = 100
    inputs = input0.ForwardPropagation(feed=x)
    print("\n")
    print("inputs.shape", inputs.shape)
    c1 = Pooling1.ForwardPropagation()
    print("c1.shape", c1.shape)
    assert inputs.shape == c1.shape
    check1 = c1[0, :, :, 0]
    assert check1[0, 0] == 100
    assert check1[0, 1] == 100
    assert check1[1, 0] == 100
    assert check1[1, 1] == 100
    grad_wrt_x = Pooling1.BackwardPropagation()
    print("grad.shape", grad_wrt_x.shape)
    assert list(grad_wrt_x.shape) == [2, 5, 5, 2]
    # Todo Check the value
    print(grad_wrt_x[0, :, :, 0])


def test_flatten_forward():
    flatten = Flatten(name='flatten')
    input0 = InputLayer(shape=[3, 2], name='input0', dtype='float32')
    input0.outbound.append(flatten)
    flatten.inbound.append(input0)
    x = np.array([[[1, 1], [2, 2], [4, 4]], [[3, 3], [5, 5], [6, 6]]])
    i1 = input0.ForwardPropagation(feed=x)
    f1 = flatten.ForwardPropagation()
    grad = flatten.BackwardPropagation()
    assert grad.shape[1:] == (3, 2)
