"""
@File : test_forward.py
@Author: Dong Wang
@Date : 2020/6/13
"""
import numpy as np
from SemiFlow.layer import Dense, Conv2D, InputLayer, MaxPooling2D, Flatten


def test_mlp_forward():
    print("\n")
    x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    print("x shape:", x.shape, "y shape:", y.shape)
    print("\n")
    print("x", x)
    print("y", y)
    print("\n")
    num_classes = 2
    batch_size = 1
    epochs = 20

    # Build network
    input0 = InputLayer(shape=[2])
    dense1 = Dense(units=2, activation='relu', input_shape=[2])
    input0.outbound.append(dense1)
    dense1.inbound.append(input0)

    # Init parameters of dense1
    dense1.name = 'dense1'
    kernel = np.array([[1., 2.], [2., 1.]])
    bias = np.array([1., 1.])
    dense1.params = {
        'kernel': kernel,
        'bias': bias}
    dense1.isInitialized = True
    print("kernel", kernel)
    print("bias", bias)
    # ForwardPropagation
    inputs = input0.ForwardPropagation(feed=x)
    print("inputs", inputs)
    f1 = dense1.ForwardPropagation()
    print("f1", f1)
    f1_expect = np.matmul(inputs, kernel) + bias
    print("expect f1", f1_expect)


def test_conv2d_forward():
    conv1 = Conv2D(32, kernel_size=(3, 3),
                   strides=1,
                   activation='linear',
                   input_shape=(5, 5, 1),
                   use_bias=False, name='conv1',
                   dtype='float64', )
    input0 = InputLayer(shape=[5, 5, 1], name='input0', dtype='float64')
    input0.outbound.append(conv1)
    conv1.inbound.append(input0)
    conv1.InitParams()
    assert list(conv1.shape) == [3, 3, 1, 32]
    x = np.ones([2, 5, 5, 1])
    inputs = input0.ForwardPropagation(feed=x)
    print("\n")
    assert list(inputs.shape) == [2, 5, 5, 1]
    c1 = conv1.ForwardPropagation()
    assert list(c1.shape) == [2, 5, 5, 32]
    assert np.all(c1.shape[1:] == conv1.output_shape)


def test_maxpooling_2d_forward():
    Pooling1 = MaxPooling2D(pooling_size=(3, 3),
                            padding='SAME',
                            input_shape=(5, 5, 2),
                            name='conv1',
                            dtype='float32')

    input0 = InputLayer(shape=[5, 5, 2], name='input0', dtype='float32')
    input0.outbound.append(Pooling1)
    Pooling1.inbound.append(input0)
    Pooling1.InitParams()
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


def test_flatten_forward():
    flatten = Flatten(name='flatten')
    input0 = InputLayer(shape=[3, 2], name='input0', dtype='float32')
    input0.outbound.append(flatten)
    flatten.inbound.append(input0)
    flatten.InitParams()
    x = np.array([[[1, 1], [2, 2], [4, 4]], [[3, 3], [5, 5], [6, 6]]])
    i1 = input0.ForwardPropagation(feed=x)
    f1 = flatten.ForwardPropagation()
    assert np.all(f1[0] == np.array([1, 1, 2, 2, 4, 4.], dtype='float32'))
    assert np.all(f1[1] == np.array([3, 3, 5, 5, 6, 6.], dtype='float32'))

    grad = flatten.BackwardPropagation()
    print("\n")
    assert grad.shape[1:] == (3, 2)
