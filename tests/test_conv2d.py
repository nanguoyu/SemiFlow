"""
@File : test_conv2d.py
@Author: Dong Wang
@Date : 2020/6/21
"""
from SemiFlow.utils.dataset import mnist
from SemiFlow.layer import Dense, Conv2D, InputLayer
from SemiFlow.Model import Sequential
import numpy as np


# import tensorflow as tf


def test_conv2d_layer():
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


def test_conv2d_mnist():
    train_set, valid_set, test_set = mnist(one_hot=True)

    x_train, y_train = train_set[0], train_set[1]
    x_test, y_test = test_set[0], test_set[1]
    x_val, y_val = valid_set[0], valid_set[1]

    x_train = x_train.reshape((-1, 28, 28, 1))

    x_test = x_test.reshape((-1, 28, 28, 1))

    x_val = x_val.reshape((-1, 28, 28, 1))

    num_classes = 10
    batch_size = 32
    epochs = 1

    model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=(1, 28, 28)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(num_classes, activation='softmax'))
