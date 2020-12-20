"""
@File : test_conv2d.py
@Author: Dong Wang
@Date : 2020/6/21
"""
from SemiFlow.utils.dataset import mnist
from SemiFlow.layer import Dense, Conv2D, InputLayer
from SemiFlow.Model import Sequential
import numpy as np


def test_conv2d_mnist():
    # Todo test_conv2d_mnist
    train_set, test_set = mnist(one_hot=True)

    x_train, y_train = train_set[0], train_set[1]
    x_test, y_test = test_set[0], test_set[1]

    x_train = x_train.reshape((-1, 28, 28, 1))

    x_test = x_test.reshape((-1, 28, 28, 1))


    num_classes = 10
    batch_size = 32
    epochs = 1

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(num_classes, activation='softmax'))
