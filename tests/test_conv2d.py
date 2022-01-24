"""
@File : test_conv2d.py
@Author: Dong Wang
@Date : 2020/6/21
"""
from SemiFlow.utils.dataset import mnist
from SemiFlow.layer import Dense, Conv2D, InputLayer, MaxPooling2D, Flatten
from SemiFlow.Model import Sequential
import numpy as np


def test_conv2d_mnist():
    # Todo test_conv2d_mnist
    train_set, test_set = mnist(one_hot=True)

    x_train, y_train = train_set[0][:128], train_set[1][:128]
    x_test, y_test = test_set[0][:128], test_set[1][:128]

    x_train = x_train.reshape((-1, 28, 28, 1))

    x_test = x_test.reshape((-1, 28, 28, 1))

    num_classes = 10
    batch_size = 4
    epochs = 2

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pooling_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', learning_rate=0.05)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(None, None))

    score = model.evaluate(x_test, y_test, verbose=0)
