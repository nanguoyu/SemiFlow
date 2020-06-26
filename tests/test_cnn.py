"""
@File : test_cnn.py
@Author: Dong Wang
@Date : 2020/6/26
"""
from SemiFlow.layer import Dense, InputLayer, Conv2D, MaxPooling2D, Flatten
from SemiFlow.Model import Sequential
from SemiFlow.utils.dataset import mnist
import numpy as np


def test_cnn_mnist():
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
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1),
                     dtype='float32'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pooling_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', learning_rate=0.05)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_val, y_val))
    score = model.evaluate(x_test, y_test, verbose=0)
