"""
@File : test_optimizer.py
@Author: Dong Wang
@Date : 2020/6/30
"""
from SemiFlow.layer import Dense
from SemiFlow.Model import Sequential
import numpy as np


def model(opt):
    Asamples = np.random.multivariate_normal([6, 6], [[1, 0], [0, 1]], 200)
    Bsamples = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 200)

    x_train = np.vstack((Asamples, Bsamples))
    y_train = np.vstack((np.array([[0, 1]] * 200), np.array([[1, 0]] * 200)))
    num_classes = 2
    batch_size = 10
    epochs = 10

    model = Sequential()
    model.add(Dense(units=2, activation='relu', input_shape=(2,)))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=opt, learning_rate=0.05)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(None, None))


def test_sgd():
    model(opt='sgd')


def test_momentum():
    model(opt='momentum')


def test_rmsprop():
    model(opt='RMSProp')
