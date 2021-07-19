"""
@File : test_mlp.py
@Author: Dong Wang
@Date : 2020/4/30
"""

from SemiFlow.layer import Dense
from SemiFlow.Model import Sequential
from SemiFlow.utils.dataset import mnist
import numpy as np
import matplotlib.pyplot as plt


def test_mlp_mnist():
    train_set, test_set = mnist(one_hot=True)

    x_train, y_train = train_set[0], train_set[1]
    x_test, y_test = test_set[0], test_set[1]
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

    num_classes = 10
    batch_size = 32
    epochs = 1

    model = Sequential()
    model.add(Dense(units=256, activation='relu', input_shape=(784,)))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='momentum', learning_rate=0.05)
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)


def test_simple_mlp():
    Asamples = np.random.multivariate_normal([6, 6], [[1, 0], [0, 1]], 200)
    Bsamples = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 200)

    x_train = np.vstack((Asamples, Bsamples))
    y_train = np.vstack((np.array([[0, 1]] * 200), np.array([[1, 0]] * 200)))
    print(f'\n x_train.shape {x_train.shape}, y_train.shape {y_train.shape}')
    num_classes = 2
    batch_size = 10
    epochs = 100

    model = Sequential()
    model.add(Dense(units=2, activation='relu', input_shape=(2,)))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='sgd', learning_rate=0.05)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(None, None),
                        validation_split=0.2)


def test_single_layer():
    Asamples = np.random.multivariate_normal([6, 6], [[1, 0], [0, 1]], 200)
    Bsamples = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 200)
    Csamples = np.random.multivariate_normal([12, 12], [[1, 0], [0, 1]], 200)

    # plt.figure()
    # plt.plot(Asamples[:, 0], Asamples[:, 1], 'r.')
    # plt.plot(Bsamples[:, 0], Bsamples[:, 1], 'b.')
    # plt.plot(Csamples[:, 0], Csamples[:, 1], 'g.')
    # plt.show()

    x_train = np.vstack((Asamples, Bsamples, Csamples))
    y_train = np.vstack((np.array([[0, 0, 1]] * 200), np.array([[0, 1, 0]] * 200), np.array([[1, 0, 0]] * 200)))
    print(x_train.shape, y_train.shape)
    num_classes = 3
    batch_size = 20
    epochs = 100

    model = Sequential()
    model.add(Dense(units=num_classes, activation='softmax', input_shape=(2,)))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='sgd', learning_rate=0.005)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(None, None),
                        validation_split=0.2)
