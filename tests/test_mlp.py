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
    train_set, valid_set, test_set = mnist(one_hot=True)

    x_train, y_train = train_set[0], train_set[1]
    x_test, y_test = test_set[0], test_set[1]
    x_val, y_val = valid_set[0], valid_set[1]

    num_classes = 10
    batch_size = 128
    epochs = 10

    model = Sequential()
    model.add(Dense(units=256, activation='relu', input_shape=(784,)))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', learning_rate=0.05)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(None, None))
    score = model.evaluate(x_test, y_test, verbose=0)


def test_simple_mlp():
    Asamples = np.random.multivariate_normal([6, 6], [[1, 0], [0, 1]], 200)
    Bsamples = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 200)

    x_train = np.vstack((Asamples, Bsamples))
    y_train = np.vstack((np.array([[0, 1]] * 200), np.array([[1, 0]] * 200)))
    print(x_train.shape, y_train.shape)
    num_classes = 2
    batch_size = 10
    epochs = 5

    model = Sequential()
    model.add(Dense(units=2, activation='relu', input_shape=(2,)))
    model.add(Dense(units=2, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', learning_rate=0.05)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(None, None))


def test_single_layer():
    Asamples = np.random.multivariate_normal([6, 6], [[1, 0], [0, 1]], 200)
    Bsamples = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 200)
    Csamples = np.random.multivariate_normal([12, 12], [[1, 0], [0, 1]], 200)

    plt.figure()
    plt.plot(Asamples[:, 0], Asamples[:, 1], 'r.')
    plt.plot(Bsamples[:, 0], Bsamples[:, 1], 'b.')
    plt.plot(Csamples[:, 0], Csamples[:, 1], 'g.')
    plt.show()

    x_train = np.vstack((Asamples, Bsamples, Csamples))
    y_train = np.vstack((np.array([[0, 0, 1]] * 200), np.array([[0, 1, 0]] * 200), np.array([[1, 0, 0]] * 200)))
    print(x_train.shape, y_train.shape)
    num_classes = 3
    batch_size = 20
    epochs = 100

    model = Sequential()
    model.add(Dense(units=num_classes, activation='softmax', input_shape=(2,)))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', learning_rate=0.005)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(None, None))
