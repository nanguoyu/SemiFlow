"""
@File : test_save.py
@Author: Dong Wang
@Date : 7/15/2021
"""
from SemiFlow.layer import Dense
from SemiFlow.Model import Sequential
import numpy as np
import json


def test_model_save():
    Asamples = np.random.multivariate_normal([6, 6], [[1, 0], [0, 1]], 200)
    Bsamples = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 200)

    x_train = np.vstack((Asamples, Bsamples))
    y_train = np.vstack((np.array([[0, 1]] * 200), np.array([[1, 0]] * 200)))
    print(x_train.shape, y_train.shape)
    num_classes = 2
    batch_size = 10
    epochs = 20

    model = Sequential()
    model.add(Dense(units=2, activation='relu', input_shape=(2,)))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='sgd', learning_rate=0.05)
    model.save("./test_model.npy")
    model.load("./test_model.npy")
    history1 = model.fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_data=(None, None),
                         validation_split=0.2)
    model.save("./test_model.npy")
    model.load("./test_model.npy")
    print("\n")
    history2 = model.fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_data=(None, None),
                         validation_split=0.2)
    weights = model.state_dict
    print(weights)
    print(json.dumps(weights))
