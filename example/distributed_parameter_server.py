"""
@File : distributed_parameter_server.py
@Author: Dong Wang
@Date : 7/21/2021
"""

from threading import Thread
from socketserver import TCPServer
import SemiFlow
from SemiFlow.layer import Dense
from SemiFlow.Model import Sequential
import numpy as np
from SemiFlow.utils.distributed_tools import ParameterServer, Handler


def global_model():
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
    return model


if __name__ == '__main__':

    my_model = global_model()
    Handler.global_server = ParameterServer(my_model)
    NWORKERS = 3
    serv = TCPServer(('', 10086), Handler)
    for n in range(NWORKERS):
        t = Thread(target=serv.serve_forever)
        t.daemon = True
        t.start()
    serv.serve_forever()
