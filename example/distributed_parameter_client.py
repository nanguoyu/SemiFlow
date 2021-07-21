"""
@File : distributed_parameter_client.py
@Author: Dong Wang
@Date : 7/21/2021
"""
import json
import copy
from SemiFlow.layer import Dense
from SemiFlow.Model import Sequential
import numpy as np
from SemiFlow.utils.distributed_tools import Message
import socket
from SemiFlow.utils.distributed_tools import ParameterClient


def client_model():
    num_classes = 2
    learning_rate = 0.05

    model = Sequential()
    model.add(Dense(units=2, activation='relu', input_shape=(2,)))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='sgd', learning_rate=learning_rate)

    return model


if __name__ == '__main__':
    # Data
    Asamples = np.random.multivariate_normal([6, 6], [[1, 0], [0, 1]], 200)
    Bsamples = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 200)

    x_train = np.vstack((Asamples, Bsamples))
    y_train = np.vstack((np.array([[0, 1]] * 200), np.array([[1, 0]] * 200)))
    print(f'\n x_train.shape {x_train.shape}, y_train.shape {y_train.shape}')
    num_classes = 2
    batch_size = 10
    epochs = 10

    # Build model
    my_model = client_model()
    parameter_client = ParameterClient(my_model)

    # Build client socket
    server_address = ('localhost', 10086)
    client_socket = socket.socket()
    client_socket.connect(server_address)
    m = Message()

    # Train model
    print("Train model 1st time")
    history1 = parameter_client.model.fit(x=x_train,
                                          y=y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          validation_data=(None, None),
                                          validation_split=0.2)
    weights1 = parameter_client.get_model_parameters()

    for i in range(10):

        # Train model
        print("Train model 2nd time")
        history2 = parameter_client.model.fit(x=x_train,
                                              y=y_train,
                                              batch_size=batch_size,
                                              epochs=epochs,
                                              verbose=1,
                                              validation_data=(None, None),
                                              validation_split=0.2)
        weights2 = parameter_client.get_model_parameters()

        weights_update = parameter_client.model_increment(weights1, weights2)
        weights1 = copy.deepcopy(weights2)

        weights_update_str = json.dumps(weights_update).encode()

        print(weights_update_str)
        print("Send update request to server")
        m.send_data(client_socket, weights_update_str, 'local model updates', 'update model')
        print("Send require request to server")
        m.send_data(client_socket, '', '', 'get model')
        header_len = m.read_msg_header(client_socket)
        if header_len is not None:
            action, data_len, data_name = m.read_msg(client_socket, header_len)
            new_model = m.read_data(client_socket, data_len, data_name)
            parameter_client.update_model(new_model)
        else:
            print("Failed to load header")
