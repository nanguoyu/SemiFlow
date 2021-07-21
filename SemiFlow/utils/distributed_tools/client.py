"""
@File : client.py
@Author: Dong Wang
@Date : 7/20/2021
"""

import json
import threading
from threading import Thread
import copy
import warnings
from socketserver import BaseRequestHandler, TCPServer
import SemiFlow
from SemiFlow.layer import Dense
from SemiFlow.Model import Sequential
import numpy as np
from .message import Message
import socket


class ParameterClient:

    def __init__(self, model):
        self._model = model
        self._value_lock = threading.Lock()

    def get_model(self):
        with self._value_lock:
            if isinstance(self._model, SemiFlow.Model.Model):
                current_model = self._model.state_dict
                assert isinstance(current_model, dict)
                weights = json.dumps(current_model).encode()
                return weights
            else:
                return None

    def get_model_parameters(self):
        with self._value_lock:
            if isinstance(self._model, SemiFlow.Model.Model):
                current_model = self._model.parameters
                assert isinstance(current_model, dict)
                return current_model
            else:
                return None

    def update_model(self, payload):
        payload_str = payload.decode()
        print(payload_str)
        new_weights = json.loads(payload_str)
        assert isinstance(new_weights, dict)

        if isinstance(self._model, SemiFlow.Model.Model):
            current_model = self._model.parameters

            for layer_name in current_model.keys():
                parameters_each_layer = current_model[layer_name].keys()
                for para in parameters_each_layer:
                    new_para = SemiFlow.backend.array(new_weights[layer_name][para])
                    assert current_model[layer_name][para].shape == new_para.shape
                    current_model[layer_name][para] = new_para

            with self._value_lock:
                self._model.parameters = current_model
        else:
            warnings.warn("no supported models")

    def model_increment(self, model1, model2):
        assert model1 is not None and model2 is not None
        if isinstance(self._model, SemiFlow.Model.Model):
            assert isinstance(model1, dict)
            assert isinstance(model2, dict)
            weights_update = copy.deepcopy(model2)
            for layer_name in model2.keys():
                for para_name in model2[layer_name].keys():
                    weights_update[layer_name][para_name] = model2[layer_name][para_name] - model1[layer_name][
                        para_name]
                    weights_update[layer_name][para_name] = weights_update[layer_name][para_name].tolist()
            return weights_update
        else:
            return None

    @property
    def model(self):
        with self._value_lock:
            if isinstance(self._model, SemiFlow.Model.Model):
                return self._model
            else:
                return None


if __name__ == '__main__':
    def client_model():
        num_classes = 2
        learning_rate = 0.05

        model = Sequential()
        model.add(Dense(units=2, activation='relu', input_shape=(2,)))
        model.add(Dense(units=num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer='sgd', learning_rate=learning_rate)

        return model


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
        # time.sleep(5)
