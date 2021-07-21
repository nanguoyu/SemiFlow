"""
@File : server.py
@Author: Dong Wang
@Date : 7/20/2021
"""
import json
import threading
from threading import Thread
import warnings
from socketserver import BaseRequestHandler, TCPServer
import SemiFlow
from SemiFlow.layer import Dense
from SemiFlow.Model import Sequential
import numpy as np
from .message import Message


class ParameterServer:

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

    def update_model(self, payload):
        payload_str = payload.decode()
        print(payload_str)
        weight_updates = json.loads(payload_str)
        assert isinstance(weight_updates, dict)

        if isinstance(self._model, SemiFlow.Model.Model):
            current_model = self._model.parameters

            for layer_name in current_model.keys():
                parameters_each_layer = current_model[layer_name].keys()
                for para in parameters_each_layer:
                    para_updates = SemiFlow.backend.array(weight_updates[layer_name][para])
                    assert current_model[layer_name][para].shape == para_updates.shape
                    current_model[layer_name][para] = current_model[layer_name][para] + para_updates

            with self._value_lock:
                self._model.parameters = current_model
        else:
            warnings.warn("no supported models")


class Handler(BaseRequestHandler):
    global_server = None

    def handle(self):
        print('Got connection from', self.client_address)
        conn = self.request
        m = Message()
        while True:
            print("?<")
            header_len = m.read_msg_header(conn)
            if not header_len:
                break
            action, data_len, data_name = m.read_msg(conn, header_len)
            if action == "get model":
                print("Try to send the global model")
                current_model = Handler.global_server.get_model()
                m.send_data(conn, current_model, 'global model', 'send model')
                print("send model done")
            elif action == "update model":
                print("Try to update the global model")
                updates = m.read_data(conn, data_len, data_name)
                Handler.global_server.update_model(updates)
            else:
                break
            print("?>")


if __name__ == '__main__':

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


    my_model = global_model()
    Handler.global_server = ParameterServer(my_model)
    NWORKERS = 3
    serv = TCPServer(('', 10086), Handler)
    for n in range(NWORKERS):
        t = Thread(target=serv.serve_forever)
        t.daemon = True
        t.start()
    serv.serve_forever()
