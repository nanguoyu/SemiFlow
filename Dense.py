"""
@File : Dense.py
@Author: Dong Wang
@Date : 2020/3/23
"""
import numpy as np

np.random.seed(seed=6)


class Dense:
    def __init__(self):
        # Firstly, we define a naive network.
        # Input tensor is 2-D. There is a hidden layer with 4 units. There is 1 unit in output layer.

        # offset in first layer
        self._b11 = np.random.rand(1)
        self._b12 = np.random.rand(1)
        self._b13 = np.random.rand(1)
        self._b14 = np.random.rand(1)
        # x1 to 4 hidden nodes
        self._w111 = np.random.rand(1)
        self._w112 = np.random.rand(1)
        self._w113 = np.random.rand(1)
        self._w114 = np.random.rand(1)
        # x2 to 4 hidden nodes
        self._w121 = np.random.rand(1)
        self._w122 = np.random.rand(1)
        self._w123 = np.random.rand(1)
        self._w124 = np.random.rand(1)
        # 4 hidden nodes to output node
        self._w211 = np.random.rand(1)
        self._w221 = np.random.rand(1)
        self._w231 = np.random.rand(1)
        self._w241 = np.random.rand(1)
        # offset in second layer
        self._b2 = np.random.rand(1)
        self._logits = np.random.rand(1)
        self._h1 = np.random.rand(1)
        self._h2 = np.random.rand(1)
        self._h3 = np.random.rand(1)
        self._h4 = np.random.rand(1)
        self._o1 = np.random.rand(1)

    def _feedforward(self, x):
        x1 = x[0]
        x2 = x[1]
        h1 = max(0, self._w111 * x1 + self._w121 * x2 + self._b11)
        h2 = max(0, self._w112 * x1 + self._w122 * x2 + self._b12)
        h3 = max(0, self._w113 * x1 + self._w123 * x2 + self._b13)
        h4 = max(0, self._w114 * x1 + self._w124 * x2 + self._b14)
        logits = self._b2 + self._w211 * h1 + self._w221 * h2 + self._w231 * h3 + self._w241 * h4
        pred = sigmoid(logits)
        return pred

    def _ForwardPropagation(self, x):
        self._x1 = x[0]
        self._x2 = x[1]
        self._h1 = max(0, self._w111 * x[0] + self._w121 * x[1] + self._b11)
        self._h2 = max(0, self._w112 * x[0] + self._w122 * x[1] + self._b12)
        self._h3 = max(0, self._w113 * x[0] + self._w123 * x[1] + self._b13)
        self._h4 = max(0, self._w114 * x[0] + self._w124 * x[1] + self._b14)
        self._logits = self._b2 + self._w211 * self._h1 + self._w221 * self._h2 + self._w231 * self._h3 + self._w241 * self._h4
        self._pred = sigmoid(self._logits)

    def _BackPropagation(self, y_true, learning_rate):
        # TODO backPropagation

        # loss = - (y_true * np.log(self._pred) + (1 - y_true) * np.log(1 - self._pred))
        # d_L_d_Pred = y_true / self._pred + (y_true - 1) / (1 - self._pred)
        # d_pred_d_logits = deriv_sigmoid(self._logits)

        d_L_d_logits = self._pred - y_true

        d_L_d_W211 = d_L_d_logits * self._h1
        d_L_d_W221 = d_L_d_logits * self._h2
        d_L_d_W231 = d_L_d_logits * self._h3
        d_L_d_W241 = d_L_d_logits * self._h4
        d_L_d_b2 = d_L_d_logits

        d_L_d_h1 = d_L_d_logits * self._w211
        d_L_d_h2 = d_L_d_logits * self._w221
        d_L_d_h3 = d_L_d_logits * self._w231
        d_L_d_h4 = d_L_d_logits * self._w241

        d_h1_d_h1 = 1
        d_h1_d_w111 = d_h1_d_h1 * self._x1
        d_h1_d_w121 = d_h1_d_h1 * self._x2
        d_h1_d_b11 = d_h1_d_h1

        d_h2_d_h2 = 1
        d_h2_d_w112 = d_h2_d_h2 * self._x1
        d_h2_d_w122 = d_h2_d_h2 * self._x2
        d_h2_d_b12 = d_h2_d_h2

        d_h3_d_h3 = 1
        d_h3_d_w113 = d_h3_d_h3 * self._x1
        d_h3_d_w123 = d_h3_d_h3 * self._x2
        d_h3_d_b13 = d_h3_d_h3

        d_h4_d_h4 = 1
        d_h4_d_w114 = d_h4_d_h4 * self._x1
        d_h4_d_w124 = d_h4_d_h4 * self._x2
        d_h4_d_b14 = d_h4_d_h4

        # Update parameters

        self._w111 -= learning_rate * d_L_d_h1 * d_h1_d_w111
        self._w121 -= learning_rate * d_L_d_h1 * d_h1_d_w121
        self._w112 -= learning_rate * d_L_d_h2 * d_h2_d_w112
        self._w122 -= learning_rate * d_L_d_h2 * d_h2_d_w122
        self._w113 -= learning_rate * d_L_d_h3 * d_h3_d_w113
        self._w123 -= learning_rate * d_L_d_h3 * d_h3_d_w123
        self._w114 -= learning_rate * d_L_d_h4 * d_h4_d_w114
        self._w124 -= learning_rate * d_L_d_h4 * d_h4_d_w124

        self._b11 -= learning_rate * d_L_d_h1 * d_h1_d_b11
        self._b12 -= learning_rate * d_L_d_h2 * d_h2_d_b12
        self._b13 -= learning_rate * d_L_d_h3 * d_h3_d_b13
        self._b14 -= learning_rate * d_L_d_h4 * d_h4_d_b14
        self._w211 -= learning_rate * d_L_d_W211
        self._w221 -= learning_rate * d_L_d_W221
        self._w231 -= learning_rate * d_L_d_W231
        self._w241 -= learning_rate * d_L_d_W241
        self._b2 -= learning_rate * d_L_d_b2

    def train(self, x_train, y_train, learning_rate=0.01, epochs=100):
        for epoch in range(epochs):
            for i, x in enumerate(x_train):
                self._ForwardPropagation(x)
                self._BackPropagation(y_train[i], learning_rate)

            if epoch % 10 == 0:
                preds = np.zeros(y_train.shape[0])
                for i in range(y_train.shape[0]):
                    preds[i] = self._feedforward(x_train[i])

                loss = binary_cross_entropy(y_train, preds)
                print("[Epoch %2d] loss : %.3f" % (epoch, loss))

    def evaluate(self, x_test, y_test):
        preds = np.zeros(y_test.shape[0])
        for i in range(y_test.shape[0]):
            preds[i] = self._feedforward(x_test[i])
        num = 0
        for i in range(x_test.shape[0]):
            y = 0
            if 0.5 <= preds[i] < 1:
                y = 1
            elif preds[i] >= 0:
                y = 0
            if y_test[i] == y:
                num += 1
        accuracy = num/x_test.shape[0]
        print("[Train accuracy: %.3f]" % accuracy)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLu(x):
    return max(0, x)


def binary_cross_entropy(y, p):
    assert y.shape[0] == p.shape[0], "wrong shape"
    loss = 0
    for i in range(y.shape[0]):
        loss += - (y[i] * np.log(p[i]) + (1 - y[i]) * np.log(1 - p[i]))
    return loss / y.shape[0]


def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)
