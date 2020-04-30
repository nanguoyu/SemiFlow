"""
@File : tests.py
@Author: Dong Wang
@Date : 2020/3/23
"""

# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from Dense import Dense

Asamples = np.random.multivariate_normal([6, 6], [[1, 0], [0, 1]], 200)
Bsamples = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 200)
plt.figure()
plt.plot(Asamples[:, 0], Asamples[:, 1], 'r.')
plt.plot(Bsamples[:, 0], Bsamples[:, 1], 'b.')
plt.show()

x_train = np.vstack((Asamples, Bsamples))
y_train = np.append(np.ones(200), np.zeros(200))

# print(x_train[0])

nn = Dense()

nn.fit(x_train=x_train, y_train=y_train, learning_rate=0.01, epochs=110)

nn.evaluate(x_train,y_train)