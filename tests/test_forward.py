"""
@File : test_forward.py
@Author: Dong Wang
@Date : 2020/6/13
"""
import numpy as np
from SemiFlow.layer import Dense, InputLayer


def test_forward():
    print("\n")
    x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    print("x shape:", x.shape, "y shape:", y.shape)
    print("\n")
    print("x", x)
    print("y", y)
    print("\n")
    num_classes = 2
    batch_size = 1
    epochs = 20

    # Build network
    input0 = InputLayer(shape=2)
    dense1 = Dense(units=2, activation='relu', input_shape=(2,))
    input0.outbound.append(dense1)
    dense1.inbound.append(input0)

    # Init parameters of dense1
    dense1.name = 'dense1'
    kernel = np.array([[1., 2.], [2., 1.]])
    bias = np.array([1., 1.])
    dense1.params = {
        'kernel': kernel,
        'bias': bias}
    dense1.isInitialized = True
    print("kernel", kernel)
    print("bias", bias)
    # ForwardPropagation
    inputs = input0.ForwardPropagation(feed=x)
    print("inputs", inputs)
    f1 = dense1.ForwardPropagation()
    print("f1", f1)
    f1_expect = np.matmul(inputs, kernel) + bias
    print("expect f1", f1_expect)
