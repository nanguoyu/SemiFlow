

# SemiFlow

[![Build Status](https://travis-ci.com/nanguoyu/SemiFlow.svg?branch=master)](https://travis-ci.com/nanguoyu/SemiFlow)
[![codecov](https://codecov.io/gh/nanguoyu/SemiFlow/branch/master/graph/badge.svg)](https://codecov.io/gh/nanguoyu/SemiFlow)
[![DOI](https://zenodo.org/badge/249524882.svg)](https://zenodo.org/badge/latestdoi/249524882)

<img src="./.github/SemiFlow.png" alt="SemiFlow Logo" style="zoom:50%;" />

English | [中文](README_CN.md)

SemiFlow is a deep learning framework with auto-differentiation, developing from Numpy.

## News!

> Jan 24, 2022. SemiFlow supports converting models from SemiFlow to [ONNX](https://onnx.ai/)

> July 21, 2021. SemiFlow is now supporting **Distributed deep learning**. The first
> parallel solution is Parameter Sever.

> July 2021. We are introducing **model.save** and **model.load** !

## Installation

``` 
git clone https://github.com/nanguoyu/SemiFlow.git
cd SemiFlow
pip install .
```

### Quick start

#### MNIST_MLP
> A classification model trained in MNIST.

``` Python 
# Import SemiFlow

from SemiFlow.layer import Dense
from SemiFlow.Model import Sequential
from SemiFlow.utils.dataset import mnist
import numpy as np

# Prepare MNIST data.
train_set, test_set = mnist(one_hot=True)

x_train, y_train = train_set[0][:128], train_set[1][:128]
x_test, y_test = test_set[0][:128], test_set[1][:128]
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

# Specify trainig setting

num_classes = 10
batch_size = 32
epochs = 30

# Init a sequential model
model = Sequential()

# Add the first layer and specify the input shape
model.add(Dense(units=256, activation='relu', input_shape=(784,)))
# Add more layer
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Print model structure
model.summary()

# Compile model and specify optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', learning_rate=0.05)

# Train model
history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_split=0.2,
                validation_data=(None, None))
                
# Evaluate model in test data 
score = model.evaluate(x_test, y_test, verbose=0)

```

#### MNIST_CNN
``` Python 
# Import SemiFlow

from SemiFlow.layer import Dense, Conv2D, Flatten, MaxPooling2D
from SemiFlow.Model import Sequential
from SemiFlow.utils.dataset import mnist
import numpy as np

# Prepare MNIST data.
train_set, test_set = mnist(one_hot=True)

x_train, y_train = train_set[0][:128], train_set[1][:128]
x_test, y_test = test_set[0][:128], test_set[1][:128]

# Resize to height * width * channel
x_train = x_train.reshape((-1, 28, 28, 1))

x_test = x_test.reshape((-1, 28, 28, 1))

# Specify trainig setting

num_classes = 10
batch_size = 32
epochs = 30

# Init a sequential model
model = Sequential()

# Add the first layer and specify the input shape
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1),
                 dtype='float32'))
# Add other Conv2D layer
model.add(Conv2D(64, (3, 3), activation='relu'))
# Add a MaxPooling2D layer
model.add(MaxPooling2D(pooling_size=(3, 3)))
# Add a Flatten layer
model.add(Flatten())
# Add a Dense layer
model.add(Dense(units=64, activation='relu'))
# Add another Dense layer as output layer
model.add(Dense(num_classes, activation='softmax'))

# Print model structure
model.summary()

# Compile model and specify optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', learning_rate=0.05)

# Train model
history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_split=0.2,
                validation_data=(None, None))
                
# Evaluate model in test data 
score = model.evaluate(x_test, y_test, verbose=0)

```

### Distributed Machine Learning

Distributed machine learning is now launched! SemiFLow now supports parameter server.

#### parameter server example

Client/Worker example code in [distributed_parameter_client.py](./example/distributed_parameter_client.py)

Server/Master example code in [distributed_parameter_server.py](./example/distributed_parameter_server.py)

### Features

- [x] Dense/Full-connected layer
- [x] Model manager for training
- [x] Optimizer
- [x] Activation function
  - [x] ReLU
  - [x] Sigmoid
  - [x] tanh
- [x] Loss
    - [x] mse
    - [x] mae
    - [x] bce
    - [x] ce
- [ ] Complex Layer
  - [x] Conv2D layer
  - [x] MaxPooling2D layer
  - [x] Flatten layer
  - [x] RNN layer
- [x] Stochastic gradient descent
- [x] Momentum
- [x] RMSProp
- [x] Big dataset support
  - [x] Train MNIST
  - [x] cifar10
- [x] Save model
- [x] Load model
- [x] Distributed machine learning
  - [x] Parameter Server
- [x] SemiFlow-ONNX
- [ ] CUDA support
- [ ] Examples and other docs


## Other

> There is an independent part in [A computation graph part](SemiFlow/engine). In this part, we develop
> a deep learning engine like Tensorflow. It also supports auto-differentiation and computation graph. There is
>an example for [Regression a line](tests/test_engine_compute_gradients.py). 
>This part is dated and will not be updated. We are going to introduce a sub-class of **Model** containing computation
>graph in the future.

### Features
- [x] computational graph
    - [x] feedforward
    - [x] numpy style operator
    - [x] compute gradient
- [x] Auto differentiate
- [ ] <del>Tensor support</del>

## Blogs
 - [[SemiFlow 动手实现深度学习框架 00] 初步的计划](https://www.nanguoyu.com/semiflow-00)
 - [[SemiFlow 动手实现深度学习框架 01] 从一个例子开始](https://www.nanguoyu.com/semiflow-01)
     - Code: [A naive dense layer in a python file](./NaiveExample)
 - [[SemiFlow 动手实现深度学习框架 02] 设计神经网络组件](https://www.nanguoyu.com/semiflow-02)

## Reference
- [simpleflow](https://github.com/PytLab/simpleflow)
- [tinynn](https://github.com/borgwang/tinynn/)
- [The Supervised Machine Learning book(An upcoming textbook)](http://smlbook.org/)

