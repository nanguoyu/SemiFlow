

# SemiFlow
[![Build Status](https://travis-ci.com/nanguoyu/SemiFlow.svg?branch=master)](https://travis-ci.com/nanguoyu/SemiFlow)
[![codecov](https://codecov.io/gh/nanguoyu/SemiFlow/branch/master/graph/badge.svg)](https://codecov.io/gh/nanguoyu/SemiFlow)

<img src="./.github/SemiFlow.png" alt="SemiFlow Logo" style="zoom:50%;" />

[English](README.md) | 中文

SemiFlow 是一个基于Numpy的支持自动求导的深度学习框架。

------

## 安装

``` 
git https://github.com/nanguoyu/SemiFlow.git
cd SemiFlow
pip install .
```

### 快速开始

> 一个MNIST分类模型

#### MNIST_MLP
> A classification MLP model trained in MNIST.

``` Python 
# Import SemiFlow

from SemiFlow.layer import Dense
from SemiFlow.Model import Sequential
from SemiFlow.utils.dataset import mnist
import numpy as np

# Prepare MNIST data.
train_set, valid_set, test_set = mnist(one_hot=True)

x_train, y_train = train_set[0], train_set[1]
x_test, y_test = test_set[0], test_set[1]
x_val, y_val = valid_set[0], valid_set[1]

# Specify trainig setting

num_classes = 10
batch_size = 128
epochs = 10

# Init a sequential model
model = Sequential()

# Add the first layer and specify the input shape
model.add(Dense(units=256, activation='relu', input_shape=(784,)))
# Add more layer
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Pring model structure
model.summary()

# Compile model and specify optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', learning_rate=0.05)

# Train model
history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(None, None))
                
# Evaluate model in test data 
score = model.evaluate(x_test, y_test, verbose=0)

```

#### MNIST_CNN
> A classification CNN model trained in MNIST.

``` Python 
# Import SemiFlow

from SemiFlow.layer import Dense
from SemiFlow.Model import Sequential
from SemiFlow.utils.dataset import mnist
import numpy as np

# Prepare MNIST data.
train_set, valid_set, test_set = mnist(one_hot=True)

x_train, y_train = train_set[0], train_set[1]
x_test, y_test = test_set[0], test_set[1]
x_val, y_val = valid_set[0], valid_set[1]

# Resize to height * width * channel
x_train = x_train.reshape((-1, 28, 28, 1))

x_test = x_test.reshape((-1, 28, 28, 1))

x_val = x_val.reshape((-1, 28, 28, 1))

# Specify trainig setting

num_classes = 10
batch_size = 128
epochs = 10

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

# Pring model structure
model.summary()

# Compile model and specify optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', learning_rate=0.05)

# Train model
history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(None, None))
                
# Evaluate model in test data 
score = model.evaluate(x_test, y_test, verbose=0)

```




### 特征
- [x] Dense layer
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
    - [x] Convolutional layer
    - [x] MaxPooling2D layer
    - [x] Flatten layer
- [x] Stochastic gradient descent
- [x] Big dataset support
    - [x] Train MNIST
    - [ ] cifar10
- [ ] CUDA support
- [ ] Examples and other docs


## 其他

>我们也开发了一个独立的计算图引擎在[A computation graph part](SemiFlow/engine)。它还支持自动微分和计算图。 
>在[Regression a line](tests/test_engine_compute_gradients.py)中有一个对回归问题的示例。
>请注意，这部分已过时，将不会更新。将来我们将引入一个新的支持计算图的model类。

### Features
- [x] computational graph
    - [x] feedforward
    - [x] numpy style operator
    - [x] compute gradient
- [x] Auto differentiate
- [ ] <del>Tensor support</del>

## Blogs
 - [[SemiFlow 动手实现深度学习框架 00] 初步的计划](https://www.nanguoyu.com/semiflow-00)
    - Code: [A naive dense layer in a python file](./A%20naive%20example)
 - [[SemiFlow 动手实现深度学习框架 01] 从一个例子开始](https://www.nanguoyu.com/semiflow-01)
 - [[SemiFlow 动手实现深度学习框架 02] 设计神经网络组件](https://www.nanguoyu.com/semiflow-02)

## Reference
- [The Supervised Machine Learning book(An upcoming textbook)](http://smlbook.org/)
- [simpleflow](https://github.com/PytLab/simpleflow)
