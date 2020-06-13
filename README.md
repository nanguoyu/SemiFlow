

# SemiFlow
[![Build Status](https://travis-ci.com/nanguoyu/SemiFlow.svg?branch=master)](https://travis-ci.com/nanguoyu/SemiFlow)
[![codecov](https://codecov.io/gh/nanguoyu/SemiFlow/branch/master/graph/badge.svg)](https://codecov.io/gh/nanguoyu/SemiFlow)


SemiFlow is a neural network framework for machine learning. There are two parts in this repository.
The first part is a tensorflow style deep learning framework. The second part is a Keras style
deep learning framework.

## Install

``` 
git https://github.com/nanguoyu/SemiFlow.git
cd SemiFlow
pip install .
```

## A Keras style part

### Quick start
> A classification model trained in MNIST.

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
   


### Features

Progress
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
    - [ ] Convolutional layer
    - [ ] Pooling layer
    - [ ] Stochastic gradient descent
- [ ] Big dataset support
    - [x] Train MNIST
- [ ] CUDA support
- [ ] Examples and other docs


## A Tensorflow style part.

### Features
- [x] computational graph
    - [x] feedforward
    - [x] numpy style operator
    - [x] compute gradient
- [x] Auto differentiate
- [ ] <del>Tensor support</del>

### Examples
- Regression a line: [Regression a line](tests/test_engine_compute_gradients.py)


### Blogs
 - [[SemiFlow 动手实现深度学习框架 00] 初步的计划](https://www.nanguoyu.com/semiflow-00)
    - Code: [A naive dense layer in a python file](./A%20naive%20example)
 - [[SemiFlow 动手实现深度学习框架 01] 从一个例子开始](https://www.nanguoyu.com/semiflow-01)


## Reference
- [The Supervised Machine Learning book(An upcoming textbook)](http://smlbook.org/)
- [simpleflow](https://github.com/PytLab/simpleflow)
