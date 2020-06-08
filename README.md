

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


## A Keras style part
> In the development, I find it is hard for me to implement a deep learning 
> framework support functional model like the Pytorch. As a result, I changed
> the plan to develop a sequential model after finish tensorflow style design.

### Features

Progress
- [x] Dense layer
- [ ] Model manager for training
- [ ] Optimizer
- [ ] Activation function
    - [x] ReLU
    - [x] Sigmoid
    - [x] tanh
- [ ] Loss
    - [x] mse
    - [x] mae
    - [x] bce
    - [x] ce
- [ ] Complex Layer
    - [ ] Convolutional layer
    - [ ] Pooling layer
    - [ ] Stochastic gradient descent
- [ ] Big dataset support
    - [ ] Train MNIST
- [ ] CUDA support
- [ ] Examples and other docs

## Reference
- [The Supervised Machine Learning book(An upcoming textbook)](http://smlbook.org/)
- [simpleflow](https://github.com/PytLab/simpleflow)
