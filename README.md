

# SemiFlow
[![Build Status](https://travis-ci.com/nanguoyu/SemiFlow.svg?branch=master)](https://travis-ci.com/nanguoyu/SemiFlow)
[![codecov](https://codecov.io/gh/nanguoyu/SemiFlow/branch/master/graph/badge.svg)](https://codecov.io/gh/nanguoyu/SemiFlow)


SemiFlow is a neural network framework for machine learning.

I am going to implement a neural network framework in about a month. :)

## Install

``` 
git https://github.com/nanguoyu/SemiFlow.git
cd SemiFlow
pip install .
```

## A naive example.
Blogs
 - [[SemiFlow 动手实现深度学习框架 00] 初步的计划](https://www.nanguoyu.com/semiflow-00)
    - Code: [A naive dense layer](./A%20naive%20example)
 - [[SemiFlow 动手实现深度学习框架 01] 从一个例子开始](https://www.nanguoyu.com/semiflow-01)

Code: [A naive dense layer](./A%20naive%20example)

## Tensorflow style design
Progress
- [x] computational graph
    - [x] feedforward
    - [x] numpy style operator
    - [x] compute gradient
- [ ] Tensor support
- [x] Auto differentiate

## Sequential Model 
> In the development, I find it is hard for me to implement a deep learning 
> framework support functional model like the Pytorch. As a result, I changed
> the plan to develop a sequential model after finish tensorflow style design.

>Progress
- [ ] Dense layer
- [ ] Model manager for training
- [ ] Optimizer
- [ ] Activation function
    - [x] ReLU
    - [x] Sigmoid
    - [x] tanh
## Complex layer
Progress
- [ ] Convolutional layer
- [ ] Pooling layer
- [ ] Stochastic gradient descent


## Big dataset support
- [ ] Train MNIST

## Advance 
- [ ] CUDA support

## Support Docs
- [ ] Examples and other docs

## Reference
- [The Supervised Machine Learning book(An upcoming textbook)](http://smlbook.org/)
- [simpleflow](https://github.com/PytLab/simpleflow)
