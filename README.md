

# SemiFlow

SemiFlow is a neural network framework for machine learning.

I am going to implement a neural network framework in a about month. :)

> SemiFlow是一个从零(严格来说是numpy)开始的神经网络框架的。我已经使用Tensorflow和keras好些年，
并且最近完成了"Statistical machine learning", "Artificial intelligence","Natural computation 
for machine learning"。我对自己动手实现深度学习框架很感兴趣。SemiFlow就是用来练手的。


## A naive example.
Blogs
 - [[SemiFlow 动手实现深度学习框架 00] 初步的计划](https://www.nanguoyu.com/semiflow-00)
    - Code: [A naive dense layer](./A%20naive%20example)
 - [[SemiFlow 动手实现深度学习框架 01] 从一个例子开始](https://www.nanguoyu.com/semiflow-01)

Code: [A naive dense layer](./A%20naive%20example)

Progress
- [x] forward propagation
- [x] backward propagation
- [x] gradient based method
- [x] ReLU
- [x] Sigmoid

## Tensorflow style design
Progress
- [x] computational graph
    - [x] feedforward
    - [x] numpy style operator
    - [x] compute gradient
- [ ] Tensor support
- [x] Auto differentiate

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
