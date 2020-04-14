"""
@File : test_compute_gradients.py
@Author: Dong Wang
@Date : 2020/4/12
"""
from SemiFlow.engine import *
from SemiFlow.activations import Sigmoid, ReLU
from SemiFlow.losses import ReduceSum
from SemiFlow.GradientDescentOptimizer import GradientDescentOptimizer
from SemiFlow.engine.session import Session
import numpy as np
import matplotlib.pyplot as plt

input_x = np.linspace(-1, 1, 100)
input_y = input_x * 3 + np.random.randn(input_x.shape[0]) * 0.5
plt.scatter(input_x, input_y)
plt.show()
# Placeholder for training data
x = Placeholder(name='x')
y_ = Placeholder(name='y_')
# 权重参数, 因为还未实现随机初始, 目前使用1.0来初始化
w = Variable(1.0, name='w')
# 阈值
b = Variable(0.0, name='b')
# 模型预测值
y = x * w + b
y.name = 'y'

d = y - y_
d.name = 'y-y_'

loss = ReduceSum(Square(d, name='Square'), name='loss')

train_op = GradientDescentOptimizer(learning_rate=0.005, name='GD').minimize(loss)
# 训练
feed_dict = {x: input_x, y_: input_y}

with Session() as sess:
    for step in range(20):
        loss_value = sess.run(loss, feed_dict=feed_dict)
        mse = loss_value / len(input_x)
        # 优化一步
        print('step: {}, loss: {}, mse: {}'.format(step, loss_value, mse))
        sess.run(train_op, feed_dict)
    # 训练后的参数值
    w_value = sess.run(w, feed_dict=feed_dict)
    b_value = sess.run(b, feed_dict=feed_dict)
    print('w: {}, b: {}'.format(w_value, b_value))
