"""
@File : test_dataset.py
@Author: Dong Wang
@Date : 2020/6/9
"""
from SemiFlow.engine import backend
from SemiFlow.utils.dataset import mnist


def test_mnist():
    train_set, valid_set, test_set = mnist(one_hot=True)
    print(train_set[0].shape, train_set[1].shape)
    print(valid_set[0].shape, valid_set[1].shape)
    print(test_set[0].shape, test_set[1].shape)
